// SPDX-License-Identifier: Apache-2.0
// SANKHYA - detecting block-angular structure (#525). See structure.hpp for what is looked for,
// how, and the references.

#include "decomp/structure.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "sankhya/sparse.hpp"

namespace sankhya::decomp {

const char* to_string(Linking linking) noexcept {
  return linking == Linking::kColumns ? "linking columns" : "linking rows";
}

namespace {

constexpr std::size_t at(Index i) {
  return static_cast<std::size_t>(i);
}

/// Vertices joined by nets, both in compressed form from either side.
struct Hypergraph {
  Index vertices = 0;
  Index nets = 0;
  std::vector<Index> net_start, net_pin;        ///< net -> its vertices
  std::vector<Index> vertex_start, vertex_net;  ///< vertex -> its nets
  std::vector<double> weight;                   ///< per vertex: its number of nets, at least 1

  [[nodiscard]] Index pins(Index net) const {
    return net_start[at(net) + 1] - net_start[at(net)];
  }
};

/// `rows_are_vertices`: linking columns (each column a net over the rows it is in); otherwise
/// linking rows (each row a net over its columns). Entries that are exactly zero are not
/// structure.
Hypergraph build_hypergraph(const Model& model, bool rows_are_vertices) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  std::vector<Index> csc_start(at(n) + 1, 0), csc_row;
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      if (column.values[k] != 0.0) csc_row.push_back(column.rows[k]);
    }
    csc_start[at(j) + 1] = static_cast<Index>(csc_row.size());
  }
  std::vector<Index> csr_start(at(m) + 1, 0);
  for (const Index row : csc_row) ++csr_start[at(row) + 1];
  for (Index i = 0; i < m; ++i) csr_start[at(i) + 1] += csr_start[at(i)];
  std::vector<Index> csr_col(csc_row.size());
  {
    std::vector<Index> next(csr_start.begin(), csr_start.end() - 1);
    for (Index j = 0; j < n; ++j) {
      for (Index k = csc_start[at(j)]; k < csc_start[at(j) + 1]; ++k) {
        csr_col[at(next[at(csc_row[at(k)])]++)] = j;
      }
    }
  }
  Hypergraph g;
  if (rows_are_vertices) {
    g.vertices = m;
    g.nets = n;
    g.net_start = std::move(csc_start);
    g.net_pin = std::move(csc_row);
    g.vertex_start = std::move(csr_start);
    g.vertex_net = std::move(csr_col);
  } else {
    g.vertices = n;
    g.nets = m;
    g.net_start = std::move(csr_start);
    g.net_pin = std::move(csr_col);
    g.vertex_start = std::move(csc_start);
    g.vertex_net = std::move(csc_row);
  }
  g.weight.assign(at(g.vertices), 1.0);
  for (Index v = 0; v < g.vertices; ++v) {
    g.weight[at(v)] = std::max<double>(1.0, g.vertex_start[at(v) + 1] - g.vertex_start[at(v)]);
  }
  return g;
}

/// Balanced recursive bisection by net cut.
class Partitioner {
 public:
  Partitioner(const Hypergraph& graph, Index max_refine)
      : g_(graph),
        max_refine_(max_refine),
        net_cut_(at(graph.nets), 0),
        part_(at(graph.vertices), 0),
        seen_net_(at(graph.nets), 0),
        visited_vertex_(at(graph.vertices), 0),
        visited_net_(at(graph.nets), 0),
        side_(at(graph.vertices), 0),
        locked_(at(graph.vertices), 0),
        gain_(at(graph.vertices), 0),
        count_{std::vector<Index>(at(graph.nets), 0), std::vector<Index>(at(graph.nets), 0)} {}

  void run(Index parts) {
    std::vector<std::vector<Index>> groups(1);
    for (Index v = 0; v < g_.vertices; ++v) groups[0].push_back(v);
    while (static_cast<Index>(groups.size()) < parts) {
      std::size_t widest = groups.size();
      double widest_weight = -1.0;
      for (std::size_t k = 0; k < groups.size(); ++k) {
        if (groups[k].size() < 2) continue;
        double w = 0.0;
        for (const Index v : groups[k]) w += g_.weight[at(v)];
        if (w > widest_weight) {
          widest_weight = w;
          widest = k;
        }
      }
      if (widest == groups.size()) break;
      std::vector<Index> low, high;
      bisect(groups[widest], &low, &high);
      groups[widest] = std::move(low);
      groups.push_back(std::move(high));
    }
    for (std::size_t k = 0; k < groups.size(); ++k) {
      for (const Index v : groups[k]) part_[at(v)] = static_cast<Index>(k);
    }
  }

  [[nodiscard]] const std::vector<Index>& part() const { return part_; }
  [[nodiscard]] const std::vector<char>& net_cut() const { return net_cut_; }

 private:
  /// Splits `members` (one current part) in two, marks the nets the split cuts.
  void bisect(const std::vector<Index>& members, std::vector<Index>* low,
              std::vector<Index>* high) {
    ++clock_;
    nets_.clear();
    for (const Index v : members) {
      for (Index k = g_.vertex_start[at(v)]; k < g_.vertex_start[at(v) + 1]; ++k) {
        const Index e = g_.vertex_net[at(k)];
        // A net already cut is a linking item, paid for; one with a single vertex cannot be
        // cut.
        if (net_cut_[at(e)] != 0 || g_.pins(e) < 2 || seen_net_[at(e)] == clock_) continue;
        seen_net_[at(e)] = clock_;
        nets_.push_back(e);
      }
    }
    double total = 0.0;
    double heaviest = 0.0;
    for (const Index v : members) {
      total += g_.weight[at(v)];
      heaviest = std::max(heaviest, g_.weight[at(v)]);
    }

    // The starting order: a breadth-first sweep from the far end of a first sweep.
    const std::vector<Index> sweep = breadth_first(members, members.front());
    const std::vector<Index> order = breadth_first(members, sweep.back());
    double accumulated = 0.0;
    for (const Index v : order) {
      side_[at(v)] = accumulated < total / 2.0 ? 0 : 1;
      accumulated += g_.weight[at(v)];
    }
    if (static_cast<Index>(members.size()) <= max_refine_ && !nets_.empty()) {
      refine(members, total, heaviest);
    }

    for (const Index e : nets_) {
      Index on[2] = {0, 0};
      for (Index k = g_.net_start[at(e)]; k < g_.net_start[at(e) + 1]; ++k) {
        ++on[side_[at(g_.net_pin[at(k)])]];
      }
      if (on[0] > 0 && on[1] > 0) net_cut_[at(e)] = 1;
    }
    for (const Index v : members) (side_[at(v)] == 0 ? low : high)->push_back(v);
  }

  /// Every member in breadth-first order from `start`, then the members its component did not
  /// reach, each from the first of them not yet seen.
  std::vector<Index> breadth_first(const std::vector<Index>& members, Index start) {
    ++sweep_;
    std::vector<Index> order;
    order.reserve(members.size());
    const auto expand = [&](Index from) {
      std::size_t head = order.size();
      visited_vertex_[at(from)] = sweep_;
      order.push_back(from);
      while (head < order.size()) {
        const Index v = order[head++];
        for (Index k = g_.vertex_start[at(v)]; k < g_.vertex_start[at(v) + 1]; ++k) {
          const Index e = g_.vertex_net[at(k)];
          if (seen_net_[at(e)] != clock_ || visited_net_[at(e)] == sweep_) continue;
          visited_net_[at(e)] = sweep_;
          for (Index q = g_.net_start[at(e)]; q < g_.net_start[at(e) + 1]; ++q) {
            const Index u = g_.net_pin[at(q)];
            if (visited_vertex_[at(u)] == sweep_) continue;
            visited_vertex_[at(u)] = sweep_;
            order.push_back(u);
          }
        }
      }
    };
    expand(start);
    for (const Index v : members) {
      if (visited_vertex_[at(v)] != sweep_) expand(v);
    }
    return order;
  }

  /// Fiduccia-Mattheyses passes: move the unlocked vertex of greatest gain that keeps the
  /// sides balanced, lock it, and keep the prefix of moves with the least cut.
  void refine(const std::vector<Index>& members, double total, double heaviest) {
    const double slack = std::max(kImbalance * total, heaviest);
    const double lo = total / 2.0 - slack;
    const double hi = total / 2.0 + slack;
    for (int pass = 0; pass < kPasses; ++pass) {
      double weight[2] = {0.0, 0.0};
      for (const Index e : nets_) {
        count_[0][at(e)] = count_[1][at(e)] = 0;
        for (Index k = g_.net_start[at(e)]; k < g_.net_start[at(e) + 1]; ++k) {
          ++count_[side_[at(g_.net_pin[at(k)])]][at(e)];
        }
      }
      for (const Index v : members) {
        weight[side_[at(v)]] += g_.weight[at(v)];
        locked_[at(v)] = 0;
        gain_[at(v)] = 0;
        for (Index k = g_.vertex_start[at(v)]; k < g_.vertex_start[at(v) + 1]; ++k) {
          const Index e = g_.vertex_net[at(k)];
          if (seen_net_[at(e)] != clock_) continue;
          const int from = side_[at(v)];
          if (count_[from][at(e)] == 1) ++gain_[at(v)];
          if (count_[1 - from][at(e)] == 0) --gain_[at(v)];
        }
      }
      std::vector<Index> moves;
      Index cumulative = 0;
      Index best = 0;
      std::size_t best_length = 0;
      for (std::size_t step = 0; step < members.size(); ++step) {
        Index pick = -1;
        for (const Index v : members) {
          if (locked_[at(v)] != 0) continue;
          const int from = side_[at(v)];
          const double w = g_.weight[at(v)];
          if (weight[1 - from] + w > hi || weight[from] - w < lo) continue;
          if (pick < 0 || gain_[at(v)] > gain_[at(pick)]) pick = v;
        }
        if (pick < 0) break;
        cumulative += gain_[at(pick)];
        weight[side_[at(pick)]] -= g_.weight[at(pick)];
        weight[1 - side_[at(pick)]] += g_.weight[at(pick)];
        move(pick);
        moves.push_back(pick);
        if (cumulative > best) {
          best = cumulative;
          best_length = moves.size();
        }
      }
      for (std::size_t k = moves.size(); k > best_length; --k) {
        side_[at(moves[k - 1])] = 1 - side_[at(moves[k - 1])];
      }
      if (best <= 0) break;
    }
  }

  /// Move `v` to the other side, keeping the net counts and the gains of its neighbours exact
  /// (the four cases of Fiduccia and Mattheyses 1982), and lock it.
  void move(Index v) {
    const int from = side_[at(v)];
    const int to = 1 - from;
    for (Index k = g_.vertex_start[at(v)]; k < g_.vertex_start[at(v) + 1]; ++k) {
      const Index e = g_.vertex_net[at(k)];
      if (seen_net_[at(e)] != clock_) continue;
      const auto each_pin = [&](auto&& action) {
        for (Index q = g_.net_start[at(e)]; q < g_.net_start[at(e) + 1]; ++q) {
          const Index u = g_.net_pin[at(q)];
          if (u != v && locked_[at(u)] == 0) action(u);
        }
      };
      if (count_[to][at(e)] == 0) {
        each_pin([&](Index u) { ++gain_[at(u)]; });
      } else if (count_[to][at(e)] == 1) {
        each_pin([&](Index u) {
          if (side_[at(u)] == to) --gain_[at(u)];
        });
      }
      --count_[from][at(e)];
      ++count_[to][at(e)];
      if (count_[from][at(e)] == 0) {
        each_pin([&](Index u) { --gain_[at(u)]; });
      } else if (count_[from][at(e)] == 1) {
        each_pin([&](Index u) {
          if (side_[at(u)] == from) ++gain_[at(u)];
        });
      }
    }
    side_[at(v)] = to;
    locked_[at(v)] = 1;
  }

  static constexpr double kImbalance = 0.1;
  static constexpr int kPasses = 6;

  const Hypergraph& g_;
  Index max_refine_;
  std::vector<char> net_cut_;
  std::vector<Index> part_;
  std::vector<Index> seen_net_, visited_vertex_, visited_net_;
  std::vector<int> side_;
  std::vector<char> locked_;
  std::vector<Index> gain_;
  std::vector<Index> count_[2];
  std::vector<Index> nets_;
  Index clock_ = 0;
  Index sweep_ = 0;
};

/// The blocks and the linking part a finished partition makes of `model`'s matrix.
BlockStructure assemble(Linking kind, const Hypergraph& g, const Partitioner& partitioner) {
  BlockStructure s;
  s.kind = kind;
  // Vertices and nets, in the model's own terms: for linking columns the vertices are rows.
  std::vector<Index> vertex_block(at(g.vertices), kUnattached);
  std::vector<Index> net_block(at(g.nets), kUnattached);
  for (Index e = 0; e < g.nets; ++e) {
    if (g.pins(e) == 0) continue;
    net_block[at(e)] = partitioner.net_cut()[at(e)] != 0
                           ? kLinking
                           : partitioner.part()[at(g.net_pin[at(g.net_start[at(e)])])];
  }
  for (Index v = 0; v < g.vertices; ++v) {
    if (g.vertex_start[at(v) + 1] == g.vertex_start[at(v)]) continue;
    vertex_block[at(v)] = kLinking;
    for (Index k = g.vertex_start[at(v)]; k < g.vertex_start[at(v) + 1]; ++k) {
      // A vertex with a net that stayed inside one part is owned by that part.
      if (net_block[at(g.vertex_net[at(k)])] >= 0) {
        vertex_block[at(v)] = partitioner.part()[at(v)];
        break;
      }
    }
  }
  // Parts that kept no vertex are not blocks; number the rest consecutively.
  Index largest = -1;
  for (const Index b : vertex_block) largest = std::max(largest, b);
  std::vector<Index> renumber(at(largest + 1), -1);
  for (const Index b : vertex_block) {
    if (b >= 0) renumber[at(b)] = 0;
  }
  Index blocks = 0;
  for (Index& r : renumber) {
    if (r == 0) r = blocks++;
  }
  for (Index& b : vertex_block) {
    if (b >= 0) b = renumber[at(b)];
  }
  for (Index& b : net_block) {
    if (b >= 0) b = renumber[at(b)];
  }
  s.blocks = blocks;
  s.block_rows.assign(at(blocks), 0);
  s.block_columns.assign(at(blocks), 0);
  if (kind == Linking::kColumns) {
    s.row_block = std::move(vertex_block);
    s.col_block = std::move(net_block);
  } else {
    s.col_block = std::move(vertex_block);
    s.row_block = std::move(net_block);
  }
  for (const Index b : s.row_block) {
    if (b >= 0) ++s.block_rows[at(b)];
    if (b == kLinking) ++s.linking_rows;
  }
  for (const Index b : s.col_block) {
    if (b >= 0) ++s.block_columns[at(b)];
    if (b == kLinking) ++s.linking_columns;
  }
  return s;
}

}  // namespace

Detection detect_block_structure(const Model& model, Linking kind,
                                 const DetectionOptions& options) {
  Detection detection;
  const bool rows_are_vertices = kind == Linking::kColumns;
  const Hypergraph graph = build_hypergraph(model, rows_are_vertices);
  const std::vector<Index> tries =
      options.blocks > 0 ? std::vector<Index>{options.blocks} : std::vector<Index>{4, 2};
  for (const Index wanted : tries) {
    if (wanted < 2 || graph.vertices < 2 * wanted) {
      detection.description = fmt::format("too small to split into {} blocks", wanted);
      continue;
    }
    Partitioner partitioner(graph, options.max_refine_vertices);
    partitioner.run(wanted);
    BlockStructure s = assemble(kind, graph, partitioner);

    const Index linking = rows_are_vertices ? s.linking_columns : s.linking_rows;
    Index items = 0;  // the linking part's denominator
    for (Index e = 0; e < graph.nets; ++e) items += graph.pins(e) > 0 ? 1 : 0;
    s.linking_fraction =
        items > 0 ? static_cast<double>(linking) / static_cast<double>(items) : 0.0;
    const std::vector<Index>& owned = rows_are_vertices ? s.block_rows : s.block_columns;
    Index blocked = 0;
    Index widest = 0;
    for (const Index c : owned) {
      blocked += c;
      widest = std::max(widest, c);
    }
    s.largest_share =
        blocked > 0 ? static_cast<double>(widest) / static_cast<double>(blocked) : 1.0;
    const std::string found = fmt::format(
        "{} block(s) with {} {} ({:.1f}% of {}), the largest holding {:.0f}% of the {}",
        s.blocks, linking, rows_are_vertices ? "linking column(s)" : "linking row(s)",
        100.0 * s.linking_fraction,
        rows_are_vertices ? "the columns in some row" : "the rows with an entry",
        100.0 * s.largest_share, rows_are_vertices ? "blocked rows" : "blocked columns");
    if (s.blocks < 2) {
      detection.description = "the split left fewer than two blocks";
    } else if (s.linking_fraction > options.max_linking_fraction) {
      detection.description = fmt::format("{}: linking part above the {:.0f}% limit", found,
                                          100.0 * options.max_linking_fraction);
    } else if (s.largest_share > options.max_largest_share) {
      detection.description = fmt::format("{}: the blocks are lopsided (limit {:.0f}%)", found,
                                          100.0 * options.max_largest_share);
    } else {
      detection.accepted = true;
      detection.structure = std::move(s);
      detection.description = found;
      return detection;
    }
  }
  return detection;
}

}  // namespace sankhya::decomp
