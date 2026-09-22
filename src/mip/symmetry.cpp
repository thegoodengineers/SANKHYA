// SPDX-License-Identifier: Apache-2.0
// SANKHYA - formulation symmetry (#413). See symmetry.hpp for what is detected and why every
// generator is verified.

#include "symmetry.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

using Hash = std::uint64_t;

/// splitmix64's finaliser: cheap, and every bit of the input reaches every bit of the output.
[[nodiscard]] Hash mix(Hash h) noexcept {
  h += 0x9e3779b97f4a7c15ULL;
  h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
  h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
  return h ^ (h >> 31);
}

[[nodiscard]] Hash combine(Hash a, Hash b) noexcept {
  return mix(a ^ mix(b + 0x632be59bd9b4e019ULL));
}

/// The bits of a double, with -0.0 folded onto 0.0 so the two colour alike; infinities and
/// NaNs keep their own bit patterns, which is what a bound comparison would say too.
[[nodiscard]] Hash hash_double(double v) noexcept {
  if (v == 0.0) v = 0.0;
  return mix(std::bit_cast<std::uint64_t>(v));
}

constexpr Hash kColumnTag = 0x1d1e1f2021222324ULL;
constexpr Hash kRowTag = 0x25262728292a2b2cULL;
constexpr Hash kIndividualiseTag = 0x2d2e2f3031323334ULL;

/// The coloured bipartite graph: vertices 0..n-1 are columns, n..n+m-1 rows.
struct Graph {
  Index n = 0;
  Index m = 0;
  std::vector<std::vector<std::pair<Index, Hash>>> adjacent;  ///< (neighbour, edge colour)
  std::vector<Hash> initial;                                  ///< vertex colours
  [[nodiscard]] Index vertices() const noexcept { return n + m; }
};

Graph build_graph(const Model& model) {
  Graph g;
  g.n = model.num_cols();
  g.m = model.num_rows();
  const auto total = static_cast<std::size_t>(g.vertices());
  g.adjacent.resize(total);
  g.initial.resize(total);
  for (Index j = 0; j < g.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    Hash h = combine(kColumnTag, hash_double(model.col_cost[u]));
    h = combine(h, hash_double(model.col_lower[u]));
    h = combine(h, hash_double(model.col_upper[u]));
    h = combine(h, static_cast<Hash>(model.col_type[u] == VarType::kInteger ? 1 : 0));
    g.initial[u] = h;
  }
  for (Index i = 0; i < g.m; ++i) {
    const auto r = static_cast<std::size_t>(i);
    Hash h = combine(kRowTag, hash_double(model.row_lower[r]));
    h = combine(h, hash_double(model.row_upper[r]));
    g.initial[static_cast<std::size_t>(g.n + i)] = h;
  }
  for (Index j = 0; j < g.n; ++j) {
    const ColumnView view = model.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) {
      const Hash edge = hash_double(view.values[k]);
      const Index row_vertex = g.n + view.rows[k];
      g.adjacent[static_cast<std::size_t>(j)].emplace_back(row_vertex, edge);
      g.adjacent[static_cast<std::size_t>(row_vertex)].emplace_back(j, edge);
    }
  }
  return g;
}

[[nodiscard]] std::size_t distinct_colours(const std::vector<Hash>& colour) {
  std::unordered_set<Hash> seen(colour.begin(), colour.end());
  return seen.size();
}

/// One-dimensional colour refinement to a stable partition. A vertex's new colour is its
/// old colour together with the multiset of (edge colour, neighbour colour) around it,
/// folded commutatively so the order of the adjacency list does not matter. The colours are
/// canonical - functions of the structure alone - which is what lets two leaves of the
/// search be compared by colour. A hash collision can only merge classes, never split them:
/// it costs search effort, and the verification below is what decides correctness.
void refine(const Graph& g, std::vector<Hash>* colour, Count* refinements) {
  const auto total = static_cast<std::size_t>(g.vertices());
  std::vector<Hash> next(total);
  std::size_t classes = distinct_colours(*colour);
  for (;;) {
    ++*refinements;
    for (std::size_t v = 0; v < total; ++v) {
      Hash sum = 0;
      Hash xr = 0;
      for (const auto& [u, edge] : g.adjacent[v]) {
        const Hash term = mix(combine(edge, (*colour)[static_cast<std::size_t>(u)]));
        sum += term;
        xr ^= mix(term);
      }
      next[v] = combine(combine((*colour)[v], sum), xr);
    }
    const std::size_t after = distinct_colours(next);
    colour->swap(next);
    if (after <= classes) return;
    classes = after;
  }
}

/// The vertices of the non-singleton cell with the smallest colour, in index order; empty
/// when the partition is discrete.
[[nodiscard]] std::vector<Index> target_cell(const std::vector<Hash>& colour) {
  std::unordered_map<Hash, Index> count;
  for (const Hash c : colour) ++count[c];
  bool found = false;
  Hash best = 0;
  for (const auto& [c, k] : count) {
    if (k < 2) continue;
    if (!found || c < best) {
      best = c;
      found = true;
    }
  }
  std::vector<Index> cell;
  if (!found) return cell;
  for (std::size_t v = 0; v < colour.size(); ++v) {
    if (colour[v] == best) cell.push_back(static_cast<Index>(v));
  }
  return cell;
}

void individualise(std::vector<Hash>* colour, Index vertex, int depth) {
  Hash& c = (*colour)[static_cast<std::size_t>(vertex)];
  c = combine(combine(kIndividualiseTag, c), static_cast<Hash>(depth));
}

struct Search {
  const Graph& g;
  const Model& model;
  Count limit;
  SymmetryGroup out;

  Search(const Graph& graph, const Model& m, Count search_limit)
      : g(graph), model(m), limit(search_limit) {}

  [[nodiscard]] bool exhausted() const noexcept { return out.search_nodes >= limit; }

  /// Descend from `colour` by individualising the first vertex of the target cell each
  /// time, to a discrete partition or the budget. Returns false when the budget stopped it.
  bool leftmost_leaf(std::vector<Hash>* colour, int depth) {
    for (;;) {
      const std::vector<Index> cell = target_cell(*colour);
      if (cell.empty()) return true;
      if (exhausted()) return false;
      ++out.search_nodes;
      individualise(colour, cell.front(), depth);
      refine(g, colour, &out.refinements);
      ++depth;
    }
  }

  /// The bijection mapping each vertex of `from` to the vertex of `to` with the same
  /// colour, both discrete; false when a colour of one is missing from the other.
  [[nodiscard]] bool bijection(const std::vector<Hash>& from, const std::vector<Hash>& to,
                               Permutation* p) const {
    std::unordered_map<Hash, Index> where;
    for (std::size_t v = 0; v < to.size(); ++v) where.emplace(to[v], static_cast<Index>(v));
    if (where.size() != to.size()) return false;
    p->columns.assign(static_cast<std::size_t>(g.n), -1);
    p->rows.assign(static_cast<std::size_t>(g.m), -1);
    for (std::size_t v = 0; v < from.size(); ++v) {
      const auto found = where.find(from[v]);
      if (found == where.end()) return false;
      const Index image = found->second;
      const auto source = static_cast<Index>(v);
      if (source < g.n) {
        if (image >= g.n) return false;
        p->columns[v] = image;
      } else {
        if (image < g.n) return false;
        p->rows[static_cast<std::size_t>(source - g.n)] = image - g.n;
      }
    }
    return true;
  }

  [[nodiscard]] bool moves_a_column(const Permutation& p) const {
    for (std::size_t j = 0; j < p.columns.size(); ++j) {
      if (p.columns[j] != static_cast<Index>(j)) return true;
    }
    return false;
  }

  /// Does a generator found so far fix every vertex individualised on the current path
  /// and map `a` to `b`? Then `b` is already in the orbit of `a` under the group found and
  /// need not be searched (orbit pruning at this level).
  [[nodiscard]] bool already_reached(const std::vector<Index>& path, Index a, Index b) const {
    // Union-find over the vertices, on the generators that fix the path.
    const auto total = static_cast<std::size_t>(g.vertices());
    std::vector<Index> parent(total);
    std::iota(parent.begin(), parent.end(), 0);
    const auto find = [&](Index v) {
      while (parent[static_cast<std::size_t>(v)] != v) {
        Index& p = parent[static_cast<std::size_t>(v)];
        p = parent[static_cast<std::size_t>(p)];
        v = p;
      }
      return v;
    };
    bool any = false;
    for (const Permutation& p : out.generators) {
      bool fixes = true;
      for (const Index v : path) {
        const Index image = v < g.n ? p.columns[static_cast<std::size_t>(v)]
                                    : g.n + p.rows[static_cast<std::size_t>(v - g.n)];
        if (image != v) {
          fixes = false;
          break;
        }
      }
      if (!fixes) continue;
      any = true;
      for (Index v = 0; v < g.vertices(); ++v) {
        const Index image = v < g.n ? p.columns[static_cast<std::size_t>(v)]
                                    : g.n + p.rows[static_cast<std::size_t>(v - g.n)];
        const Index ra = find(v);
        const Index rb = find(image);
        if (ra != rb) parent[static_cast<std::size_t>(ra)] = rb;
      }
    }
    return any && find(a) == find(b);
  }

  /// Search the subtree under `colour` for a leaf equivalent to `target`: a discrete
  /// partition whose colour-matched bijection onto the target verifies as an automorphism.
  /// Depth first, every vertex of the target cell in turn, stopping at the first success or
  /// the budget. The subtree under a vertex that is truly equivalent to the first vertex is
  /// usually shallow, and one that is not is pruned by the colours long before its leaves.
  bool find_equivalent_leaf(const std::vector<Hash>& colour, int depth,
                            const std::vector<Hash>& target, Permutation* found) {
    const std::vector<Index> cell = target_cell(colour);
    if (cell.empty()) {
      return bijection(colour, target, found) && moves_a_column(*found) &&
             is_automorphism(model, *found);
    }
    for (const Index vertex : cell) {
      if (exhausted()) return false;
      std::vector<Hash> branch = colour;
      ++out.search_nodes;
      individualise(&branch, vertex, depth);
      refine(g, &branch, &out.refinements);
      if (find_equivalent_leaf(branch, depth + 1, target, found)) return true;
    }
    return false;
  }

  /// The search proper, down the leftmost path: at each level, the leaf under the first
  /// vertex of the target cell is the target, and the subtree under every other vertex of
  /// that cell is searched for a leaf equivalent to it; each one found is a generator.
  void run(std::vector<Hash> colour, std::vector<Index> path, int depth) {
    for (;;) {
      const std::vector<Index> cell = target_cell(colour);
      if (cell.empty()) return;
      if (exhausted()) {
        out.budget_exhausted = true;
        return;
      }
      const Index first = cell.front();
      std::vector<Hash> first_branch = colour;
      ++out.search_nodes;
      individualise(&first_branch, first, depth);
      refine(g, &first_branch, &out.refinements);
      std::vector<Hash> first_leaf = first_branch;
      const bool first_discrete = leftmost_leaf(&first_leaf, depth + 1);
      if (first_discrete) {
        for (std::size_t k = 1; k < cell.size(); ++k) {
          const Index other = cell[k];
          if (exhausted()) {
            out.budget_exhausted = true;
            break;
          }
          if (already_reached(path, first, other)) continue;
          std::vector<Hash> branch = colour;
          ++out.search_nodes;
          individualise(&branch, other, depth);
          refine(g, &branch, &out.refinements);
          Permutation candidate;
          if (find_equivalent_leaf(branch, depth + 1, first_leaf, &candidate)) {
            out.generators.push_back(std::move(candidate));
          } else if (exhausted()) {
            out.budget_exhausted = true;
            break;
          }
        }
      } else {
        out.budget_exhausted = true;
        return;
      }
      // Continue down the first branch.
      path.push_back(first);
      colour = std::move(first_branch);
      ++depth;
    }
  }
};

void compute_orbits(SymmetryGroup* group, Index n) {
  std::vector<Index> parent(static_cast<std::size_t>(n));
  std::iota(parent.begin(), parent.end(), 0);
  const auto find = [&](Index v) {
    while (parent[static_cast<std::size_t>(v)] != v) {
      Index& p = parent[static_cast<std::size_t>(v)];
      p = parent[static_cast<std::size_t>(p)];
      v = p;
    }
    return v;
  };
  for (const Permutation& p : group->generators) {
    for (Index j = 0; j < n; ++j) {
      const Index a = find(j);
      const Index b = find(p.columns[static_cast<std::size_t>(j)]);
      if (a != b) parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
    }
  }
  group->orbit_of.assign(static_cast<std::size_t>(n), 0);
  std::vector<Index> size(static_cast<std::size_t>(n), 0);
  for (Index j = 0; j < n; ++j) {
    const Index root = find(j);
    group->orbit_of[static_cast<std::size_t>(j)] = root;
    ++size[static_cast<std::size_t>(root)];
  }
  group->nontrivial_orbits = 0;
  group->largest_orbit = n > 0 ? 1 : 0;
  for (Index j = 0; j < n; ++j) {
    const Index s = size[static_cast<std::size_t>(j)];
    if (s > 1) ++group->nontrivial_orbits;
    group->largest_orbit = std::max(group->largest_orbit, s);
  }
}

}  // namespace

SymmetryGroup detect_symmetry(const Model& model, Count search_limit) {
  const Graph g = build_graph(model);
  Search search(g, model, search_limit);
  std::vector<Hash> colour = g.initial;
  refine(g, &colour, &search.out.refinements);
  search.run(std::move(colour), {}, 0);
  SymmetryGroup group = std::move(search.out);
  compute_orbits(&group, model.num_cols());
  return group;
}

bool is_automorphism(const Model& model, const Permutation& p) {
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (p.columns.size() != static_cast<std::size_t>(n) ||
      p.rows.size() != static_cast<std::size_t>(m)) {
    return false;
  }
  const auto bijective = [](const std::vector<Index>& map) {
    std::vector<bool> hit(map.size(), false);
    for (const Index v : map) {
      if (v < 0 || static_cast<std::size_t>(v) >= map.size() ||
          hit[static_cast<std::size_t>(v)]) {
        return false;
      }
      hit[static_cast<std::size_t>(v)] = true;
    }
    return true;
  };
  if (!bijective(p.columns) || !bijective(p.rows)) return false;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const auto w = static_cast<std::size_t>(p.columns[u]);
    if (model.col_cost[u] != model.col_cost[w] || model.col_lower[u] != model.col_lower[w] ||
        model.col_upper[u] != model.col_upper[w] || model.col_type[u] != model.col_type[w]) {
      return false;
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto r = static_cast<std::size_t>(i);
    const auto s = static_cast<std::size_t>(p.rows[r]);
    if (model.row_lower[r] != model.row_lower[s] || model.row_upper[r] != model.row_upper[s]) {
      return false;
    }
  }
  // Every entry (i, j, a) must reappear as (rows[i], columns[j], a): column j's entries with
  // their rows mapped, sorted, must equal the image column's entries exactly.
  std::vector<std::pair<Index, double>> mapped;
  for (Index j = 0; j < n; ++j) {
    const ColumnView from = model.matrix.column(j);
    const ColumnView to = model.matrix.column(p.columns[static_cast<std::size_t>(j)]);
    if (from.size != to.size) return false;
    mapped.clear();
    for (Index k = 0; k < from.size; ++k) {
      mapped.emplace_back(p.rows[static_cast<std::size_t>(from.rows[k])], from.values[k]);
    }
    std::sort(mapped.begin(), mapped.end());
    for (Index k = 0; k < to.size; ++k) {
      if (mapped[static_cast<std::size_t>(k)].first != to.rows[k] ||
          mapped[static_cast<std::size_t>(k)].second != to.values[k]) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::pair<Index, Index>> ordering_rows(const SymmetryGroup& group) {
  std::vector<std::pair<Index, Index>> rows;
  for (const Permutation& p : group.generators) {
    for (std::size_t j = 0; j < p.columns.size(); ++j) {
      const Index image = p.columns[j];
      if (image == static_cast<Index>(j)) continue;
      // The first moved column maps past itself: everything before it is fixed, so its
      // image cannot be one of those.
      const std::pair<Index, Index> pair{static_cast<Index>(j), image};
      if (std::find(rows.begin(), rows.end(), pair) == rows.end()) rows.push_back(pair);
      break;
    }
  }
  return rows;
}

}  // namespace sankhya::mip
