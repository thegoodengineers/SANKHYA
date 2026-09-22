// SPDX-License-Identifier: Apache-2.0
// SANKHYA - clique cuts and {0,1/2}-Chvatal-Gomory cuts (#358). References and the validity
// argument for each family are on the declarations.

#include "combinatorial_cuts.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

/// Past this many conflict pairs the graph stops growing. Fewer conflicts only means fewer
/// cliques - a missing edge can never make a cut invalid - so the cap trades strength for
/// memory and never correctness.
constexpr int kMaxConflictEdges = 200000;
/// Clique separation starts from at most this many of the largest LP values.
constexpr int kCliqueStarts = 50;
/// {0,1/2}: partners tried per row, and row pairs tried in total.
constexpr int kPartnersPerRow = 10;
constexpr int kMaxRowPairs = 20000;
/// A cut must be violated at the LP point by more than this, relative to max(1, |rhs|).
constexpr double kMinViolation = 1e-6;
/// The mod-2 elimination (#358 follow-up): at most this many rows (smallest slack first), this
/// many columns (largest value first), and this many row sets handed to the exact check. A
/// row or column left out only means fewer cuts; every candidate is re-derived exactly.
constexpr std::size_t kMaxMod2Rows = 300;
constexpr std::size_t kMaxMod2Columns = 1000;
constexpr std::size_t kMaxMod2Sets = 500;

bool is_binary(const Model& model, const std::vector<double>& lower,
               const std::vector<double>& upper, std::size_t j) {
  return model.col_type[j] == VarType::kInteger && lower[j] == 0.0 && upper[j] == 1.0;
}

}  // namespace

// ---- Clique cuts
// -----------------------------------------------------------------------------

std::vector<Cut> generate_clique_cuts(const Model& model, const Solution& solution,
                                      const std::vector<double>& col_lower,
                                      const std::vector<double>& col_upper,
                                      CombinatorialCutStats* stats) {
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  const auto un = static_cast<std::size_t>(n);
  if (solution.col_value.size() != un) return cuts;
  const std::vector<double>& x = solution.col_value;

  // ---- The conflict graph, from every row in <= orientation.
  std::vector<std::vector<Index>> adjacent(un);
  std::unordered_set<std::uint64_t> edge;
  const auto key = [n](Index p, Index q) {
    const Index a = std::min(p, q);
    const Index b = std::max(p, q);
    return static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(n) +
           static_cast<std::uint64_t>(b);
  };
  bool capped = false;
  const CsrView by_row(model.matrix);
  for (Index i = 0; i < model.num_rows() && !capped; ++i) {
    const ColumnView row = by_row.row(i);
    const auto ui = static_cast<std::size_t>(i);
    for (const double sign : {1.0, -1.0}) {
      // sign +1: a x <= u.  sign -1: -a x <= -l.
      const double bound = sign > 0.0 ? model.row_upper[ui] : -model.row_lower[ui];
      if (!std::isfinite(bound)) continue;
      // The least every column outside the positive binaries can contribute.
      std::vector<std::pair<double, Index>> positive;
      double least_others = 0.0;
      bool finite = true;
      for (Index k = 0; k < row.size && finite; ++k) {
        const Index j = row.rows[k];
        const auto u = static_cast<std::size_t>(j);
        const double a = sign * row.values[k];
        if (a > 0.0 && is_binary(model, col_lower, col_upper, u)) {
          positive.emplace_back(a, j);
        } else if (a != 0.0) {
          const double low = a > 0.0 ? col_lower[u] : col_upper[u];
          if (!std::isfinite(low)) finite = false;
          least_others += a * low;
        }
      }
      if (!finite || positive.size() < 2) continue;
      // Both at 1 is impossible when a_p + a_q + least_others > bound. The margin is a
      // feasibility tolerance, scaled, so a pair the solver would accept within tolerance is
      // never declared in conflict.
      const double room = bound - least_others + 1e-6 * std::max(1.0, std::fabs(bound));
      std::sort(positive.begin(), positive.end(), [](const auto& l, const auto& r) {
        return l.first > r.first || (l.first == r.first && l.second < r.second);
      });
      for (std::size_t p = 0; p < positive.size() && !capped; ++p) {
        for (std::size_t q = p + 1; q < positive.size(); ++q) {
          if (positive[p].first + positive[q].first <= room) break;  // sorted: none further
          const Index a = positive[p].second;
          const Index b = positive[q].second;
          if (edge.insert(key(a, b)).second) {
            adjacent[static_cast<std::size_t>(a)].push_back(b);
            adjacent[static_cast<std::size_t>(b)].push_back(a);
            if (static_cast<int>(edge.size()) >= kMaxConflictEdges) {
              capped = true;
              break;
            }
          }
        }
      }
    }
  }
  if (stats != nullptr) {
    stats->conflict_edges = static_cast<int>(edge.size());
    stats->conflict_graph_capped = capped;
  }
  if (edge.empty()) return cuts;
  const auto conflict = [&](Index p, Index q) { return edge.count(key(p, q)) > 0; };

  // ---- Greedy separation from the largest LP values.
  std::vector<Index> order;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (is_binary(model, col_lower, col_upper, u) && x[u] > 1e-6 && !adjacent[u].empty()) {
      order.push_back(j);
    }
  }
  std::sort(order.begin(), order.end(), [&](Index l, Index r) {
    const double xl = x[static_cast<std::size_t>(l)];
    const double xr = x[static_cast<std::size_t>(r)];
    return xl > xr || (xl == xr && l < r);
  });
  std::set<std::vector<Index>> seen;
  const std::size_t starts = std::min<std::size_t>(order.size(), kCliqueStarts);
  for (std::size_t s = 0; s < starts; ++s) {
    std::vector<Index> clique{order[s]};
    double weight = x[static_cast<std::size_t>(order[s])];
    for (const Index v : order) {
      if (v == order[s]) continue;
      if (std::all_of(clique.begin(), clique.end(), [&](Index c) { return conflict(v, c); })) {
        clique.push_back(v);
        weight += x[static_cast<std::size_t>(v)];
      }
    }
    if (weight <= 1.0 + kMinViolation) continue;
    // Extend to a maximal clique with the start's other neighbours: a larger clique is a
    // stronger cut, and every member still conflicts with every other.
    std::vector<Index> neighbours = adjacent[static_cast<std::size_t>(order[s])];
    std::sort(neighbours.begin(), neighbours.end());
    for (const Index v : neighbours) {
      if (std::find(clique.begin(), clique.end(), v) != clique.end()) continue;
      if (std::all_of(clique.begin(), clique.end(), [&](Index c) { return conflict(v, c); })) {
        clique.push_back(v);
      }
    }
    std::sort(clique.begin(), clique.end());
    if (!seen.insert(clique).second) continue;
    Cut cut;
    cut.coeff.assign(un, 0.0);
    for (const Index c : clique) cut.coeff[static_cast<std::size_t>(c)] = 1.0;
    cut.rhs = 1.0;
    cuts.push_back(std::move(cut));
  }
  return cuts;
}

// ---- {0,1/2}-Chvatal-Gomory cuts
// -------------------------------------------------------------

namespace {

/// A row in <= form over the substituted non-negative columns x' (x = lower + x', or
/// x = upper - x' for a column bounded above only).
struct IntegerRow {
  std::map<Index, double> coef;  // integer-valued, in x'
  double rhs = 0.0;
};

/// A fixed-size set of small integers, as 64-bit words.
struct Bits {
  std::vector<std::uint64_t> words;
  explicit Bits(std::size_t size = 0) : words((size + 63) / 64, 0) {}
  [[nodiscard]] bool test(std::size_t k) const {
    return ((words[k / 64] >> (k % 64)) & 1U) != 0;
  }
  void flip(std::size_t k) { words[k / 64] ^= std::uint64_t{1} << (k % 64); }
  void flip_all(const Bits& other) {
    for (std::size_t w = 0; w < words.size(); ++w) words[w] ^= other.words[w];
  }
  template <typename F>
  void for_each(F&& visit) const {
    for (std::size_t w = 0; w < words.size(); ++w) {
      for (std::uint64_t word = words[w]; word != 0; word &= word - 1) {
        visit(w * 64 + static_cast<std::size_t>(std::countr_zero(word)));
      }
    }
  }
};

/// Row sets whose {0,1/2} combination may be violated, by Gaussian elimination over GF(2)
/// (Koster, Zymolka & Kutschka, "Algorithms to separate {0,1/2}-Chvatal-Gomory cuts",
/// Algorithmica 55, 2009).
///
/// THE ARITHMETIC THIS RESTS ON. For a set S of rows at 1/2 with an odd right-hand-side sum,
/// the cut floor(sum a / 2) x' <= floor(sum b / 2) is violated at x' by exactly
///   (1 - sum_{r in S} slack_r - sum_{j with odd sum a_j} x'_j) / 2,
/// so a violated cut is a set of rows whose parities cancel on every column with x'_j > 0,
/// whose right-hand sides sum to an odd number, and whose slacks sum to less than one. Rows
/// with slack of one or more cannot be in such a set and columns at zero cost nothing, so both
/// are dropped, and elimination on the columns with the largest x' first cancels the expensive
/// parities. Every set found is only a CANDIDATE: the caller re-derives the cut from the
/// original rows and checks its violation, so a slip in this bookkeeping costs a cut, never a
/// wrong one.
std::vector<std::vector<int>> mod2_row_sets(const std::vector<IntegerRow>& rows,
                                            const std::vector<double>& value) {
  std::vector<std::vector<int>> sets;
  std::vector<std::pair<double, int>> by_slack;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    double activity = 0.0;
    for (const auto& [j, a] : rows[r].coef) activity += a * value[static_cast<std::size_t>(j)];
    const double slack = std::max(0.0, rows[r].rhs - activity);
    if (slack < 1.0 - kMinViolation) by_slack.emplace_back(slack, static_cast<int>(r));
  }
  std::sort(by_slack.begin(), by_slack.end());
  if (by_slack.size() > kMaxMod2Rows) by_slack.resize(kMaxMod2Rows);
  const std::size_t m = by_slack.size();
  if (m < 2) return sets;

  // The columns worth cancelling: odd in some kept row, and positive at x'.
  std::map<Index, std::size_t> bit_of;
  {
    std::vector<std::pair<double, Index>> by_value;
    std::set<Index> seen;
    for (const auto& [slack, r] : by_slack) {
      for (const auto& [j, a] : rows[static_cast<std::size_t>(r)].coef) {
        const double v = value[static_cast<std::size_t>(j)];
        if (std::fmod(std::fabs(a), 2.0) != 1.0 || v <= kMinViolation) continue;
        if (seen.insert(j).second) by_value.emplace_back(-v, j);
      }
    }
    std::sort(by_value.begin(), by_value.end());
    if (by_value.size() > kMaxMod2Columns) by_value.resize(kMaxMod2Columns);
    for (std::size_t k = 0; k < by_value.size(); ++k) bit_of[by_value[k].second] = k;
  }
  const std::size_t columns = bit_of.size();
  std::vector<double> column_value(columns, 0.0);
  for (const auto& [j, k] : bit_of) column_value[k] = value[static_cast<std::size_t>(j)];

  struct Work {
    Bits parity;   // odd columns
    Bits members;  // which kept rows are summed
    bool odd_rhs = false;
    bool active = true;
  };
  std::vector<Work> work(m);
  std::vector<double> slack(m);
  for (std::size_t q = 0; q < m; ++q) {
    const IntegerRow& row = rows[static_cast<std::size_t>(by_slack[q].second)];
    slack[q] = by_slack[q].first;
    work[q].parity = Bits(columns);
    work[q].members = Bits(m);
    work[q].members.flip(q);
    for (const auto& [j, a] : row.coef) {
      const auto found = bit_of.find(j);
      if (found != bit_of.end() && std::fmod(std::fabs(a), 2.0) == 1.0) {
        work[q].parity.flip(found->second);
      }
    }
    work[q].odd_rhs = std::fmod(std::fabs(row.rhs), 2.0) == 1.0;
  }
  const auto weight = [&](const Work& w) {
    double total = 0.0;
    w.members.for_each([&](std::size_t q) { total += slack[q]; });
    w.parity.for_each([&](std::size_t k) { total += column_value[k]; });
    return total;
  };

  std::set<std::vector<int>> found;
  const auto consider = [&](const Work& w) {
    if (!w.odd_rhs || sets.size() >= kMaxMod2Sets) return;
    if (weight(w) >= 1.0 - kMinViolation) return;
    std::vector<int> set;
    w.members.for_each([&](std::size_t q) { set.push_back(by_slack[q].second); });
    if (set.size() >= 2 && found.insert(set).second) sets.push_back(std::move(set));
  };

  for (std::size_t k = 0; k < columns; ++k) {
    // The pivot is the cheapest active row with this column odd; it leaves the system.
    std::size_t pivot = m;
    double best = 0.0;
    for (std::size_t q = 0; q < m; ++q) {
      if (!work[q].active || !work[q].parity.test(k)) continue;
      const double w = weight(work[q]);
      if (pivot == m || w < best) {
        pivot = q;
        best = w;
      }
    }
    if (pivot == m) continue;
    work[pivot].active = false;
    for (std::size_t q = 0; q < m; ++q) {
      if (!work[q].active || !work[q].parity.test(k)) continue;
      work[q].parity.flip_all(work[pivot].parity);
      work[q].members.flip_all(work[pivot].members);
      work[q].odd_rhs = work[q].odd_rhs != work[pivot].odd_rhs;
      consider(work[q]);
    }
  }
  for (const Work& w : work) {
    if (w.active) consider(w);
  }
  return sets;
}

}  // namespace

std::vector<Cut> generate_zero_half_cuts(const Model& model, const Solution& solution,
                                         const std::vector<double>& col_lower,
                                         const std::vector<double>& col_upper,
                                         CombinatorialCutStats* stats) {
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  const auto un = static_cast<std::size_t>(n);
  if (solution.col_value.size() != un) return cuts;
  const std::vector<double>& x = solution.col_value;

  // The substitution that makes every column non-negative, chosen once per column so every
  // row is expressed in the same variables. 0: x = l + x'. 1: x = u - x'. 2: unusable.
  std::vector<int> shift(un, 2);
  std::vector<double> anchor(un, 0.0);
  for (std::size_t j = 0; j < un; ++j) {
    if (model.col_type[j] != VarType::kInteger) continue;
    if (std::isfinite(col_lower[j])) {
      shift[j] = 0;
      anchor[j] = std::ceil(col_lower[j] - 1e-9);
    } else if (std::isfinite(col_upper[j])) {
      shift[j] = 1;
      anchor[j] = std::floor(col_upper[j] + 1e-9);
    }
  }

  // Every eligible row side: pure-integer, integer coefficients, every column substitutable.
  std::vector<IntegerRow> rows;
  const CsrView by_row(model.matrix);
  for (Index i = 0; i < model.num_rows(); ++i) {
    const ColumnView row = by_row.row(i);
    const auto ui = static_cast<std::size_t>(i);
    bool eligible = row.size > 0;
    for (Index k = 0; k < row.size && eligible; ++k) {
      const auto u = static_cast<std::size_t>(row.rows[k]);
      eligible = shift[u] != 2 && row.values[k] == std::round(row.values[k]);
    }
    if (!eligible) continue;
    for (const double sign : {1.0, -1.0}) {
      const double bound = sign > 0.0 ? model.row_upper[ui] : -model.row_lower[ui];
      if (!std::isfinite(bound)) continue;
      IntegerRow r;
      r.rhs = bound;
      for (Index k = 0; k < row.size; ++k) {
        const Index j = row.rows[k];
        const auto u = static_cast<std::size_t>(j);
        const double a = sign * row.values[k];
        // a x = a (l + x') = a l + a x'   or   a (u - x') = a u - a x'
        r.rhs -= a * anchor[u];
        r.coef[j] = shift[u] == 0 ? a : -a;
      }
      rows.push_back(std::move(r));
    }
  }

  // Which rows touch each column, to pair rows that share one.
  std::vector<std::vector<int>> rows_of(un);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (const auto& [j, a] : rows[r].coef)
      rows_of[static_cast<std::size_t>(j)].push_back(static_cast<int>(r));
  }
  std::vector<std::vector<int>> sets;
  for (std::size_t r = 0; r < rows.size(); ++r) sets.push_back({static_cast<int>(r)});
  std::set<std::pair<int, int>> paired;
  for (std::size_t r = 0; r < rows.size() && static_cast<int>(paired.size()) < kMaxRowPairs;
       ++r) {
    int partners = 0;
    for (const auto& [j, a] : rows[r].coef) {
      for (const int s : rows_of[static_cast<std::size_t>(j)]) {
        if (s <= static_cast<int>(r) || partners >= kPartnersPerRow) continue;
        if (paired.insert({static_cast<int>(r), s}).second) {
          sets.push_back({static_cast<int>(r), s});
          ++partners;
        }
      }
      if (partners >= kPartnersPerRow) break;
    }
  }
  // Sets of any size, from the elimination over GF(2): what reaches an odd cycle of three or
  // more rows, which no single row or pair can.
  {
    std::vector<double> value(un, 0.0);
    for (std::size_t j = 0; j < un; ++j) {
      if (shift[j] == 0) value[j] = std::max(0.0, x[j] - anchor[j]);
      if (shift[j] == 1) value[j] = std::max(0.0, anchor[j] - x[j]);
    }
    std::vector<std::vector<int>> wide = mod2_row_sets(rows, value);
    if (stats != nullptr) stats->mod2_row_sets = static_cast<int>(wide.size());
    for (std::vector<int>& set : wide) sets.push_back(std::move(set));
  }
  if (stats != nullptr) stats->candidate_row_sets = static_cast<int>(sets.size());

  std::set<std::pair<std::vector<std::pair<Index, double>>, double>> seen;
  for (const std::vector<int>& set : sets) {
    std::map<Index, double> sum;
    double rhs_sum = 0.0;
    for (const int r : set) {
      for (const auto& [j, a] : rows[static_cast<std::size_t>(r)].coef) sum[j] += a;
      rhs_sum += rows[static_cast<std::size_t>(r)].rhs;
    }
    // floor(u'A) x' <= floor(u'b), u = 1/2 on each row of the set.
    std::vector<std::pair<Index, double>> in_x;  // back in the model's own columns
    double rhs = std::floor(rhs_sum / 2.0 + 1e-12);
    for (const auto& [j, a] : sum) {
      const double c = std::floor(a / 2.0);
      if (c == 0.0) continue;
      const auto u = static_cast<std::size_t>(j);
      // c x' with x' = x - l  ->  c x - c l ;   x' = u - x  ->  -c x + c u
      if (shift[u] == 0) {
        in_x.emplace_back(j, c);
        rhs += c * anchor[u];
      } else {
        in_x.emplace_back(j, -c);
        rhs -= c * anchor[u];
      }
    }
    if (in_x.empty()) continue;
    double activity = 0.0;
    for (const auto& [j, c] : in_x) activity += c * x[static_cast<std::size_t>(j)];
    if (activity <= rhs + kMinViolation * std::max(1.0, std::fabs(rhs))) continue;
    if (!seen.insert({in_x, rhs}).second) continue;
    Cut cut;
    cut.coeff.assign(un, 0.0);
    for (const auto& [j, c] : in_x) cut.coeff[static_cast<std::size_t>(j)] = c;
    cut.rhs = rhs;
    cuts.push_back(std::move(cut));
  }
  return cuts;
}

}  // namespace sankhya::mip
