// SPDX-License-Identifier: Apache-2.0
// SANKHYA - binary probing and the clique table (#512). See probing.hpp for the references,
// what each outcome of a probe licenses, and the literal encoding.

#include "presolve/probing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {
namespace {

/// Past this many conflicts the table stops growing: a missing edge only means fewer
/// cliques, never a wrong one. The same cap the clique separator puts on its own graph.
constexpr std::size_t kMaxConflicts = 200000;
/// Hash lookups the greedy clique merge may spend; past it the table is what was merged.
constexpr std::int64_t kMaxMergeWork = 50000000;

[[nodiscard]] bool finite(double v) {
  return std::fabs(v) < kInfinity;
}

struct Change {
  Index column;
  double lower;
  double upper;
};

class Prober {
 public:
  Prober(const Model& model, const std::vector<double>& lower, const std::vector<double>& upper,
         std::int64_t work_limit)
      : model_(model),
        n_(model.num_cols()),
        by_row_(model.matrix),
        lo_(lower),
        up_(upper),
        work_limit_(work_limit),
        queued_(static_cast<std::size_t>(model.num_rows()), 0),
        stamp_(static_cast<std::size_t>(n_), -1),
        seen_lo_(static_cast<std::size_t>(n_), 0.0),
        seen_up_(static_cast<std::size_t>(n_), 0.0) {}

  ProbingResult run() {
    ProbingResult result;
    std::vector<Index> candidates;
    for (Index j = 0; j < n_; ++j) {
      if (is_binary(j)) candidates.push_back(j);
    }
    // Most constrained first: the binaries in the most rows have the most to propagate.
    std::stable_sort(candidates.begin(), candidates.end(), [&](Index a, Index b) {
      return model_.matrix.column(a).size > model_.matrix.column(b).size;
    });
    for (std::size_t c = 0; c < candidates.size() && !result.infeasible; ++c) {
      if (work_ >= work_limit_) {
        result.work_limit_reached = true;
        break;
      }
      const Index j = candidates[c];
      if (lo_[static_cast<std::size_t>(j)] == up_[static_cast<std::size_t>(j)]) continue;
      probe_one(j, &result);
    }
    merge_cliques(&result);
    result.col_lower = std::move(lo_);
    result.col_upper = std::move(up_);
    return result;
  }

 private:
  [[nodiscard]] bool is_binary(Index j) const {
    const auto u = static_cast<std::size_t>(j);
    return model_.col_type[u] == VarType::kInteger && lo_[u] == 0.0 && up_[u] == 1.0;
  }
  [[nodiscard]] bool is_integer(Index j) const {
    return model_.col_type[static_cast<std::size_t>(j)] == VarType::kInteger;
  }
  [[nodiscard]] Index literal(Index j, double value) const { return value > 0.5 ? j : n_ + j; }

  /// Probe x_j = 0 and x_j = 1 and apply what the two outcomes license.
  void probe_one(Index j, ProbingResult* result) {
    ++result->probed;
    std::vector<Change> outcome[2];
    bool feasible[2];
    for (int v = 0; v < 2; ++v) {
      feasible[v] = run_probe(j, static_cast<double>(v), &outcome[v]);
    }
    if (!feasible[0] && !feasible[1]) {
      result->infeasible = true;
      return;
    }
    if (!feasible[0] || !feasible[1]) {
      const double value = feasible[0] ? 0.0 : 1.0;
      trail_.clear();
      probe_work_ = 0;
      const bool ok = fix(j, value) && propagate();
      if (!ok) {
        result->infeasible = true;
        return;
      }
      ++result->fixings;
      result->tightenings += static_cast<Count>(changed_columns()) - 1;
      trail_.clear();
      return;
    }
    // Both feasible: implications and conflicts from each, tightenings from their union.
    for (int v = 0; v < 2; ++v) {
      for (const Change& change : outcome[v]) {
        if (change.column == j) continue;
        ++result->implications;
        if (is_binary(change.column) && change.lower == change.upper) {
          add_conflict(literal(j, static_cast<double>(v)),
                       literal(change.column, 1.0 - change.lower), result);
        }
      }
    }
    ++probe_id_;
    probe_work_ = 0;
    for (const Change& change : outcome[0]) {
      const auto u = static_cast<std::size_t>(change.column);
      stamp_[u] = probe_id_;
      seen_lo_[u] = change.lower;
      seen_up_[u] = change.upper;
    }
    trail_.clear();
    bool ok = true;
    bool moved = false;
    for (const Change& change : outcome[1]) {
      const auto u = static_cast<std::size_t>(change.column);
      if (stamp_[u] != probe_id_ || change.column == j) continue;
      const double lower = std::min(seen_lo_[u], change.lower);
      const double upper = std::max(seen_up_[u], change.upper);
      if (lower > lo_[u]) {
        const auto before = trail_.size();
        ok = set_lower(change.column, lower) && ok;
        moved = moved || trail_.size() > before;
      }
      if (upper < up_[u]) {
        const auto before = trail_.size();
        ok = set_upper(change.column, upper) && ok;
        moved = moved || trail_.size() > before;
      }
    }
    if (moved) ok = propagate() && ok;
    if (!ok) {
      result->infeasible = true;
      return;
    }
    result->tightenings += static_cast<Count>(changed_columns());
    trail_.clear();
  }

  /// Fix x_j = value on top of the global bounds, propagate, record the bounds that moved,
  /// and undo. False when the probe is infeasible.
  bool run_probe(Index j, double value, std::vector<Change>* changes) {
    trail_.clear();
    probe_work_ = 0;
    const bool ok = fix(j, value) && propagate();
    changes->clear();
    if (ok) {
      ++probe_id_;
      for (auto it = trail_.rbegin(); it != trail_.rend(); ++it) {
        const auto u = static_cast<std::size_t>(it->column);
        if (stamp_[u] == probe_id_) continue;
        stamp_[u] = probe_id_;
        changes->push_back({it->column, lo_[u], up_[u]});
      }
    }
    for (auto it = trail_.rbegin(); it != trail_.rend(); ++it) {
      lo_[static_cast<std::size_t>(it->column)] = it->lower;
      up_[static_cast<std::size_t>(it->column)] = it->upper;
    }
    trail_.clear();
    clear_queue();
    return ok;
  }

  [[nodiscard]] std::size_t changed_columns() {
    ++probe_id_;
    std::size_t count = 0;
    for (const Change& change : trail_) {
      const auto u = static_cast<std::size_t>(change.column);
      if (stamp_[u] == probe_id_) continue;
      stamp_[u] = probe_id_;
      ++count;
    }
    return count;
  }

  bool fix(Index j, double value) { return set_lower(j, value) && set_upper(j, value); }

  void clear_queue() {
    for (const Index r : queue_) queued_[static_cast<std::size_t>(r)] = 0;
    queue_.clear();
  }

  void record(Index j) {
    const auto u = static_cast<std::size_t>(j);
    trail_.push_back({j, lo_[u], up_[u]});
    const ColumnView column = model_.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto r = static_cast<std::size_t>(column.rows[k]);
      if (queued_[r] == 0) {
        queued_[r] = 1;
        queue_.push_back(column.rows[k]);
      }
    }
  }

  /// Integer bounds round inward past the integrality tolerance; a continuous bound only
  /// moves by a meaningful step (tol::kProbingBoundMinStep), so propagation terminates.
  bool set_upper(Index j, double v) {
    const auto u = static_cast<std::size_t>(j);
    if (!std::isfinite(v) || std::fabs(v) > tol::kProbingMaxBound) return true;
    if (is_integer(j)) v = std::floor(v + tol::kIntegrality);
    const double old = up_[u];
    const bool improves =
        !finite(old) ||
        (is_integer(j) ? v < old
                       : v < old - tol::kProbingBoundMinStep * std::max(1.0, std::fabs(old)));
    if (!improves) return true;
    const double lo = lo_[u];
    if (finite(lo) && v < lo) {
      if (v < lo - tol::kProbingInfeasibility * std::max(1.0, std::fabs(lo))) return false;
      v = lo;
      if (v >= old) return true;
    }
    record(j);
    up_[u] = v;
    return true;
  }

  bool set_lower(Index j, double v) {
    const auto u = static_cast<std::size_t>(j);
    if (!std::isfinite(v) || std::fabs(v) > tol::kProbingMaxBound) return true;
    if (is_integer(j)) v = std::ceil(v - tol::kIntegrality);
    const double old = lo_[u];
    const bool improves =
        !finite(old) ||
        (is_integer(j) ? v > old
                       : v > old + tol::kProbingBoundMinStep * std::max(1.0, std::fabs(old)));
    if (!improves) return true;
    const double up = up_[u];
    if (finite(up) && v > up) {
      if (v > up + tol::kProbingInfeasibility * std::max(1.0, std::fabs(up))) return false;
      v = up;
      if (v <= old) return true;
    }
    record(j);
    lo_[u] = v;
    return true;
  }

  /// Rows touched by a moved bound, to a fixed point or until this probe's work runs out.
  /// Running out is not infeasibility: the bounds so far are still implied.
  bool propagate() {
    std::size_t head = 0;
    bool ok = true;
    while (head < queue_.size() && ok) {
      if (probe_work_ >= tol::kProbingProbeWorkLimit || work_ >= work_limit_) break;
      const Index i = queue_[head++];
      queued_[static_cast<std::size_t>(i)] = 0;
      ok = propagate_row(i);
    }
    clear_queue();
    return ok;
  }

  bool propagate_row(Index i) {
    const auto r = static_cast<std::size_t>(i);
    const double row_lower = model_.row_lower[r];
    const double row_upper = model_.row_upper[r];
    const ColumnView row = by_row_.row(i);
    work_ += row.size;
    probe_work_ += row.size;
    double min = 0.0;
    double max = 0.0;
    int min_inf = 0;
    int max_inf = 0;
    double magnitude = 0.0;
    for (Index k = 0; k < row.size; ++k) {
      const auto u = static_cast<std::size_t>(row.rows[k]);
      const double a = row.values[k];
      const double low = a > 0.0 ? lo_[u] : up_[u];
      const double high = a > 0.0 ? up_[u] : lo_[u];
      if (finite(low)) {
        min += a * low;
        magnitude += std::fabs(a * low);
      } else {
        ++min_inf;
      }
      if (finite(high)) {
        max += a * high;
        magnitude += std::fabs(a * high);
      } else {
        ++max_inf;
      }
    }
    const double scale = std::max(1.0, magnitude);
    // A row is only called violated by more than the probing tolerance plus the rounding
    // the activity sums can carry: a false "infeasible" would fix a binary wrongly.
    const auto missed_by = [&](double bound) {
      return tol::kProbingInfeasibility * std::max(1.0, std::fabs(bound)) +
             tol::kProbingSafety * scale;
    };
    if (finite(row_upper) && min_inf == 0 && min > row_upper + missed_by(row_upper)) {
      return false;
    }
    if (finite(row_lower) && max_inf == 0 && max < row_lower - missed_by(row_lower)) {
      return false;
    }
    for (Index k = 0; k < row.size; ++k) {
      const Index j = row.rows[k];
      const auto u = static_cast<std::size_t>(j);
      const double a = row.values[k];
      if (std::fabs(a) <= tol::kZeroDrop) continue;
      // The activity above was summed before any of this row's columns moved, so a column
      // tightened earlier in this loop only leaves the others a wider, weaker range.
      const double own_min = a > 0.0 ? lo_[u] : up_[u];
      const double own_max = a > 0.0 ? up_[u] : lo_[u];
      const double slack = tol::kProbingSafety * scale / std::fabs(a);
      if (finite(row_upper) && min_inf - (finite(own_min) ? 0 : 1) == 0) {
        const double rest = finite(own_min) ? min - a * own_min : min;
        const double v = (row_upper - rest) / a;
        if (!(a > 0.0 ? set_upper(j, v + slack) : set_lower(j, v - slack))) return false;
      }
      if (finite(row_lower) && max_inf - (finite(own_max) ? 0 : 1) == 0) {
        const double rest = finite(own_max) ? max - a * own_max : max;
        const double v = (row_lower - rest) / a;
        if (!(a > 0.0 ? set_lower(j, v - slack) : set_upper(j, v + slack))) return false;
      }
    }
    return true;
  }

  void add_conflict(Index a, Index b, ProbingResult* result) {
    if (a == b) return;
    if (result->conflicts.size() >= kMaxConflicts) {
      result->conflicts_capped = true;
      return;
    }
    const Index first = std::min(a, b);
    const Index second = std::max(a, b);
    if (edges_.insert(key(first, second)).second) result->conflicts.emplace_back(first, second);
  }

  [[nodiscard]] std::uint64_t key(Index a, Index b) const {
    return static_cast<std::uint64_t>(std::min(a, b)) * static_cast<std::uint64_t>(2 * n_) +
           static_cast<std::uint64_t>(std::max(a, b));
  }

  /// Clique merging (Atamturk, Nemhauser & Savelsbergh 2000; Achterberg et al. 2020): cover
  /// the conflict edges with maximal cliques, greedily from the literals of highest degree.
  /// Every pair inside a clique is an edge, checked, so each clique is valid on its own.
  void merge_cliques(ProbingResult* result) {
    if (result->conflicts.empty()) return;
    const auto literals = static_cast<std::size_t>(2 * n_);
    std::vector<std::vector<Index>> adjacent(literals);
    for (const auto& [a, b] : result->conflicts) {
      adjacent[static_cast<std::size_t>(a)].push_back(b);
      adjacent[static_cast<std::size_t>(b)].push_back(a);
    }
    const auto degree = [&](Index l) { return adjacent[static_cast<std::size_t>(l)].size(); };
    std::vector<Index> order;
    for (Index l = 0; l < 2 * n_; ++l) {
      if (degree(l) > 0) order.push_back(l);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](Index a, Index b) { return degree(a) > degree(b); });
    for (auto& list : adjacent) {
      std::stable_sort(list.begin(), list.end(),
                       [&](Index a, Index b) { return degree(a) > degree(b); });
    }
    std::unordered_set<std::uint64_t> covered;
    std::int64_t work = 0;
    for (const Index start : order) {
      for (const Index partner : adjacent[static_cast<std::size_t>(start)]) {
        if (work > kMaxMergeWork) return;
        if (covered.count(key(start, partner)) > 0) continue;
        std::vector<Index> clique{start, partner};
        for (const Index v : adjacent[static_cast<std::size_t>(start)]) {
          if (v == partner) continue;
          bool all = true;
          for (const Index c : clique) {
            ++work;
            if (edges_.count(key(v, c)) == 0) {
              all = false;
              break;
            }
          }
          if (all) clique.push_back(v);
        }
        for (std::size_t p = 0; p < clique.size(); ++p) {
          for (std::size_t q = p + 1; q < clique.size(); ++q) {
            covered.insert(key(clique[p], clique[q]));
          }
        }
        std::sort(clique.begin(), clique.end());
        result->largest_clique =
            std::max(result->largest_clique, static_cast<Count>(clique.size()));
        result->cliques.push_back(std::move(clique));
      }
    }
  }

  const Model& model_;
  Index n_;
  CsrView by_row_;
  std::vector<double> lo_;
  std::vector<double> up_;
  std::int64_t work_limit_;
  std::int64_t work_ = 0;
  std::int64_t probe_work_ = 0;
  std::vector<Change> trail_;  ///< bounds before each change, in order
  std::vector<Index> queue_;
  std::vector<char> queued_;
  std::vector<std::int64_t> stamp_;
  std::vector<double> seen_lo_;
  std::vector<double> seen_up_;
  std::int64_t probe_id_ = 0;
  std::unordered_set<std::uint64_t> edges_;
};

}  // namespace

ProbingResult probe_binaries(const Model& model, const std::vector<double>& col_lower,
                             const std::vector<double>& col_upper, std::int64_t work_limit) {
  if (!model.has_integrality() || model.num_rows() == 0) {
    ProbingResult nothing;
    nothing.col_lower = col_lower;
    nothing.col_upper = col_upper;
    return nothing;
  }
  return Prober(model, col_lower, col_upper, work_limit).run();
}

}  // namespace sankhya::presolve
