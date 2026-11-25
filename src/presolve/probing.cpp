// SPDX-License-Identifier: Apache-2.0
// SANKHYA - binary probing and the clique table (#512). See probing.hpp for the references,
// what each outcome of a probe licenses, and the literal encoding.
//
// ROW ACTIVITIES (Savelsbergh 1994, sec. 2; Achterberg, "Constraint Integer Programming",
// PhD thesis, TU Berlin 2007, the chapter on domain propagation). Each row keeps its minimum
// and maximum activity over the current bounds - the finite part of each sum and how many
// contributions are infinite - updated by every bound change through the column's entries
// and restored exactly when a probe is undone. A queued row is rescanned only when those
// activities say the rescan can do something. On ex9 (MIPLIB 2017) each probe rescanned
// about 770,000 matrix entries: one fixed binary cascades through much of the model.
//
// WHY THE PASS-OVER IS EXACT. propagate_row() changes a bound of column k in row i only if
// the bound it computes beats the current one by more than set_upper()/set_lower() ignore:
// for an integer column with integral bounds by more than kIntegrality, for a continuous one
// by more than kProbingBoundMinStep. On the upper side with a > 0 that needs
//   (row_upper - (min - a lo_k)) / a < up_k - kIntegrality,
// that is min + |a| (up_k - lo_k) > row_upper + |a| kIntegrality, and the same with a < 0
// and, with max, on the lower side. So with span_i >= |a| (up_k - lo_k) for every member
// (taken over the bounds the prober starts from, which only ever tighten) and excess_i at
// most every member's |a| kIntegrality or |a| kProbingBoundMinStep, a row with
//   min + span_i <= row_upper + excess_i - drift_i  and  min <= row_upper - drift_i
// (and the mirror with max and row_lower) can neither tighten a bound nor be found violated,
// and the rescan would leave everything as it was. drift_i, 1e-9 times the row's largest
// possible activity magnitude, covers the difference between the kept sum and the rescan's
// fresh one by many orders of magnitude. A row with an infinite contribution on the side
// tested, or an integer member with a fractional bound (where the rounding argument fails),
// is always rescanned. The pass-over is charged the work the rescan would have been, so the
// work limits stop probing at the same probe as before.

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
         std::int64_t work_limit, bool row_activities)
      : model_(model),
        n_(model.num_cols()),
        by_row_(model.matrix),
        lo_(lower),
        up_(upper),
        work_limit_(work_limit),
        queued_(static_cast<std::size_t>(model.num_rows()), 0),
        stamp_(static_cast<std::size_t>(n_), -1),
        seen_lo_(static_cast<std::size_t>(n_), 0.0),
        seen_up_(static_cast<std::size_t>(n_), 0.0),
        row_activities_(row_activities) {
    if (row_activities_) start_activities();
  }

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
    result.rows_passed_over = rows_passed_over_;
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
      keep();
      probe_work_ = 0;
      const bool ok = fix(j, value) && propagate();
      if (!ok) {
        result->infeasible = true;
        return;
      }
      ++result->fixings;
      result->tightenings += static_cast<Count>(changed_columns()) - 1;
      keep();
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
    keep();
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
    keep();
  }

  /// Fix x_j = value on top of the global bounds, propagate, record the bounds that moved,
  /// and undo. False when the probe is infeasible.
  bool run_probe(Index j, double value, std::vector<Change>* changes) {
    keep();
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
    // The activities as they were, exactly: restored, not recomputed by reverse deltas.
    for (auto it = activity_trail_.rbegin(); it != activity_trail_.rend(); ++it) {
      activity_[static_cast<std::size_t>(it->row)] = it->before;
    }
    keep();
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
    if (row_activities_) note_moved(j, old, v, true);
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
    if (row_activities_) note_moved(j, old, v, false);
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
      if (row_activities_ && !may_imply(i)) {
        // Charged as the rescan it replaces, so the work limits are unchanged.
        const Index size = by_row_.row(i).size;
        work_ += size;
        probe_work_ += size;
        ++rows_passed_over_;
        continue;
      }
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

  /// The bounds as they stand are kept: the trails that would undo them are dropped.
  void keep() {
    trail_.clear();
    activity_trail_.clear();
  }

  /// Every row's activities over the starting bounds, and the span, excess and drift of the
  /// pass-over test (file comment).
  void start_activities() {
    const auto m = static_cast<std::size_t>(model_.num_rows());
    activity_.assign(m, Activity{});
    span_.assign(m, 0.0);
    excess_.assign(m, kInfinity);
    drift_.assign(m, 0.0);
    rescan_always_.assign(m, 0);
    for (Index i = 0; i < model_.num_rows(); ++i) {
      const auto r = static_cast<std::size_t>(i);
      const ColumnView row = by_row_.row(i);
      double magnitude = 0.0;
      for (Index k = 0; k < row.size; ++k) {
        const Index j = row.rows[k];
        const auto u = static_cast<std::size_t>(j);
        const double a = row.values[k];
        add(&activity_[r], a, lo_[u], false, +1);
        add(&activity_[r], a, up_[u], true, +1);
        magnitude += std::fabs(a) * std::max(finite(lo_[u]) ? std::fabs(lo_[u]) : 0.0,
                                             finite(up_[u]) ? std::fabs(up_[u]) : 0.0);
        if (std::fabs(a) <= tol::kZeroDrop) continue;
        const double range = finite(lo_[u]) && finite(up_[u]) ? up_[u] - lo_[u] : kInfinity;
        span_[r] = std::max(span_[r], std::fabs(a) * range);
        const bool integer = is_integer(j);
        excess_[r] = std::min(excess_[r], std::fabs(a) * (integer ? tol::kIntegrality
                                                                  : tol::kProbingBoundMinStep));
        if (integer && ((finite(lo_[u]) && lo_[u] != std::floor(lo_[u])) ||
                        (finite(up_[u]) && up_[u] != std::floor(up_[u])))) {
          rescan_always_[r] = 1;
        }
      }
      drift_[r] = kActivityDrift * (1.0 + magnitude);
    }
  }

  struct Activity {
    double min = 0.0;   ///< finite part of the minimum activity
    double max = 0.0;   ///< finite part of the maximum activity
    Index min_inf = 0;  ///< contributions to the minimum that are infinite
    Index max_inf = 0;  ///< contributions to the maximum that are infinite
  };
  struct ActivityChange {
    Index row;
    Activity before;
  };

  /// Add (sign +1) or remove (sign -1) bound `value` of a column with coefficient `a` on the
  /// side of `act` that bound feeds: a lower bound feeds the minimum when a > 0 and the
  /// maximum when a < 0, an upper bound the other way round.
  static void add(Activity* act, double a, double value, bool upper, int sign) {
    const bool to_min = (a > 0.0) != upper;
    double& sum = to_min ? act->min : act->max;
    Index& inf = to_min ? act->min_inf : act->max_inf;
    if (finite(value)) {
      sum += static_cast<double>(sign) * a * value;
    } else {
      inf += sign;
    }
  }

  /// Column j's lower or upper bound moved from `old` to `now`: every row of its column.
  void note_moved(Index j, double old, double now, bool upper) {
    const ColumnView column = model_.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto r = static_cast<std::size_t>(column.rows[k]);
      activity_trail_.push_back({column.rows[k], activity_[r]});
      add(&activity_[r], column.values[k], old, upper, -1);
      add(&activity_[r], column.values[k], now, upper, +1);
    }
  }

  /// False only when propagate_row(i) would change nothing and find nothing violated (the
  /// file comment gives the argument).
  [[nodiscard]] bool may_imply(Index i) const {
    const auto r = static_cast<std::size_t>(i);
    if (rescan_always_[r] != 0) return true;
    const Activity& act = activity_[r];
    const double upper = model_.row_upper[r];
    const double lower = model_.row_lower[r];
    const double drift = drift_[r];
    if (finite(upper) && act.min_inf <= 1) {
      if (act.min_inf == 1) return true;
      if (act.min > upper - drift) return true;
      if (act.min + span_[r] > upper + excess_[r] - drift) return true;
    }
    if (finite(lower) && act.max_inf <= 1) {
      if (act.max_inf == 1) return true;
      if (act.max < lower + drift) return true;
      if (act.max - span_[r] < lower - excess_[r] + drift) return true;
    }
    return false;
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
  /// 1e-9 of a row's largest possible activity magnitude: the margin the pass-over test
  /// leaves for the kept sums against the rescan's fresh ones (file comment).
  static constexpr double kActivityDrift = 1e-9;
  bool row_activities_ = false;
  std::vector<Activity> activity_;
  std::vector<ActivityChange> activity_trail_;  ///< activities before each change, in order
  std::vector<double> span_;
  std::vector<double> excess_;
  std::vector<double> drift_;
  std::vector<char> rescan_always_;
  Count rows_passed_over_ = 0;
};

}  // namespace

ProbingResult probe_binaries(const Model& model, const std::vector<double>& col_lower,
                             const std::vector<double>& col_upper, std::int64_t work_limit,
                             bool row_activities) {
  if (!model.has_integrality() || model.num_rows() == 0) {
    ProbingResult nothing;
    nothing.col_lower = col_lower;
    nothing.col_upper = col_upper;
    return nothing;
  }
  return Prober(model, col_lower, col_upper, work_limit, row_activities).run();
}

}  // namespace sankhya::presolve
