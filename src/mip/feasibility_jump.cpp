// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Feasibility Jump (#506). Luteberget and Sartor, "Feasibility Jump: an LP-free
// Lagrangian MIP heuristic", Mathematical Programming Computation 15 (2023). See
// feasibility_jump.hpp for the method and for where this implementation departs from it.
#include "feasibility_jump.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>

#include "sankhya/sparse.hpp"

namespace sankhya::mip {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Distance of an activity from its row's interval; zero inside it.
double violation(double activity, double lower, double upper) {
  double v = 0.0;
  if (is_finite_bound(lower) && activity < lower) v += lower - activity;
  if (is_finite_bound(upper) && activity > upper) v += activity - upper;
  return v;
}

/// A breakpoint of one column's weighted violation: where one row's activity crosses one of
/// its bounds, and how much the slope rises there.
struct Breakpoint {
  double value;
  double slope_increase;
};

struct Jump {
  Index column = -1;
  double value = 0.0;
  double score = 0.0;  ///< decrease of the weighted violation plus the weighted objective
};

class Search {
 public:
  Search(const Model& model, const FeasibilityJumpSettings& settings)
      : model_(model), settings_(settings), rows_(model.matrix), rng_(settings.seed) {}

  FeasibilityJumpResult run(const std::vector<double>& start);

 private:
  bool set_up(const std::vector<double>& start);
  void recompute_activities();
  void set_membership(Index row);
  [[nodiscard]] double delta(Index j, double value);
  [[nodiscard]] Jump best_jump(Index j);
  void apply(Index j, double value);
  void bump_weights();
  [[nodiscard]] double objective() const;
  [[nodiscard]] bool out_of_work() {
    if (result_.work >= settings_.work_limit) return true;
    // The work budget alone let an unbounded mip_fj_work ignore the time limit and Ctrl-C
    // until it ran out (review of #635); the caller's stop is polled on a work stride.
    if (settings_.should_stop && result_.work >= next_poll_) {
      next_poll_ = result_.work + tol::kFeasibilityJumpPollWork;
      return settings_.should_stop();
    }
    return false;
  }
  Count next_poll_ = 0;

  const Model& model_;
  const FeasibilityJumpSettings& settings_;
  const CsrView rows_;
  std::mt19937_64 rng_;
  FeasibilityJumpResult result_;

  std::vector<double> lower_;  ///< column bounds, integer ones rounded inwards
  std::vector<double> upper_;
  std::vector<double> cost_;  ///< in minimise space
  std::vector<double> x_;
  std::vector<double> activity_;
  std::vector<double> weight_;
  double objective_weight_ = 0.0;
  std::vector<Index> violated_;     ///< rows violated beyond the tolerance
  std::vector<Index> violated_at_;  ///< a row's place in violated_, or -1
  std::vector<Index> costed_;       ///< movable columns with a nonzero cost
  std::vector<Breakpoint> breaks_;  ///< scratch for best_jump()
};

bool Search::set_up(const std::vector<double>& start) {
  const Index n = model_.num_cols();
  const Index m = model_.num_rows();
  const double sense = model_.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  lower_.resize(static_cast<std::size_t>(n));
  upper_.resize(static_cast<std::size_t>(n));
  cost_.resize(static_cast<std::size_t>(n));
  x_.resize(static_cast<std::size_t>(n));
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    double lo = model_.col_lower[u];
    double hi = model_.col_upper[u];
    if (model_.col_type[u] == VarType::kInteger) {
      if (is_finite_bound(lo)) lo = std::ceil(lo - settings_.integrality_tolerance);
      if (is_finite_bound(hi)) hi = std::floor(hi + settings_.integrality_tolerance);
    }
    if (std::isnan(lo) || std::isnan(hi) || lo > hi) return false;  // no point exists
    lower_[u] = lo;
    upper_[u] = hi;
    cost_[u] = sense * model_.col_cost[u];
    double v = u < start.size() && std::isfinite(start[u]) ? start[u] : 0.0;
    if (model_.col_type[u] == VarType::kInteger) v = std::round(v);
    x_[u] = std::clamp(v, lo, hi);
    if (settings_.use_objective && cost_[u] != 0.0 && lo < hi) costed_.push_back(j);
  }
  activity_.assign(static_cast<std::size_t>(m), 0.0);
  weight_.assign(static_cast<std::size_t>(m), 1.0);
  violated_at_.assign(static_cast<std::size_t>(m), -1);
  recompute_activities();
  return true;
}

void Search::recompute_activities() {
  std::fill(activity_.begin(), activity_.end(), 0.0);
  if (model_.num_rows() > 0) model_.matrix.multiply(x_.data(), activity_.data());
  result_.work += model_.num_nonzeros() + model_.num_rows();
  violated_.clear();
  std::fill(violated_at_.begin(), violated_at_.end(), -1);
  for (Index i = 0; i < model_.num_rows(); ++i) set_membership(i);
}

void Search::set_membership(Index row) {
  const auto u = static_cast<std::size_t>(row);
  const bool now = violation(activity_[u], model_.row_lower[u], model_.row_upper[u]) >
                   settings_.feasibility_tolerance;
  const Index at = violated_at_[u];
  if (now && at < 0) {
    violated_at_[u] = static_cast<Index>(violated_.size());
    violated_.push_back(row);
  } else if (!now && at >= 0) {
    const Index last = violated_.back();
    violated_[static_cast<std::size_t>(at)] = last;
    violated_at_[static_cast<std::size_t>(last)] = at;
    violated_.pop_back();
    violated_at_[u] = -1;
  }
}

double Search::delta(Index j, double value) {
  const auto u = static_cast<std::size_t>(j);
  const double step = value - x_[u];
  double change = objective_weight_ * cost_[u] * step;
  const ColumnView column = model_.matrix.column(j);
  for (Index k = 0; k < column.size; ++k) {
    const auto i = static_cast<std::size_t>(column.rows[k]);
    const double before = activity_[i];
    const double after = before + column.values[k] * step;
    change += weight_[i] * (violation(after, model_.row_lower[i], model_.row_upper[i]) -
                            violation(before, model_.row_lower[i], model_.row_upper[i]));
  }
  result_.work += column.size + 1;
  return change;
}

Jump Search::best_jump(Index j) {
  const auto u = static_cast<std::size_t>(j);
  Jump jump;
  if (lower_[u] >= upper_[u]) return jump;
  const double xj = x_[u];
  // The weighted violation as a function of x_j alone is convex and piecewise linear. Far
  // to the left every row whose activity then falls below a finite bound pulls with slope
  // -w|a|; each such bound, and each bound on the other side, adds w|a| to the slope where
  // the activity crosses it. The minimisers are where the slope turns non-negative.
  double slope = objective_weight_ * cost_[u];
  breaks_.clear();
  const ColumnView column = model_.matrix.column(j);
  for (Index k = 0; k < column.size; ++k) {
    const double a = column.values[k];
    if (a == 0.0) continue;
    const auto i = static_cast<std::size_t>(column.rows[k]);
    const double rest = activity_[i] - a * xj;
    const double pull = weight_[i] * std::fabs(a);
    const double first = a > 0.0 ? model_.row_lower[i] : model_.row_upper[i];
    const double second = a > 0.0 ? model_.row_upper[i] : model_.row_lower[i];
    if (is_finite_bound(first)) {
      slope -= pull;
      breaks_.push_back({(first - rest) / a, pull});
    }
    if (is_finite_bound(second)) breaks_.push_back({(second - rest) / a, pull});
  }
  result_.work += column.size + 1;
  std::sort(breaks_.begin(), breaks_.end(),
            [](const Breakpoint& p, const Breakpoint& q) { return p.value < q.value; });

  // [left, right] is the set of minimisers of the function without the column's box.
  double left = -kInf;
  double right = -kInf;
  if (slope >= 0.0) {
    right = slope > 0.0 ? -kInf : (breaks_.empty() ? kInf : breaks_.front().value);
  } else {
    left = right = kInf;  // still falling after the last breakpoint
    for (std::size_t k = 0; k < breaks_.size(); ++k) {
      slope += breaks_[k].slope_increase;
      if (k + 1 < breaks_.size() && breaks_[k + 1].value == breaks_[k].value) continue;
      if (slope >= 0.0) {
        left = breaks_[k].value;
        right = slope > 0.0 ? left : (k + 1 < breaks_.size() ? breaks_[k + 1].value : kInf);
        break;
      }
    }
  }
  // The box's best point: the minimiser nearest the current value, or, when every minimiser
  // is outside the box, the box's end nearest them (convexity).
  double target;
  if (left > upper_[u]) {
    target = upper_[u];
  } else if (right < lower_[u]) {
    target = lower_[u];
  } else {
    target = std::clamp(xj, std::max(left, lower_[u]), std::min(right, upper_[u]));
  }
  if (!std::isfinite(target)) return jump;

  double candidates[2] = {target, target};
  int count = 1;
  if (model_.col_type[u] == VarType::kInteger) {
    // The integer minimiser of a convex function is next to a real one.
    candidates[0] = std::clamp(std::floor(target), lower_[u], upper_[u]);
    candidates[1] = std::clamp(std::ceil(target), lower_[u], upper_[u]);
    count = candidates[1] != candidates[0] ? 2 : 1;
  }
  for (int c = 0; c < count; ++c) {
    const double value = candidates[c];
    if (value == xj) continue;
    const double score = -delta(j, value);
    const bool nearer = jump.column >= 0 && score == jump.score &&
                        std::fabs(value - xj) < std::fabs(jump.value - xj);
    if (jump.column < 0 || score > jump.score || nearer) {
      jump.column = j;
      jump.value = value;
      jump.score = score;
    }
  }
  return jump;
}

void Search::apply(Index j, double value) {
  const auto u = static_cast<std::size_t>(j);
  const double step = value - x_[u];
  x_[u] = value;
  const ColumnView column = model_.matrix.column(j);
  for (Index k = 0; k < column.size; ++k) {
    const Index i = column.rows[k];
    activity_[static_cast<std::size_t>(i)] += column.values[k] * step;
    set_membership(i);
  }
  result_.work += column.size + 1;
  ++result_.moves;
}

void Search::bump_weights() {
  // The paper's update at a local minimum: every violated row's weight gains 1; with no row
  // violated, the objective's does.
  if (violated_.empty()) {
    objective_weight_ += 1.0;
  } else {
    for (const Index i : violated_) weight_[static_cast<std::size_t>(i)] += 1.0;
  }
  result_.work += static_cast<Count>(violated_.size()) + 1;
  ++result_.weight_updates;
}

double Search::objective() const {
  double value = 0.0;
  for (std::size_t j = 0; j < x_.size(); ++j) value += cost_[j] * x_[j];
  return value;
}

FeasibilityJumpResult Search::run(const std::vector<double>& start) {
  if (!set_up(start)) return result_;
  double best = kInf;
  bool checked = false;  // the current point has been re-measured since the last move
  while (!out_of_work()) {
    if (violated_.empty() && !checked) {
      // Feasible by the running activities. Re-measure every row from scratch before
      // believing it: the running sums drift, and a proposal should not.
      recompute_activities();
      checked = true;
      if (!violated_.empty()) continue;
      const double value = objective();
      if (result_.points.empty() ||
          value < best - tol::kFeasibilityJumpMinScore * std::max(1.0, std::fabs(best))) {
        result_.points.push_back(x_);
        if (settings_.on_point) settings_.on_point(x_);
        best = value;
      }
      if (costed_.empty()) break;  // nothing left to improve
    }

    Jump chosen;
    for (int s = 0; s < tol::kFeasibilityJumpSample; ++s) {
      ++result_.work;
      Index j = -1;
      if (!violated_.empty()) {
        std::uniform_int_distribution<std::size_t> pick_row(0, violated_.size() - 1);
        const ColumnView row = rows_.row(violated_[pick_row(rng_)]);
        if (row.size == 0) continue;
        std::uniform_int_distribution<Index> pick_entry(0, row.size - 1);
        j = row.rows[pick_entry(rng_)];
      } else {
        std::uniform_int_distribution<std::size_t> pick_column(0, costed_.size() - 1);
        j = costed_[pick_column(rng_)];
      }
      const Jump jump = best_jump(j);
      if (jump.column >= 0 && (chosen.column < 0 || jump.score > chosen.score)) chosen = jump;
    }
    if (chosen.column >= 0 && chosen.score > tol::kFeasibilityJumpMinScore) {
      apply(chosen.column, chosen.value);
      checked = false;
    } else {
      bump_weights();
    }
  }
  return result_;
}

}  // namespace

FeasibilityJumpResult feasibility_jump(const Model& model, const std::vector<double>& start,
                                       const FeasibilityJumpSettings& settings) {
  Search search(model, settings);
  return search.run(start);
}

std::vector<double> feasibility_jump_zero_start(const Model& model) {
  std::vector<double> x(static_cast<std::size_t>(model.num_cols()), 0.0);
  for (std::size_t j = 0; j < x.size(); ++j) {
    double lo = model.col_lower[j];
    double hi = model.col_upper[j];
    if (model.col_type[j] == VarType::kInteger) {
      if (is_finite_bound(lo)) lo = std::ceil(lo - tol::kIntegrality);
      if (is_finite_bound(hi)) hi = std::floor(hi + tol::kIntegrality);
    }
    if (is_finite_bound(lo) && lo > 0.0) x[j] = lo;
    if (is_finite_bound(hi) && hi < 0.0) x[j] = hi;
  }
  return x;
}

}  // namespace sankhya::mip
