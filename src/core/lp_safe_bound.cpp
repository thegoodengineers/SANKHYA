// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the safe bound and certified gap of an optimal LP (#763).
//
// The bound is safe_dual_bound() (src/core/safe_bound.hpp, Neumaier & Shcherbina, Math.
// Programming 99 (2004)) of the solution's own row duals y, in minimise space. It is valid
// for ANY y and ANY column box that contains every feasible point, so this file is free to
// choose both, and it has to: a column with a zero reduced cost - basic, or nonbasic at a
// dual-degenerate vertex - has an outward-rounded interval around it that straddles zero,
// and a straddling interval needs BOTH of the column's bounds. Most LP columns have only
// x_j >= 0, so on Netlib the duals as reported prove no finite bound on most instances.
// Three attempts, each kept only when it improves on the last, stopping once the relative
// gap is within kCertifiedGapTarget:
//
//  1. y over the model's own bounds.
//
//  2. y over the box after bound propagation: every missing column bound that one row and
//     the other columns' bounds imply, round after round, outward-rounded
//     (propagate_missing_bounds(); Savelsbergh, ORSA J. Computing 6(4), 1994). Every
//     feasible point is in that box, so the bound over it is a bound. The derived bounds
//     are reported with the row each came from, for the verifier to re-derive.
//
//  3. A step along a dual direction that moves the basic columns' reduced costs off zero
//     to the side their bounds can price. With B the reported basis (structural and
//     logical columns), solve B'dy = s with s_j = -1 for a basic column bounded only below,
//     +1 for one bounded only above, 0 for every other basic column and for basic rows;
//     y + e dy leaves r_j - e s_j on the basic columns. Nonbasic reduced costs move by
//     e a_j'dy, which keeps the sign of any that is not itself zero. e runs from
//     kSafeBoundShiftFirst up by kSafeBoundShiftGrowth (times the largest term of c and
//     A'y); the first finite bound is kept. It costs about e |b'dy| of the bound, and
//     nothing needs dy to be accurate: whatever y + e dy comes out as, its bound is rigorous.
//
//  4. The exact rational duals of the reported basis (lp_exact_dual.hpp): every basic reduced
//     cost exactly zero, so no basic column needs a bound, and the bound, evaluated
//     exactly, is finite whenever the basis is exactly dual feasible over the box. The
//     multipliers are reported as exact fractions. Within option exact_seconds. When the
//     reported basis is not exactly dual feasible, the exact repair of #757 pivots to one
//     that is exactly optimal first, and its exact duals are used.
//
//  5. When 1-4 leave the bound infinite or the gap above the target, the one-sided columns
//     of that box are what is left: a zero reduced cost - basic, or nonbasic at a dual-
//     degenerate vertex - straddles zero, and moving y along a basis direction moves the
//     degenerate nonbasic ones to either side. So the LP is re-solved (certified_gap off,
//     so once) with each one-sided column's cost moved by delta = d * (the largest term of
//     its reduced cost) towards the side its bound prices: c - delta with only a lower
//     bound, c + delta with only an upper. The new optimum's duals, dual feasible for the
//     moved costs to the solver's tolerance, leave every such column's reduced cost against
//     the ORIGINAL costs about delta away from zero on the right side, and 1-3 run on them.
//     d runs from kSafeBoundCostPerturbationFirst up by kSafeBoundCostPerturbationGrowth to
//     kSafeBoundCostPerturbationLast. The bound gives up about delta per unit of those
//     columns' values; a free column (no bound after propagation) needs its reduced cost
//     exactly zero, which no floating-point y gives, and stays unpriced. Neumaier and
//     Shcherbina (sec. 3) point the same way for unbounded variables: a perturbed problem
//     whose dual has slack where a bound is missing. When the re-solve's float duals still
//     prove nothing, stage 4 runs on its basis with the moved costs: exact duals that are
//     dual feasible for the moved costs leave the original reduced costs delta clear.
//
// 2, 3, 4 and 5 are our additions to the paper's formula, not part of it.

#include "core/lp_safe_bound.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "core/lp_exact_dual.hpp"
#include "core/safe_bound.hpp"
#include "exact/exact_repair.hpp"
#include "la/lu.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

using Sz = std::size_t;
constexpr double kInf = std::numeric_limits<double>::infinity();

/// a + b rounded towards +inf (up) or -inf: exact when Knuth's TwoSum finds no rounding
/// error, one ulp outward otherwise (the rounding argument of safe_bound.hpp).
double add_directed(double a, double b, bool up) {
  const double s = a + b;
  if (!std::isfinite(s)) return s;
  const double bb = s - a;
  const double error = (a - (s - bb)) + (b - bb);
  if (up ? error <= 0.0 : error >= 0.0) return s;
  return std::nextafter(s, up ? kInf : -kInf);
}

/// dy with B'dy = s (see the file header) for the box [lower, upper], indexed by row; empty
/// when the solution has no usable basis or no basic column needs moving.
std::vector<double> sign_fixing_direction(const Model& model, const Solution& solution,
                                          const std::vector<double>& lower,
                                          const std::vector<double>& upper) {
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (solution.col_status.size() != static_cast<Sz>(n) ||
      solution.row_status.size() != static_cast<Sz>(m)) {
    return {};
  }
  std::vector<LuColumn> columns;
  std::vector<double> sign;
  std::vector<Index> logical_row;
  logical_row.reserve(static_cast<Sz>(m));  // never reallocates: the columns point into it
  static const double kMinusOne = -1.0;
  bool any = false;
  for (Index j = 0; j < n; ++j) {
    if (solution.col_status[static_cast<Sz>(j)] != BasisStatus::kBasic) continue;
    const ColumnView col = model.matrix.column(j);
    columns.push_back({col.rows, col.values, col.size});
    const bool has_lower = is_finite_bound(lower[static_cast<Sz>(j)]);
    const bool has_upper = is_finite_bound(upper[static_cast<Sz>(j)]);
    sign.push_back(has_lower && !has_upper ? -1.0 : (has_upper && !has_lower ? 1.0 : 0.0));
    any = any || sign.back() != 0.0;
  }
  for (Index i = 0; i < m; ++i) {
    if (solution.row_status[static_cast<Sz>(i)] != BasisStatus::kBasic) continue;
    logical_row.push_back(i);
    columns.push_back({&logical_row.back(), &kMinusOne, 1});
    sign.push_back(0.0);
  }
  if (!any || columns.size() != static_cast<Sz>(m)) return {};
  SparseLu lu;
  if (!lu.factorize(columns, m, tol::kPivotTolerance, tol::kMarkowitzThreshold)) return {};
  lu.solve_transpose(sign.data());
  for (const double v : sign) {
    if (!std::isfinite(v)) return {};
  }
  return sign;
}

}  // namespace

void attach_safe_lower_bound(const Model& model, const Options& options, Solution* solution) {
  if (solution->status != SolveStatus::kOptimal || model.has_integrality() ||
      model.has_quadratic_objective() ||
      solution->row_dual.size() != static_cast<Sz>(model.num_rows()) ||
      !options.get_bool("certified_gap")) {
    return;
  }
  const bool maximize = model.sense == ObjSense::kMaximize;
  const double sense = model.sense_multiplier();
  std::vector<double> cost(model.col_cost.size());
  for (Sz j = 0; j < cost.size(); ++j) cost[j] = sense * model.col_cost[j];
  SafeBoundProblem problem;
  problem.matrix = &model.matrix;
  problem.cost = cost;
  problem.row_lower = model.row_lower;
  problem.row_upper = model.row_upper;
  problem.col_lower = model.col_lower;
  problem.col_upper = model.col_upper;

  // The model-sense bound of a minimise-space value, the offset added outward.
  const auto to_model_sense = [&](double value) {
    return maximize ? add_directed(model.objective_offset, -value, true)
                    : add_directed(model.objective_offset, value, false);
  };
  const auto relative_gap = [&](double value) {
    const double bound = to_model_sense(value);
    return (maximize ? bound - solution->objective : solution->objective - bound) /
           std::max(1.0, std::fabs(solution->objective));
  };

  SafeBound best;
  SafeBoundDetail detail;
  bool best_uses_propagation = false;
  std::vector<std::string> exact_multipliers;  // set when stage 4 gave the best bound
  bool exact_timed_out = false;                // stage 4 ran out of budget: do not retry it
  const auto good_enough = [&] {
    return std::isfinite(best.value) && relative_gap(best.value) <= tol::kCertifiedGapTarget;
  };
  const auto consider = [&](const SafeBound& candidate, SafeBoundDetail&& candidate_detail,
                            bool uses_propagation) {
    if (!std::isfinite(candidate.value)) return;
    if (std::isfinite(best.value) && candidate.value <= best.value) return;
    best = candidate;
    detail = std::move(candidate_detail);
    best_uses_propagation = uses_propagation;
    exact_multipliers.clear();
  };

  // The propagated box, computed once, the first time a stage needs it.
  std::vector<double> lower = model.col_lower;
  std::vector<double> upper = model.col_upper;
  std::vector<PropagatedBound> propagated;
  bool propagation_ran = false;
  SafeBoundProblem boxed = problem;

  // Stages 1-3 of the file header for the duals y (minimise space) and the basis `source`
  // carries, each run only while the bound is not yet good enough.
  const auto try_duals = [&](const std::vector<double>& y, const Solution& source) {
    SafeBoundDetail own_detail;
    const SafeBound own = safe_dual_bound(problem, y, true, &own_detail);
    consider(own, std::move(own_detail), false);
    if (good_enough()) return;
    if (!propagation_ran) {
      propagation_ran = true;
      propagated =
          propagate_missing_bounds(problem, &lower, &upper, tol::kSafeBoundPropagationPasses);
      boxed.col_lower = lower;
      boxed.col_upper = upper;
    }
    if (!propagated.empty()) {
      SafeBoundDetail boxed_detail;
      const SafeBound candidate = safe_dual_bound(boxed, y, true, &boxed_detail);
      consider(candidate, std::move(boxed_detail), true);
      if (good_enough()) return;
    }
    const std::vector<double> dy = sign_fixing_direction(model, source, lower, upper);
    if (dy.empty()) return;
    // The scale of a reduced cost's terms, which its rounding width is a fraction of.
    double scale = 1.0;
    for (Index j = 0; j < model.num_cols(); ++j) {
      scale = std::max(scale, std::fabs(cost[static_cast<Sz>(j)]));
      const ColumnView col = model.matrix.column(j);
      for (Index k = 0; k < col.size; ++k) {
        scale = std::max(scale, std::fabs(col.values[k] * y[static_cast<Sz>(col.rows[k])]));
      }
    }
    std::vector<double> shifted(y.size());
    for (double e = tol::kSafeBoundShiftFirst;
         e <= tol::kSafeBoundShiftLast * (1.0 + tol::kSafeBoundShiftFirst);
         e *= tol::kSafeBoundShiftGrowth) {
      for (Sz i = 0; i < y.size(); ++i) shifted[i] = y[i] + e * scale * dy[i];
      SafeBoundDetail shifted_detail;
      const SafeBound candidate = safe_dual_bound(boxed, shifted, true, &shifted_detail);
      if (!std::isfinite(candidate.value)) continue;
      consider(candidate, std::move(shifted_detail), !propagated.empty());
      return;
    }
  };

  std::vector<double> y(solution->row_dual.size());
  for (Sz i = 0; i < y.size(); ++i) y[i] = sense * solution->row_dual[i];
  try_duals(y, *solution);

  // 4. The exact duals of a basis, over the box stage 2 left: the reported one here, and in
  // stage 5 the re-solve's, for the costs it was optimal for.
  const auto try_exact = [&](std::span<const double> dual_cost, const Solution& source) {
    if (exact_timed_out) return;
    ExactDualBound exact_bound = exact_dual_bound(model, dual_cost, cost, lower, upper, source,
                                                  options.get_double("exact_seconds"));
    exact_timed_out = exact_bound.timed_out;
    if (exact_bound.proved && (!std::isfinite(best.value) || exact_bound.value > best.value)) {
      best.value = exact_bound.value;
      best_uses_propagation = !propagated.empty();
      exact_multipliers = std::move(exact_bound.multipliers);
    }
  };
  if (!good_enough()) try_exact(cost, *solution);
  // The reported basis is often optimal only to tolerance: exactly, a nonbasic reduced cost
  // of -1e-14 on a column with no upper bound, and then its exact duals prove nothing. The
  // exact repair (#757, src/exact/exact_repair.hpp) pivots from it, in rational arithmetic,
  // to a basis that is exactly optimal, whose exact duals are dual feasible by construction.
  if (!good_enough() && !exact_timed_out) {
    const exact::RepairResult repair =
        exact::repair_basis_exact(model, *solution, options.get_double("exact_seconds"));
    if (repair.verdict == exact::ExactVerdict::kVerified && repair.changed()) {
      Solution repaired;
      repaired.col_status = repair.col_status;
      repaired.row_status = repair.row_status;
      try_exact(cost, repaired);
    }
  }

  // 5. The duals of the same LP with its one-sided columns' costs moved by delta towards
  // the side their bound prices (see the file header), re-solved from scratch. Each try is
  // a whole solve, so it runs only with certified_gap_resolve on, and all tries together
  // share exact_seconds and whatever is left of the caller's time_limit.
  double resolve_budget = options.get_double("exact_seconds");
  const double caller_limit = options.get_double("time_limit");
  if (std::isfinite(caller_limit)) {
    resolve_budget = std::min(resolve_budget, caller_limit - solution->solve_seconds);
  }
  if (!good_enough() && options.get_bool("certified_gap_resolve") && resolve_budget > 0.0) {
    Options inner = options;
    inner.set_bool("certified_gap", false);  // no recursion
    inner.set_bool("log_to_console", false);
    for (double delta = tol::kSafeBoundCostPerturbationFirst;
         delta <= tol::kSafeBoundCostPerturbationLast * (1.0 + tol::kSafeBoundShiftFirst) &&
         !good_enough() && resolve_budget > 0.0;
         delta *= tol::kSafeBoundCostPerturbationGrowth) {
      inner.set_double("time_limit", resolve_budget);
      Model perturbed = model;
      for (Index j = 0; j < model.num_cols(); ++j) {
        const auto uj = static_cast<Sz>(j);
        const bool has_lower = is_finite_bound(lower[uj]);
        const bool has_upper = is_finite_bound(upper[uj]);
        if (has_lower == has_upper) continue;  // boxed (any sign is priced) or free
        double scale = std::max(1.0, std::fabs(cost[uj]));
        const ColumnView col = model.matrix.column(j);
        for (Index k = 0; k < col.size; ++k) {
          scale = std::max(scale, std::fabs(col.values[k] * y[static_cast<Sz>(col.rows[k])]));
        }
        // Minimise space: c - delta where only a lower bound exists, so the new optimum's
        // reduced cost against the ORIGINAL cost is at least delta; c + delta for an upper.
        const double moved = (has_lower ? -1.0 : 1.0) * delta * scale;
        perturbed.col_cost[uj] = sense * (cost[uj] + moved);
      }
      const Solution resolved = solve(perturbed, inner);
      resolve_budget -= resolved.solve_seconds;
      // Lowering a cost makes any zero-cost ray of the original improving: unbounded for
      // the smallest move means unbounded for every larger one.
      if (resolved.status == SolveStatus::kUnbounded) break;
      if (resolved.status != SolveStatus::kOptimal ||
          resolved.row_dual.size() != static_cast<Sz>(model.num_rows())) {
        continue;
      }
      std::vector<double> resolved_y(resolved.row_dual.size());
      for (Sz i = 0; i < resolved_y.size(); ++i) resolved_y[i] = sense * resolved.row_dual[i];
      try_duals(resolved_y, resolved);
      if (!good_enough()) {
        std::vector<double> moved_cost(cost.size());
        for (Sz j = 0; j < cost.size(); ++j) moved_cost[j] = sense * perturbed.col_cost[j];
        try_exact(moved_cost, resolved);
      }
    }
  }

  solution->safe_column_bounds.clear();
  if (!std::isfinite(best.value)) {
    solution->safe_lower_bound = maximize ? kInf : -kInf;
    solution->certified_gap = kInf;
    solution->certified_relative_gap = kInf;
    solution->safe_multipliers.clear();
    solution->safe_multipliers_exact.clear();
    return;
  }
  solution->safe_lower_bound = to_model_sense(best.value);
  solution->certified_gap = maximize ? solution->safe_lower_bound - solution->objective
                                     : solution->objective - solution->safe_lower_bound;
  solution->certified_relative_gap = relative_gap(best.value);
  // In the model's sense, as the file's duals are: negation is exact.
  solution->safe_multipliers_exact.clear();
  if (!exact_multipliers.empty()) {
    solution->safe_multipliers.clear();
    for (std::string& v : exact_multipliers) {
      if (maximize && v.rfind("0/", 0) != 0) v = v[0] == '-' ? v.substr(1) : "-" + v;
      solution->safe_multipliers_exact.push_back(std::move(v));
    }
  } else {
    solution->safe_multipliers = std::move(detail.multipliers);
    for (double& v : solution->safe_multipliers) v *= sense;
  }
  if (best_uses_propagation) {
    for (const PropagatedBound& b : propagated) {
      solution->safe_column_bounds.push_back({b.column, b.row, b.is_upper, b.value});
    }
  }
}

}  // namespace sankhya
