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
// 2 and 3 are our additions to the paper's formula, not part of it.

#include "core/lp_safe_bound.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/safe_bound.hpp"
#include "la/lu.hpp"
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

void attach_safe_lower_bound(const Model& model, Solution* solution) {
  if (solution->status != SolveStatus::kOptimal || model.has_integrality() ||
      model.has_quadratic_objective() ||
      solution->row_dual.size() != static_cast<Sz>(model.num_rows())) {
    return;
  }
  const bool maximize = model.sense == ObjSense::kMaximize;
  const double sense = model.sense_multiplier();
  std::vector<double> cost(model.col_cost.size());
  for (Sz j = 0; j < cost.size(); ++j) cost[j] = sense * model.col_cost[j];
  std::vector<double> y(solution->row_dual.size());
  for (Sz i = 0; i < y.size(); ++i) y[i] = sense * solution->row_dual[i];
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
  const auto good_enough = [&](const SafeBound& b) {
    return std::isfinite(b.value) && relative_gap(b.value) <= tol::kCertifiedGapTarget;
  };

  // 1. The duals over the model's own bounds.
  SafeBoundDetail detail;
  SafeBound best = safe_dual_bound(problem, y, true, &detail);
  bool best_uses_propagation = false;
  const auto consider = [&](const SafeBound& candidate, SafeBoundDetail&& candidate_detail,
                            bool uses_propagation) {
    if (!std::isfinite(candidate.value)) return;
    if (std::isfinite(best.value) && candidate.value <= best.value) return;
    best = candidate;
    detail = std::move(candidate_detail);
    best_uses_propagation = uses_propagation;
  };

  // 2. Over the propagated box.
  std::vector<double> lower = model.col_lower;
  std::vector<double> upper = model.col_upper;
  std::vector<PropagatedBound> propagated;
  if (!good_enough(best)) {
    propagated =
        propagate_missing_bounds(problem, &lower, &upper, tol::kSafeBoundPropagationPasses);
    if (!propagated.empty()) {
      problem.col_lower = lower;
      problem.col_upper = upper;
      SafeBoundDetail candidate_detail;
      const SafeBound candidate = safe_dual_bound(problem, y, true, &candidate_detail);
      consider(candidate, std::move(candidate_detail), true);
    }
  }

  // 3. Shifted duals, over the box 2 left (the model's own when it found nothing).
  if (!good_enough(best)) {
    const std::vector<double> dy = sign_fixing_direction(model, *solution, lower, upper);
    if (!dy.empty()) {
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
        SafeBoundDetail candidate_detail;
        const SafeBound candidate = safe_dual_bound(problem, shifted, true, &candidate_detail);
        if (!std::isfinite(candidate.value)) continue;
        consider(candidate, std::move(candidate_detail), !propagated.empty());
        break;
      }
    }
  }

  solution->safe_column_bounds.clear();
  if (!std::isfinite(best.value)) {
    solution->safe_lower_bound = maximize ? kInf : -kInf;
    solution->certified_gap = kInf;
    solution->certified_relative_gap = kInf;
    solution->safe_multipliers.clear();
    return;
  }
  solution->safe_lower_bound = to_model_sense(best.value);
  solution->certified_gap = maximize ? solution->safe_lower_bound - solution->objective
                                     : solution->objective - solution->safe_lower_bound;
  solution->certified_relative_gap = relative_gap(best.value);
  // In the model's sense, as the file's duals are: negation is exact.
  solution->safe_multipliers = std::move(detail.multipliers);
  for (double& v : solution->safe_multipliers) v *= sense;
  if (best_uses_propagation) {
    for (const PropagatedBound& b : propagated) {
      solution->safe_column_bounds.push_back({b.column, b.row, b.is_upper, b.value});
    }
  }
}

}  // namespace sankhya
