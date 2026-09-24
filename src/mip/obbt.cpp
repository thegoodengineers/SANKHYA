// SPDX-License-Identifier: Apache-2.0
// SANKHYA - optimality-based bound tightening (OBBT) at the root (#515).
//
// Reference:
//   Gleixner, Berthold, Muller and Weltge, "Three enhancements for
//   optimization-based bound tightening", J. Global Optimization 67, 2017.
//
// ALGORITHM.  For each variable x_j that is not already fixed (range >=
// tol::kObbtFixedRange), add a temporary objective row when an incumbent is known
// (c'x <= incumbent - eps minimising, c'x >= incumbent + eps maximising), then solve:
//
//   (min LP)  min x_j  s.t. Ax <= b [, c'x <= z_cutoff]
//   (max LP)  max x_j  s.t. Ax <= b [, c'x <= z_cutoff]
//
// The optimal values are valid lower and upper bounds for x_j over the whole
// feasible region intersected with the cutoff, once moved outward by the LP's own
// tolerance (and rounded inward to an integer for an integer column).  If either LP is
// infeasible the model is infeasible (or the problem is solved by the cutoff); we stop and
// leave the existing bounds intact.
//
// WARM STARTING.  After each min/max pair the basis from the max-LP is reused
// as the warm start for the next min-LP (same constraint matrix, cost just
// changed from +e_j to the next column); this keeps the iteration count low.
//
// WORK LIMIT.  We stop after `mip_obbt_max_iters` LP solves (default 100) so
// OBBT does not dominate the root budget on large models.

#include "obbt.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/primal_simplex.hpp"

namespace sankhya::mip {

namespace {

/// Build a copy of `model` with a single extra row appended on the objective, in the
/// model's sense: c'x <= incumbent - eps when minimising, c'x >= incumbent + eps when
/// maximising, so only strictly improving points remain. The row indices of the original
/// model are preserved.
Model add_cutoff_row(const Model& model, double incumbent) {
  Model m = model;
  const Index old_rows = m.num_rows();
  const Index cols = m.num_cols();
  SparseMatrix mat(old_rows + 1, cols);
  for (Index j = 0; j < cols; ++j) {
    const ColumnView view = m.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) mat.add_entry(view.rows[k], j, view.values[k]);
    if (std::fabs(m.col_cost[static_cast<std::size_t>(j)]) > tol::kZeroDrop) {
      mat.add_entry(old_rows, j, m.col_cost[static_cast<std::size_t>(j)]);
    }
  }
  mat.finalize();
  m.matrix = std::move(mat);
  m.resize_rows(old_rows + 1);
  const auto row = static_cast<std::size_t>(old_rows);
  if (model.sense == ObjSense::kMaximize) {
    m.row_lower[row] = incumbent + tol::kObbtCutoffEpsilon;
    m.row_upper[row] = kInfinity;
  } else {
    m.row_lower[row] = -kInfinity;
    m.row_upper[row] = incumbent - tol::kObbtCutoffEpsilon;
  }
  return m;
}

/// A bound read off an LP solved to tolerance, moved outward so it cannot cut off a point
/// the exact LP admits: an integer column's bound is rounded to the nearest integer it
/// cannot exclude (within the integrality tolerance), a continuous one is relaxed by the
/// primal feasibility tolerance, relative to its magnitude.
double safe_lower(double value, bool integer) {
  if (integer) return std::ceil(value - tol::kIntegrality);
  return value - tol::kPrimalFeasibility * std::max(1.0, std::fabs(value));
}
double safe_upper(double value, bool integer) {
  if (integer) return std::floor(value + tol::kIntegrality);
  return value + tol::kPrimalFeasibility * std::max(1.0, std::fabs(value));
}

}  // namespace

ObbtResult obbt_root(Model& model, const Options& options, Logger& logger, double incumbent) {
  ObbtResult result;

  const bool obbt_on = options.get_bool("mip_obbt");
  if (!obbt_on) return result;

  const int max_iters = static_cast<int>(options.get_int("mip_obbt_max_iters"));
  const Index n = model.num_cols();

  // Build the LP used for OBBT: original model possibly augmented with a cutoff row.
  const bool have_cutoff = std::isfinite(incumbent);
  Model lp = have_cutoff ? add_cutoff_row(model, incumbent) : model;
  // The probes are LPs: a MIQP's Hessian is not part of min / max x_j.
  lp.hessian = SparseMatrix(n, n);
  lp.hessian.finalize();

  // Silent options for node-like solves.
  Options lp_opts = options;
  lp_opts.set_bool("log_to_console", false);

  // The scaling cache is built per probe: solve_primal_simplex requires it to come from the
  // SAME cost vector, and every probe has a different one. Built once from the model's own
  // costs, the "max x_j" probe minimised the model's objective instead and its optimum was
  // written back as x_j's upper bound (review of #631: 2 x <= 7 gave x <= 0).

  WarmStart warm;  // reused across solves
  int solves = 0;
  bool lp_infeasible = false;  // set when the cutoff row makes the LP infeasible

  for (Index j = 0; j < n && solves < max_iters && !lp_infeasible; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = lp.col_lower[u];
    const double hi = lp.col_upper[u];
    if (hi - lo < tol::kObbtFixedRange) continue;  // already fixed or near-fixed
    const bool integer = lp.col_type[u] == VarType::kInteger;

    // ---- min x_j ----------------------------------------------------------------
    if (solves < max_iters) {
      lp.col_cost.assign(static_cast<std::size_t>(n), 0.0);
      lp.col_cost[u] = 1.0;
      lp.sense = ObjSense::kMinimize;

      const Solution sol =
          solve_primal_simplex(lp, lp_opts, logger, build_node_scaling(lp, lp_opts), nullptr,
                               warm.empty() ? nullptr : &warm);
      ++solves;

      if (sol.status == SolveStatus::kOptimal) {
        const double new_lo = safe_lower(sol.col_value[u], integer);
        if (new_lo > lo + tol::kObbtMinimumTightening && new_lo <= lp.col_upper[u]) {
          lp.col_lower[u] = new_lo;
          ++result.bounds_tightened;
        }
        warm = {sol.col_status, sol.row_status};
      } else if (sol.status == SolveStatus::kInfeasible) {
        // The cutoff row made the LP infeasible (no strictly improving point): the LP
        // relaxation bound already meets the cutoff, so every remaining probe will
        // also be infeasible.  Stop early rather than exhausting the LP budget.
        warm = {};
        lp_infeasible = true;
      } else {
        // Unbounded (x_j has no finite lower bound) or numerical issue: cannot
        // tighten this bound, but other variables may still be tightenable.
        warm = {};
      }
    }

    // ---- max x_j ----------------------------------------------------------------
    if (!lp_infeasible && solves < max_iters) {
      lp.col_cost.assign(static_cast<std::size_t>(n), 0.0);
      lp.col_cost[u] = -1.0;  // minimise -x_j == maximise x_j
      lp.sense = ObjSense::kMinimize;

      const Solution sol =
          solve_primal_simplex(lp, lp_opts, logger, build_node_scaling(lp, lp_opts), nullptr,
                               warm.empty() ? nullptr : &warm);
      ++solves;

      if (sol.status == SolveStatus::kOptimal) {
        const double new_hi = safe_upper(sol.col_value[u], integer);
        if (new_hi < hi - tol::kObbtMinimumTightening && new_hi >= lp.col_lower[u]) {
          lp.col_upper[u] = new_hi;
          ++result.bounds_tightened;
        }
        warm = {sol.col_status, sol.row_status};
      } else if (sol.status == SolveStatus::kInfeasible) {
        // Same as above: stop all remaining probes.
        warm = {};
        lp_infeasible = true;
      } else {
        warm = {};
      }
    }
  }

  result.lp_solves = solves;

  // Write tightened bounds back to the caller's model.
  if (result.bounds_tightened > 0) {
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      model.col_lower[u] = lp.col_lower[u];
      model.col_upper[u] = lp.col_upper[u];
    }
  }

  logger.info(
      "OBBT (#515): {} LP solve(s), {} bound(s) tightened{}", result.lp_solves,
      result.bounds_tightened,
      (solves >= max_iters && n * 2 > max_iters) ? " (stopped at mip_obbt_max_iters)" : "");

  return result;
}

}  // namespace sankhya::mip
