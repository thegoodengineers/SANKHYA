// SPDX-License-Identifier: Apache-2.0
// SANKHYA - optimality-based bound tightening (OBBT) at the root (#515).
//
// Reference:
//   Gleixner, Berthold, Muller and Weltge, "Three enhancements for
//   optimization-based bound tightening", J. Global Optimization 67, 2017.
//
// ALGORITHM.  For each variable x_j that is not already fixed (range >=
// 1e-6), add a temporary objective row c'x <= incumbent - epsilon when an
// incumbent is known, then solve:
//
//   (min LP)  min x_j  s.t. Ax <= b [, c'x <= z_cutoff]
//   (max LP)  max x_j  s.t. Ax <= b [, c'x <= z_cutoff]
//
// The optimal values are valid lower and upper bounds for x_j over the whole
// feasible region intersected with the cutoff.  If either LP is infeasible the
// model is infeasible (or the problem is solved by the cutoff); we stop and
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

constexpr double kFixedTolerance = 1e-6;  // skip variables whose range is smaller
constexpr double kCutoffEpsilon = 1e-6;   // c'x <= incumbent - epsilon
constexpr double kBoundTol = 1e-8;        // a tightening counts when >= this

/// Build a copy of `model` with a single extra row appended:
///   c'x <= cutoff
/// and return it.  The row indices of the original model are preserved.
Model add_cutoff_row(const Model& model, double cutoff) {
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
  m.row_lower[static_cast<std::size_t>(old_rows)] = -kInfinity;
  m.row_upper[static_cast<std::size_t>(old_rows)] = cutoff;
  return m;
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
  const double cutoff = have_cutoff ? incumbent - kCutoffEpsilon : 0.0;
  Model lp = have_cutoff ? add_cutoff_row(model, cutoff) : model;

  // Silent options for node-like solves.
  Options lp_opts = options;
  lp_opts.set_bool("log_to_console", false);

  // Precompute scaling once (same matrix for every probe).
  const NodeScaling scaling = build_node_scaling(lp, lp_opts);

  WarmStart warm;  // reused across solves
  int solves = 0;

  for (Index j = 0; j < n && solves < max_iters; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = lp.col_lower[u];
    const double hi = lp.col_upper[u];
    if (hi - lo < kFixedTolerance) continue;  // already fixed or near-fixed

    // ---- min x_j ----------------------------------------------------------------
    if (solves < max_iters) {
      lp.col_cost.assign(static_cast<std::size_t>(n), 0.0);
      lp.col_cost[u] = 1.0;
      lp.sense = ObjSense::kMinimize;

      const Solution sol = solve_primal_simplex(lp, lp_opts, logger, scaling, nullptr,
                                                warm.empty() ? nullptr : &warm);
      ++solves;

      if (sol.status == SolveStatus::kOptimal) {
        const double new_lo = sol.col_value[u];
        if (new_lo > lo + kBoundTol) {
          lp.col_lower[u] = new_lo;
          ++result.bounds_tightened;
        }
        warm = {sol.col_status, sol.row_status};
      } else {
        // Infeasible or unbounded with the cutoff row means we can stop.
        warm = {};
      }
    }

    // ---- max x_j ----------------------------------------------------------------
    if (solves < max_iters) {
      lp.col_cost.assign(static_cast<std::size_t>(n), 0.0);
      lp.col_cost[u] = -1.0;  // minimise -x_j == maximise x_j
      lp.sense = ObjSense::kMinimize;

      const Solution sol = solve_primal_simplex(lp, lp_opts, logger, scaling, nullptr,
                                                warm.empty() ? nullptr : &warm);
      ++solves;

      if (sol.status == SolveStatus::kOptimal) {
        const double new_hi = sol.col_value[u];
        if (new_hi < hi - kBoundTol) {
          lp.col_upper[u] = new_hi;
          ++result.bounds_tightened;
        }
        warm = {sol.col_status, sol.row_status};
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
