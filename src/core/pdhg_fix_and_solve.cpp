// SPDX-License-Identifier: Apache-2.0
// SANKHYA - from a first-order point to a vertex by fixing the active bounds first (#888).
// See the header for the method and its citations.
#include "core/pdhg_fix_and_solve.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "core/resource_limits.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/crossover.hpp"

namespace sankhya {
namespace {

// The same at-bound rule as crossover_guess (simplex/crossover.cpp): within this fraction
// of the value's own size, and the reduced cost not pushing away from the bound.
constexpr double kAtBoundFraction = 1e-6;

enum class Fix { kNone, kLower, kUpper };

Fix classify(const Model& model, const Solution& first, Index j) {
  const auto u = static_cast<std::size_t>(j);
  const double lo = model.col_lower[u];
  const double hi = model.col_upper[u];
  if (lo == hi) return Fix::kNone;  // already fixed by the model
  const double x = first.col_value[u];
  const double d = model.sense_multiplier() * first.col_dual[u];
  const double scale = std::max(1.0, std::fabs(x));
  if (is_finite_bound(lo) && x - lo <= kAtBoundFraction * scale && d >= -tol::kDualFeasibility) {
    return Fix::kLower;
  }
  if (is_finite_bound(hi) && hi - x <= kAtBoundFraction * scale && d <= tol::kDualFeasibility) {
    return Fix::kUpper;
  }
  return Fix::kNone;
}

}  // namespace

Index count_fixable_columns(const Model& model, const Solution& first) {
  const Index n = model.num_cols();
  if (static_cast<Index>(first.col_value.size()) != n ||
      static_cast<Index>(first.col_dual.size()) != n) {
    return 0;
  }
  Index fixed = 0;
  for (Index j = 0; j < n; ++j) {
    if (classify(model, first, j) != Fix::kNone) ++fixed;
  }
  return fixed;
}

Solution fix_and_solve_then_crossover(const Model& model, Solution first, const Options& options,
                                      Logger& logger, SolveControl* control,
                                      const Timer& timer) {
  const Index n = model.num_cols();
  if (!claims_a_point(first) || static_cast<Index>(first.col_value.size()) != n ||
      static_cast<Index>(first.col_dual.size()) != n) {
    return first;
  }
  // 1. The restricted model: every column the point holds at a bound is fixed there.
  Model restricted = model;
  Index fixed = 0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    switch (classify(model, first, j)) {
      case Fix::kLower: restricted.col_upper[u] = model.col_lower[u]; ++fixed; break;
      case Fix::kUpper: restricted.col_lower[u] = model.col_upper[u]; ++fixed; break;
      case Fix::kNone: break;
    }
  }
  if (fixed == 0) {
    first.message += "; fix-and-solve: the point holds no column at a bound, nothing to fix";
    return first;
  }
  // 2. The dual simplex on it, with presolve (the fixed columns leave the model there), on
  //    the time the limit has left. Not this route again, and not the first-order engine.
  const ResourceLimits limits(options, logger);
  Options reduced = options;
  reduced.set_string("algorithm", "dual-simplex");
  reduced.set_bool("pdhg_crossover", false);
  reduced.set_bool("pdhg_crossover_fix_and_solve", false);
  reduced.set_bool("crossover", false);
  if (limits.has_time_limit()) {
    const double remaining = limits.remaining_seconds(timer.elapsed_seconds());
    if (remaining <= 0.0) {
      first.message += "; fix-and-solve: no time left";
      return first;
    }
    reduced.set_double("time_limit", remaining);
  }
  logger.info(
      "Fix-and-solve (#888): {} of {} columns fixed at the bound the first-order point holds "
      "them at; the dual simplex on the rest",
      fixed, n);
  Timer restricted_clock;
  Solution vertex = solve(restricted, reduced, control);
  if (vertex.status != SolveStatus::kOptimal) {
    first.message += fmt::format(
        "; fix-and-solve: the restricted LP ({} columns fixed) ended {} in {:.2f}s, the "
        "first-order point stands",
        fixed, to_string(vertex.status), restricted_clock.elapsed_seconds());
    logger.info("Fix-and-solve: restricted LP ended {}; back to the plain crossover",
                to_string(vertex.status));
    return first;
  }
  // 3. The restricted optimum is a primal-feasible vertex of the original model (a fixed
  //    column sits inside its original bounds). Its reduced costs are c - A^T y for every
  //    column, the fixed ones included, as solve() reports them, so the gate in
  //    crossover_to_vertex measures exactly what the released columns violate.
  vertex.status = SolveStatus::kFeasible;
  vertex.recompute_quality(model);
  const Count restricted_iterations = vertex.iterations;
  logger.info(
      "Fix-and-solve: restricted vertex in {:.2f}s and {} pivots; released, scaled "
      "infeasibility {:.2e} / {:.2e}; the crossover from it",
      restricted_clock.elapsed_seconds(), restricted_iterations,
      vertex.primal_infeasibility_scaled, vertex.dual_infeasibility_scaled);
  Options wanted = options;
  wanted.set_bool("crossover", true);
  wanted.set_bool("crossover_from_nonoptimal", true);
  Solution crossed = crossover_to_vertex(model, std::move(vertex), wanted, logger, control, timer);
  if (crossed.status != SolveStatus::kOptimal) {
    first.message += fmt::format(
        "; fix-and-solve: the restricted LP ({} columns fixed, {:.2f}s) gave a vertex but "
        "the crossover from it did not finish ({}), the first-order point stands",
        fixed, restricted_clock.elapsed_seconds(), crossed.message);
    return first;
  }
  crossed.iterations += first.iterations;
  crossed.algorithm = first.algorithm + "+fix-and-solve+crossover";
  crossed.kkt_1e4_seconds = first.kkt_1e4_seconds;
  crossed.kkt_1e6_seconds = first.kkt_1e6_seconds;
  crossed.kkt_1e8_seconds = first.kkt_1e8_seconds;
  crossed.kkt_1e4_iterations = first.kkt_1e4_iterations;
  crossed.kkt_1e6_iterations = first.kkt_1e6_iterations;
  crossed.kkt_1e8_iterations = first.kkt_1e8_iterations;
  crossed.message = fmt::format(
      "fix-and-solve (#888): {} of {} columns fixed at the first-order point's bounds, "
      "restricted LP optimal in {:.2f}s and {} pivots; {}",
      fixed, n, restricted_clock.elapsed_seconds(), restricted_iterations, crossed.message);
  return crossed;
}

}  // namespace sankhya
