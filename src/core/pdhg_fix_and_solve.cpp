// SPDX-License-Identifier: Apache-2.0
// SANKHYA - from a first-order point to a vertex by fixing the active bounds first (#888).
// See the header for the method and its citations.
#include "core/pdhg_fix_and_solve.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "core/resource_limits.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/crossover.hpp"
#include "util/profiler.hpp"

namespace sankhya {
namespace {

enum class Fix { kNone, kLower, kUpper };

/// Within `fraction` of a finite bound, relative to the value's own size; for a column the
/// reduced cost (in the minimisation sense) must not push away from that bound, as in
/// crossover_guess (simplex/crossover.cpp). A row is read from its activity alone: its dual
/// is the one quantity a first-order point gets least precisely, and a row snapped to the
/// wrong bound costs the primal loop a pivot, never the answer.
Fix classify(double value, double lo, double hi, double reduced_cost, double fraction) {
  if (lo == hi) return Fix::kNone;  // already fixed by the model
  const double scale = std::max(1.0, std::fabs(value));
  const bool at_lower = is_finite_bound(lo) && value - lo <= fraction * scale;
  const bool at_upper = is_finite_bound(hi) && hi - value <= fraction * scale;
  if (at_lower && reduced_cost >= -tol::kDualFeasibility) return Fix::kLower;
  if (at_upper && reduced_cost <= tol::kDualFeasibility) return Fix::kUpper;
  return Fix::kNone;
}

[[nodiscard]] bool has_a_point(const Model& model, const Solution& first) {
  return claims_a_point(first) &&
         static_cast<Index>(first.col_value.size()) == model.num_cols() &&
         static_cast<Index>(first.col_dual.size()) == model.num_cols() &&
         static_cast<Index>(first.row_activity.size()) == model.num_rows();
}

}  // namespace

FixAndSolveClassification classify_for_fix_and_solve(const Model& model, const Solution& first,
                                                     double fraction) {
  FixAndSolveClassification c;
  if (!has_a_point(model, first)) return c;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const double sense = model.sense_multiplier();
  c.col_fix.assign(static_cast<std::size_t>(n), BasisStatus::kUnknown);
  c.row_fix.assign(static_cast<std::size_t>(m), BasisStatus::kUnknown);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    switch (classify(first.col_value[u], model.col_lower[u], model.col_upper[u],
                     sense * first.col_dual[u], fraction)) {
      case Fix::kLower:
        c.col_fix[u] = BasisStatus::kAtLower;
        ++c.cols_fixed;
        break;
      case Fix::kUpper:
        c.col_fix[u] = BasisStatus::kAtUpper;
        ++c.cols_fixed;
        break;
      case Fix::kNone: break;
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    switch (classify(first.row_activity[u], model.row_lower[u], model.row_upper[u], 0.0,
                     fraction)) {
      case Fix::kLower:
        c.row_fix[u] = BasisStatus::kAtLower;
        ++c.rows_snapped;
        break;
      case Fix::kUpper:
        c.row_fix[u] = BasisStatus::kAtUpper;
        ++c.rows_snapped;
        break;
      case Fix::kNone: break;
    }
  }
  return c;
}

Solution fix_and_solve_then_crossover(const Model& model, Solution first,
                                      const Options& options, Logger& logger,
                                      SolveControl* control, const Timer& timer) {
  const ProfileScope whole(logger.profiler(), "fix-and-solve");
  if (!has_a_point(model, first)) return first;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const double fraction = options.get_double("pdhg_crossover_fix_tolerance");
  const FixAndSolveClassification c = classify_for_fix_and_solve(model, first, fraction);
  if (c.cols_fixed + c.rows_snapped == 0) {
    first.message += "; fix-and-solve: the point holds nothing at a bound, nothing to fix";
    return first;
  }

  // 1. The restricted model: every column the point holds at a bound is fixed there, every
  //    row whose activity sits on a bound becomes an equality at it.
  Model restricted = model;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (c.col_fix[u] == BasisStatus::kAtLower) restricted.col_upper[u] = model.col_lower[u];
    if (c.col_fix[u] == BasisStatus::kAtUpper) restricted.col_lower[u] = model.col_upper[u];
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (c.row_fix[u] == BasisStatus::kAtLower) restricted.row_upper[u] = model.row_lower[u];
    if (c.row_fix[u] == BasisStatus::kAtUpper) restricted.row_lower[u] = model.row_upper[u];
  }

  // 2. The dual simplex on it, with presolve (the fixed columns and the singleton rows they
  //    leave come out there), from its slack basis, on the time the limit has left. Not this
  //    route again, and not the first-order engine.
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
      "Fix-and-solve (#888): {} of {} columns fixed and {} of {} rows snapped at the bound the "
      "first-order point holds them at (within {:.0e}); the dual simplex on the rest",
      c.cols_fixed, n, c.rows_snapped, m, fraction);
  Timer restricted_clock;
  Solution vertex;
  {
    const ProfileScope timed(logger.profiler(), "fix-and-solve restricted LP");
    vertex = solve(restricted, reduced, control);
  }
  const bool basis_sized = static_cast<Index>(vertex.col_status.size()) == n &&
                           static_cast<Index>(vertex.row_status.size()) == m;
  if (vertex.status != SolveStatus::kOptimal || !basis_sized) {
    first.message += fmt::format(
        "; fix-and-solve: the restricted LP ({} columns fixed, {} rows snapped) ended {} in "
        "{:.2f}s{}, the first-order point stands",
        c.cols_fixed, c.rows_snapped, to_string(vertex.status),
        restricted_clock.elapsed_seconds(), basis_sized ? "" : " without a basis");
    logger.info("Fix-and-solve: restricted LP ended {} in {:.2f}s; back to the plain crossover",
                to_string(vertex.status), restricted_clock.elapsed_seconds());
    return first;
  }

  // 3. RELEASE. The restricted optimum is a primal-feasible vertex of the original model
  //    and its basis is a basis of the original model: a fixed column is nonbasic at the
  //    bound it was fixed to, a snapped row's logical nonbasic at the bound it was snapped
  //    to, everything else as the restricted solve left it. The crossover's push installs
  //    exactly that basis; there is nothing superbasic to push, and its primal loop pivots
  //    in whichever released entry carries a wrong-signed reduced cost.
  WarmStart warm;
  warm.col_status = vertex.col_status;
  warm.row_status = vertex.row_status;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (c.col_fix[u] != BasisStatus::kUnknown && warm.col_status[u] != BasisStatus::kBasic) {
      warm.col_status[u] = c.col_fix[u];
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (c.row_fix[u] != BasisStatus::kUnknown && warm.row_status[u] != BasisStatus::kBasic) {
      warm.row_status[u] = c.row_fix[u];
    }
  }
  const Count restricted_iterations = vertex.iterations;
  const double restricted_seconds = restricted_clock.elapsed_seconds();
  vertex.status = SolveStatus::kFeasible;
  vertex.recompute_quality(model);
  // The push adds its pivots to the start's count: PDHG's iterations and the restricted
  // LP's pivots together are what came before it.
  vertex.iterations = first.iterations + restricted_iterations;
  logger.info(
      "Fix-and-solve: restricted vertex in {:.2f}s and {} pivots; released, scaled "
      "infeasibility {:.2e} / {:.2e}; the crossover push from its basis",
      restricted_seconds, restricted_iterations, vertex.primal_infeasibility_scaled,
      vertex.dual_infeasibility_scaled);
  Solution crossed = crossover_push_from_basis(model, std::move(vertex), std::move(warm),
                                               options, logger, control, timer);
  if (crossed.status != SolveStatus::kOptimal) {
    first.message += fmt::format(
        "; fix-and-solve: the restricted LP ({} columns fixed, {} rows snapped, {:.2f}s) gave "
        "a vertex but the crossover from its basis did not finish ({}), the first-order point "
        "stands",
        c.cols_fixed, c.rows_snapped, restricted_seconds, crossed.message);
    return first;
  }
  crossed.algorithm = first.algorithm + "+fix-and-solve+crossover";
  crossed.kkt_1e4_seconds = first.kkt_1e4_seconds;
  crossed.kkt_1e6_seconds = first.kkt_1e6_seconds;
  crossed.kkt_1e8_seconds = first.kkt_1e8_seconds;
  crossed.kkt_1e4_iterations = first.kkt_1e4_iterations;
  crossed.kkt_1e6_iterations = first.kkt_1e6_iterations;
  crossed.kkt_1e8_iterations = first.kkt_1e8_iterations;
  crossed.message = fmt::format(
      "fix-and-solve (#888): {} of {} columns fixed and {} of {} rows snapped at the "
      "first-order point's bounds, restricted LP optimal in {:.2f}s and {} pivots; {}",
      c.cols_fixed, n, c.rows_snapped, m, restricted_seconds, restricted_iterations,
      crossed.message);
  return crossed;
}

}  // namespace sankhya
