// SPDX-License-Identifier: Apache-2.0
// SANKHYA - feasibility polishing of an answer another engine returned (#1177). See the
// header; the polish itself is pdhg_feasibility_polish.cpp, written from [AHLL25].
#include "pdhg_host_polish.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/resource_limits.hpp"
#include "../la/scaling.hpp"
#include "pdhg_evaluate.hpp"
#include "pdhg_feasibility_polish.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::pdhg {
namespace {
// The CPU engine's preconditioning (pdhg.cpp): 10 Ruiz passes, 30 power iterations.
constexpr int kRuizIterations = 10;
constexpr int kPowerIterations = 30;
}  // namespace

void polish_engine_answer(Solution* first, const Model& model, const Options& options,
                          Logger& logger, SolveControl* control, const Timer& timer) {
  if (!options.get_bool("pdhg_feasibility_polish")) return;
  if (first->status == SolveStatus::kOptimal || first->status == SolveStatus::kInterrupted ||
      !claims_a_point(*first)) {
    return;
  }
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  if (static_cast<Index>(first->col_value.size()) != cols ||
      static_cast<Index>(first->row_dual.size()) != rows) {
    return;
  }
  const double sense = model.sense_multiplier();

  // The unscaled problem in minimise space, exactly as solve_pdhg builds it.
  Problem problem;
  problem.model = &model;
  problem.cost.resize(static_cast<std::size_t>(cols));
  for (Index j = 0; j < cols; ++j) {
    problem.cost[static_cast<std::size_t>(j)] =
        sense * model.col_cost[static_cast<std::size_t>(j)];
  }
  problem.cost_norm = euclidean_norm(problem.cost);
  double bound_square = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double b = is_finite_bound(model.row_lower[u])
                         ? model.row_lower[u]
                         : (is_finite_bound(model.row_upper[u]) ? model.row_upper[u] : 0.0);
    bound_square += b * b;
  }
  problem.bound_norm = std::sqrt(bound_square);
  const Scaling scaling = build_scaling(model, problem.cost, kRuizIterations);
  const double spectral_norm =
      estimate_spectral_norm(scaling.matrix, kPowerIterations,
                             static_cast<unsigned>(options.get_int("random_seed")) + 1u);

  // PDHG's y from the reported multiplier: the engines report row_dual = sense * (-y).
  const std::vector<double>& x = first->col_value;
  std::vector<double> y(static_cast<std::size_t>(rows));
  for (Index i = 0; i < rows; ++i) {
    y[static_cast<std::size_t>(i)] = -sense * first->row_dual[static_cast<std::size_t>(i)];
  }
  std::vector<double> activity(static_cast<std::size_t>(rows));
  std::vector<double> reduced(static_cast<std::size_t>(cols));
  const Residuals before = evaluate(problem, x, y, activity, reduced);
  if (before.primal <= tol::kPdhgTight && before.absolute_primal <= tol::kPrimalFeasibility) {
    return;  // already primal feasible to the target: nothing a polish could add
  }

  const ResourceLimits limits(options, logger);
  const auto should_stop = [&]() {
    return (control != nullptr && control->interruption_requested()) ||
           limits.time_exhausted(timer.elapsed_seconds());
  };
  if (should_stop()) {
    first->message += "; feasibility polish on the host (#1177): no time left";
    return;
  }
  const double eta = spectral_norm > 0.0 ? 1.0 / spectral_norm : 1.0;
  const Count budget =
      tol::kPdhgFinalPolishBudgetMultiplier * std::max<Count>(1, first->iterations);
  logger.info(
      "Feasibility polish on the host (#1177) of the {} answer: {} steps per phase at most",
      first->algorithm, budget);
  FeasibilityPolisher polisher(problem, scaling, spectral_norm);
  FeasibilityPolisher::Outcome polished = polisher.polish(x, y, budget, 1.0, eta, should_stop);

  const double tolerance = options.get_double("pdhg_tolerance");
  const bool met_before = first->status == SolveStatus::kFeasible;
  const bool passes = polished.residuals.meets_request(tolerance);
  if (!polished.primal_reached || (met_before && !passes)) {
    first->message += fmt::format(
        "; feasibility polish on the host (#1177): {} steps, primal {}, not adopted "
        "(relative primal {:.3e}, dual {:.3e}, gap {:.3e})",
        polished.iterations, polished.primal_reached ? "reached" : "not reached",
        polished.residuals.primal, polished.residuals.dual, polished.residuals.gap);
    first->iterations += polished.iterations;
    return;
  }

  const Residuals& r = polished.residuals;
  for (Index j = 0; j < cols; ++j) {
    first->col_value[static_cast<std::size_t>(j)] = polished.x[static_cast<std::size_t>(j)];
  }
  const Residuals final_residuals =
      evaluate(problem, polished.x, polished.y, activity, reduced);
  for (Index j = 0; j < cols; ++j) {
    first->col_dual[static_cast<std::size_t>(j)] = sense * reduced[static_cast<std::size_t>(j)];
  }
  for (Index i = 0; i < rows; ++i) {
    first->row_dual[static_cast<std::size_t>(i)] =
        sense * (-polished.y[static_cast<std::size_t>(i)]);
  }
  const SolveStatus was = first->status;
  first->status = final_residuals.meets_project_standard() ? SolveStatus::kOptimal
                  : passes                                 ? SolveStatus::kFeasible
                                                           : was;
  // The CPU engine's rule for stating the gap of a polished pair (pdhg.cpp, #483): a dual
  // bound only when the reported y is dual feasible to kPdhgTight and the absolute standard.
  if (first->status == SolveStatus::kOptimal ||
      (final_residuals.dual <= tol::kPdhgTight &&
       final_residuals.absolute_dual <= tol::kDualFeasibility)) {
    first->dual_bound = sense * final_residuals.dual_objective + model.objective_offset;
  }
  first->iterations += polished.iterations;
  first->algorithm += "+feasibility-polish";
  first->message += fmt::format(
      "; feasibility polish on the host (#1177): {} steps, reported point polished, {} to {}, "
      "relative primal {:.3e}, dual {:.3e}, gap {:.3e}",
      polished.iterations, to_string(was), to_string(first->status), r.primal, r.dual, r.gap);
  first->recompute_quality(model);
}

}  // namespace sankhya::pdhg
