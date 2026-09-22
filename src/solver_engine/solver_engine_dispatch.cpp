// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/solver_engine_dispatch.hpp"

#include "core/status_guard.hpp"
#include "solver_engine/solver_engine.hpp"
#include "solver_engine/solver_selector.hpp"

namespace sankhya::engine {

Solution solve(const SolverRegistry& registry, const Model& model, const Options& options,
               Logger& logger, SolveControl* control, bool warm_start) {
  Solution solution;
  solution.allocate_for(model);

  const std::string problem = model.validate();
  if (!problem.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = problem;
    return solution;
  }

  const EngineChoice choice = select(registry, model, options, logger, warm_start);
  if (choice.engine == nullptr) {
    solution.status = SolveStatus::kNotSolved;
    solution.algorithm = "none";
    solution.message = choice.reason;
    return solution;
  }

  Solution answer = choice.engine->solve(model, options, logger, control);
  // The two guards solve() applies to every engine's answer (src/core/status_guard.hpp): an
  // engine's own "optimal" is reconciled with the measured point, and a claimed point with a
  // non-finite objective is a numerical error, not an answer. The dual check is the LP one,
  // as in solve(): a branch-and-bound incumbent's reduced costs belong to its node.
  reconcile_status_with_measurement(&answer, options, logger,
                                    /*check_dual=*/classify(model) == ProblemClass::kLp);
  refuse_a_non_finite_answer(&answer, logger);
  return answer;
}

}  // namespace sankhya::engine
