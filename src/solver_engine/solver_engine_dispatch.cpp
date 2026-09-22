// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/solver_engine_dispatch.hpp"

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

  return choice.engine->solve(model, options, logger, control);
}

}  // namespace sankhya::engine
