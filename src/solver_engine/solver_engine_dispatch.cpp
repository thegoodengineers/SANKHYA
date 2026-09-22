// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/solver_engine_dispatch.hpp"

#include <functional>

#include "core/presolve_pipeline.hpp"
#include "core/resource_limits.hpp"
#include "core/status_guard.hpp"
#include "sankhya/timer.hpp"
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

  const Timer timer;
  // Presolve, resource limits and the out-of-memory guard are the same shared
  // infrastructure solve() runs every engine under (#297 full integration) - run_with_presolve
  // (src/core/presolve_pipeline.cpp) and run_engine_guarded
  // (src/core/status_guard.hpp) are the identical functions solve() itself calls, not a
  // second copy of either. What solve() does NOT share here, deliberately: the LP-specific
  // choreography (PDHG-to-IPM polish, IPM-to-dual-simplex fallback, node-scaling reuse) that
  // only exists because solve() may run a SECOND engine after the first declines - this
  // entry point runs exactly the one engine select() chose, honestly, which is its contract.
  const ResourceLimits limits(options, logger);
  const SolverEngine* chosen = choice.engine;
  const ProblemClass problem_class = classify(model);
  const std::function<Solution(const Model&)> run_chosen_engine = [&](const Model& target) {
    Options engine_options = options;
    if (limits.has_time_limit()) {
      engine_options.set_double("time_limit",
                                limits.remaining_seconds(timer.elapsed_seconds()));
    }
    // solve_verified(), not solve(): `chosen` was already verified against `model` - the
    // ORIGINAL, pre-presolve model - by select() above. Presolve can legitimately change
    // what classify() reports for `target` (fixing every integer column removes the columns
    // classify() would read, degenerating a MILP into what looks like an LP); re-gating on
    // that would refuse an engine solve()'s own dispatcher (src/core/solve.cpp) has never
    // refused for the same reason, since it calls the raw solve_*() function directly and
    // never re-classifies a presolved model (see solve_verified()'s doc comment).
    return chosen->solve_verified(target, engine_options, logger, control);
  };

  Solution answer = run_engine_guarded(
      [&] {
        PresolveOutcome outcome =
            run_with_presolve(model, options, logger, timer, problem_class, run_chosen_engine);
        return outcome.solution;
      },
      chosen->name(), timer, logger);

  // The two guards solve() applies to every engine's answer (src/core/status_guard.hpp): an
  // engine's own "optimal" is reconciled with the measured point, and a claimed point with a
  // non-finite objective is a numerical error, not an answer. The dual check is the LP one,
  // as in solve(): a branch-and-bound incumbent's reduced costs belong to its node.
  reconcile_status_with_measurement(&answer, options, logger,
                                    /*check_dual=*/problem_class == ProblemClass::kLp);
  refuse_a_non_finite_answer(&answer, logger);
  return answer;
}

}  // namespace sankhya::engine
