// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the status/measurement reconciliation guard.
//
// Declared in its own header so it can be tested DIRECTLY. The guard's job is to catch an
// engine that claims more than its own returned point supports; testing it only through an
// engine means the test can exist only while some engine is misbehaving, and it silently
// loses its subject the moment that engine is fixed. That is exactly what happened here -
// PDHG now polices itself, so the path through PDHG no longer reaches this code.
#pragma once

#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <fmt/format.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// Force the reported status to agree with the measured quality of the point. See the
/// definition in src/core/solve.cpp for the full reasoning.
///
/// `check_dual` is false for a MILP: a branch-and-bound incumbent comes from a node LP whose
/// bounds were tightened by branching, so its reduced costs are dual feasible for that node
/// and generally not for the original model.
void reconcile_status_with_measurement(Solution* solution, const Options& options,
                                       Logger& logger, bool check_dual);

/// Downgrade a claimed point whose objective is not finite to a numerical error, and leave a
/// limited search that found nothing alone (#289). Defined beside the guard above in
/// src/core/solve.cpp; declared here so the registry's own entry point applies both.
void refuse_a_non_finite_answer(Solution* solution, Logger& logger);

/// The answer for an engine that ran out of memory (#246): a numerical error whose message
/// names the engine, so a reader knows which budget to lower. Shared by both guards below.
inline Solution out_of_memory_answer(std::string_view engine, const Timer& timer) {
  Solution solution;
  solution.status = SolveStatus::kNumericalError;
  solution.algorithm = std::string(engine);
  solution.message = fmt::format(
      "the solve ran out of memory inside the {} after {:.1f}s; the model is larger than "
      "this machine can hold for that engine - lower its size budget, add memory, or use "
      "another engine",
      engine, timer.elapsed_seconds());
  solution.solve_seconds = timer.elapsed_seconds();
  return solution;
}

/// The engine's name for the guards' messages: a string, or a callable returning one, read
/// AT THE CATCH (#437) - solve() learns which engine it is running only after `auto` has
/// selected, and the answer must name the engine that actually ran out, not "solver".
template <typename Name>
std::string engine_name_at_the_catch(Name&& name) {
  if constexpr (std::is_invocable_v<Name>) {
    return std::string(name());
  } else {
    return std::string(std::string_view(name));
  }
}

/// Fill Solution::stopped_by from the status when the engine named no reason itself (#289):
/// kTimeLimit is the clock, kIterationLimit the iteration count, kNodeLimit the node count,
/// kInterrupted the caller. A reason the engine did set is kept - the branch and bound sets
/// it because a limit hit holding an incumbent reports kFeasible. Defined in
/// src/core/solve.cpp; declared here so SolverEngine::solve() and the registry's entry point
/// apply the same mapping solve() does (#297).
void record_why_it_stopped(Solution* solution);

/// Run an engine and turn an out-of-memory condition into a status (#246).
///
/// Every other failure in this project is a SolveStatus with a message and a stats blob; a
/// std::bad_alloc escaping solve() was the one that was not - the CLI died with the log
/// buffer, the C API's caller got an error code and nothing else, a benchmark runner saw
/// `no_output`. The engines allocate in the places where a model can be too large for the
/// machine (the ordering's quotient graph, the factor, a dense working set), and each has
/// its own budget so that this is the last resort, not the first line. When it fires the
/// unwinding has already released what the engine held, so filling in the Solution here is
/// safe, and the message names the engine so a reader knows which budget to lower.
///
/// Declared here, beside the other guard, so a test can throw through it directly rather
/// than only through an engine that happens to exhaust memory today.
template <typename Body, typename Name>
Solution run_engine_guarded(Body&& body, Name&& engine, const Timer& timer, Logger& logger) {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    Solution solution =
        out_of_memory_answer(engine_name_at_the_catch(std::forward<Name>(engine)), timer);
    logger.error("{}", solution.message);
    return solution;
  }
}

/// Run an engine that is allowed to DECLINE on memory (#437): the same answer as the guard
/// above, returned to the caller as a status it can act on rather than thrown past it.
///
/// The interior point is the engine `auto`'s rule table sends large models to, and the one
/// that factorizes, so it is the one that meets the machine's memory. solve() already runs
/// another engine when it declines with a numerical error or no answer; but an exhausted
/// factor did not return, it threw, and the outer guard turned the recovery the comment
/// promised into the failure it was written to prevent (chromaticindex1024-7: `auto` reports
/// numerical_error after 77 s where PDHG alone is optimal in 6). This guard sits INSIDE the
/// recovery: the engine's memory failure comes back as the declined status the fallback
/// tests, with the message kept, and the outer guard stays the last resort for whatever
/// runs last. The rule this decides: an engine that another engine can still stand in for
/// declines on memory; the engine that runs last ends the solve.
template <typename Body>
Solution run_declining_on_out_of_memory(Body&& body, std::string_view engine,
                                        const Timer& timer, Logger& logger) {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    Solution solution = out_of_memory_answer(engine, timer);
    logger.warning("{}; declined, so another engine can run", solution.message);
    return solution;
  }
}

/// Test seam (#437): while true, solve()'s interior-point attempts - the selected engine
/// and PDHG's polish - throw std::bad_alloc before the engine runs, so the recovery can be
/// tested through solve() itself without a model that exhausts the machine. Defined in
/// src/core/solve.cpp; read in one place per attempt; never set outside a test.
bool& interior_point_out_of_memory_for_testing();

/// Test seam (#576): while above zero, solve()'s selected interior point spends this many
/// seconds more after it returns and before the crossover, so the crossover's share of a
/// time limit the interior point has partly used can be tested without a model that takes
/// that long. Defined in src/core/solve.cpp; never set outside a test.
double& interior_point_extra_seconds_for_testing();

}  // namespace sankhya
