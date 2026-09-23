// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the verifier-gated engine race behind `algorithm=auto` (#476).
//
// A rule table (#284) picks one LP engine in advance and loses whenever the rule is wrong:
// on the Mittelmann eight PDHG finishes chromaticindex1024-7 in 1.2 s where the interior
// point crashes, and the interior point reaches brazil3 in 1.4 s where PDHG takes 126.8 s.
// Running the candidates at once and keeping the first to finish is the concurrent optimizer
// the commercial LP codes document for exactly this reason (IBM ILOG CPLEX User's Manual,
// "Concurrent optimizer"; Gurobi Reference Manual, parameter Method = 3 "concurrent" and
// Method = 4 "deterministic concurrent"). Only the user-facing behaviour is taken from the
// manuals; nothing here is derived from either product.
//
// OUR ADDITION IS THE GATE. The first engine to FINISH is not the first engine to be RIGHT:
// an engine that stops early on a wrong answer would otherwise stop the engines that were
// about to get it right. So a result wins only after an in-process KKT check against the
// ORIGINAL model at the project tolerances passes (core/kkt_check.hpp: the checks
// tools/verify_solution.py makes, reimplemented so the tool itself stays independent), or,
// for infeasible and unbounded verdicts, after its certificate is checked against the
// original model (sankhya/certificate.hpp). A result that fails its check does not win and
// does not stop the others. The first accepted result wins and the others are cancelled
// through their own SolveControl.
//
// THE REST IS THE EXISTING MACHINERY. Each engine runs through run_with_presolve (its own
// presolve and postsolve, so what is checked is in the original model's rows and columns)
// under the time the limit has left, as a registered SolverEngine (the ipm engine crosses
// over, the pdhg-gpu one falls back to the CPU on its own), inside its own out-of-memory
// guard (#437) so one engine's exhausted allocation is that engine's declined status and not
// the process's end, and on its own thread with its own logger. The caller's interrupt and
// progress callback stay on the calling thread, as in the parallel tree search (#222).
//
// DETERMINISTIC MODE, or a machine with one core, runs the same engines in a FIXED
// SEQUENTIAL ORDER - dual simplex, interior point, PDHG - each to completion, stopping at
// the first accepted answer, so a run reproduces.
//
// Off by default (option engine_race) until an A/B on main.
#pragma once

#include <functional>
#include <string>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// Run the race on the LP `model` and return the answer, already postsolved. `rule_engine` is
/// the engine the rule table chose; when no engine's answer is accepted, that engine's answer
/// is returned (with an optimal claim that failed its check downgraded), so the race never
/// reports less than the rule would have. `engine_ran` receives the name of the engine whose
/// answer is returned, for the out-of-memory guard's message, and `accepted` whether that
/// answer passed its check (false: none did, and the rule table's engine's is reported). The
/// race log - every engine's status, time and verdict - is appended to the message and
/// written to `logger`.
[[nodiscard]] Solution run_engine_race(const Model& model, const Options& options,
                                       SolveControl* control, Logger& logger,
                                       const Timer& timer, const std::string& rule_engine,
                                       std::string* engine_ran, bool* accepted);

/// Whether this solve races: algorithm=auto, engine_race on, an LP, and no starting basis
/// (a basis is for a simplex, and only a simplex can use it).
[[nodiscard]] bool engine_race_applies(const Options& options, bool warm_start);

/// TEST SEAMS (#476), never set outside a test. `before_engine` runs on each engine's thread
/// before it starts, inside the engine's out-of-memory guard (a test delays the engines it
/// wants to finish last, or throws std::bad_alloc to stand for one that runs out);
/// `after_engine` may change an engine's answer before it is checked (a test injects a wrong
/// one). Both are read only while a race is running and must be set before it starts.
struct EngineRaceTestHooks {
  std::function<void(const std::string& engine)> before_engine;
  std::function<void(const std::string& engine, Solution* answer)> after_engine;
};
EngineRaceTestHooks& engine_race_hooks_for_testing();

}  // namespace sankhya
