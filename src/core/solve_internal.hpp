// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior-point/fallback choreography solve() runs under `algorithm=auto`
// (src/core/solve.cpp), exposed here so a test can force the interior point to fail without
// exhausting real memory (#437). Not a public header - nothing outside src/core/solve.cpp and
// its test includes this.
#pragma once

#include <functional>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya::detail {

using InteriorPointRunner =
    std::function<Solution(const Model&, const Options&, Logger&, SolveControl*)>;

/// The choreography solve() applies to a SELECTED interior point under `algorithm=auto`:
/// guard the attempt against std::bad_alloc, cross over to a vertex when asked, and fall back
/// to the dual simplex or PDHG - by the same size rule select_engine() uses - when it declines
/// (a factor beyond its budget, a numerical failure, or no answer at all).
///
/// THE BUG THIS GUARDS AGAINST (#437). Before this function existed, the interior point was
/// called directly in the middle of this same decline-and-fallback logic; when it exhausted
/// memory it threw std::bad_alloc instead of returning a declined Solution, which unwound
/// PAST the fallback entirely and was only caught by the dispatch-level guard
/// (run_engine_guarded around the whole solve), which reports a terminal numerical error
/// under the generic engine name "solver". The one failure mode the fallback was written for
/// first - "a factor beyond its budget" - was exactly the one it could not see. Guarding the
/// attempt HERE, inside the choreography, turns that exception into the same kind of declined
/// Solution a normal numerical failure already produces, so the existing fallback logic
/// handles it unchanged.
///
/// `run_interior_point` performs the interior-point attempt itself and is a parameter purely
/// so a test can make it throw std::bad_alloc directly, proving the guard is wired into this
/// choreography rather than only reachable at the dispatch level; production code always
/// passes a wrapper around ipm::solve_ipm. `model` is the ORIGINAL, pre-presolve model (the
/// size rule is judged on the shape select_engine() saw, matching solve()'s own convention);
/// `target` is what is actually solved (the model as given, or presolved).
[[nodiscard]] Solution run_interior_point_with_fallback(
    const Model& model, const Model& target, const Options& options,
    const Options& engine_options, bool requested_auto, Logger& logger, SolveControl* control,
    const Timer& timer, const InteriorPointRunner& run_interior_point);

}  // namespace sankhya::detail
