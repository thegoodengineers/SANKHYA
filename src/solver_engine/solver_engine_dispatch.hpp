// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the registry's own top-level entry point (#297): validate, select, run. This is
// what a caller who wants the pluggable architecture, rather than solve()'s own dispatch,
// should call - it gives the SAME model-validation contract solve() has, which calling a
// SolverEngine::solve() directly does not (see solver_engine.hpp: an engine's solve() trusts
// its caller to have validated the model, exactly as the free-function engines it wraps
// always have).
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "solver_engine/solver_registry.hpp"

namespace sankhya::engine {

/// Validate `model`, select an engine from `registry` for it under `options`, and run that
/// engine - or report why not, in the same shape solve() itself would:
///   - an invalid model comes back as kModelError with Model::validate()'s own message,
///     before any engine is reached (matching src/core/solve.cpp's first check exactly, by
///     calling the same Model::validate());
///   - a model no registered engine can take, or an `algorithm` the registry refuses (see
///     select()), comes back as kNotSolved naming why;
///   - otherwise the chosen engine runs under the SAME presolve/postsolve pipeline
///     (run_with_presolve, src/core/presolve_pipeline.cpp), the same resource-limit
///     remaining-time narrowing (ResourceLimits, src/core/resource_limits.hpp) and the same
///     out-of-memory guard (run_engine_guarded, src/core/status_guard.hpp) solve() itself
///     uses - one implementation of each, called from both places, not a second copy - and
///     the returned Solution passes through the same two status guards solve() applies to
///     every engine's answer: the reported status is reconciled with the measured point, and
///     a claimed point with a non-finite objective is a numerical error, not an answer.
///
/// What this does NOT do, deliberately: ranging, IIS, or the LP-specific choreography that
/// only exists because solve()'s LP branch may run a SECOND engine after the first declines
/// or to polish the first's answer (PDHG-to-IPM polish, IPM-to-dual-simplex fallback,
/// node-scaling reuse across warm-started node solves). That choreography is orchestration
/// ABOVE a single engine's own contract - it exists because solve() may choose to run more
/// than one engine for one request, which is not this function's job: select() already chose
/// the one engine to run, and running it honestly, under the same shared infrastructure
/// solve() itself provides, is the whole contract here.
[[nodiscard]] Solution solve(const SolverRegistry& registry, const Model& model,
                             const Options& options, Logger& logger,
                             SolveControl* control = nullptr, bool warm_start = false);

}  // namespace sankhya::engine
