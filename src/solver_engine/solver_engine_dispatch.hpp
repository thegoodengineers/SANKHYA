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
///   - otherwise the chosen engine's Solution is returned after the two status guards
///     solve() applies to every engine's answer (src/core/status_guard.hpp): the reported
///     status is reconciled with the measured point, and a claimed point with a non-finite
///     objective is a numerical error, not an answer.
///
/// What this does NOT do, deliberately: presolve, postsolve, resource-limit remaining-time
/// budgeting, ranging, or IIS. Per #297's own Motivation section those stay "shared
/// infrastructure" and remain centralized in solve.cpp; duplicating them here would be
/// exactly the kind of parallel infrastructure #297 warns against, and solve() itself remains
/// the entry point for a caller who wants them. This function's contract is deliberately
/// narrower and more literal: the engine's own answer, guarded but not presolved.
[[nodiscard]] Solution solve(const SolverRegistry& registry, const Model& model,
                             const Options& options, Logger& logger,
                             SolveControl* control = nullptr, bool warm_start = false);

}  // namespace sankhya::engine
