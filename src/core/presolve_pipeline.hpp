// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the presolve/postsolve pipeline (#301), shared by src/core/solve.cpp and
// src/solver_engine/solver_engine_dispatch.cpp's engine::solve() (#297 full integration) so
// a caller reaching an engine through either path gets presolve, postsolve and the
// pool_complete/write_presolved interactions identically - one implementation, not two
// competing ones.
#pragma once

#include <functional>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/timer.hpp"
#include "solver_engine/solver_engine.hpp"

namespace sankhya {

/// What running `run_engine` under presolve produced.
struct PresolveOutcome {
  Solution solution;
  /// True when presolve settled the model on its own (proved infeasible, or - not currently
  /// possible, but the caller does not have to know that - some future presolve pass proves
  /// optimality outright); `solution` is then complete and no further postsolve/verification
  /// step should run on it beyond what run_with_presolve already applied.
  bool proved = false;
};

/// Runs presolve around `run_engine`, exactly as solve() has always done (#301): presolve
/// settles what it can, and if it proves the model infeasible outright no engine runs at
/// all; otherwise it reduces the model, hands the reduction to `run_engine`, and recovers
/// the original-space point through postsolve. A MILP/MIQP with pool_complete set skips
/// presolve for the reason #301 and #225 disagree over (see the .cpp); presolve turned off
/// hands `run_engine` the model exactly as given, and honours write_presolved by not
/// writing anything.
///
/// `run_engine` receives the model to actually solve - reduced, or the original when
/// presolve did not run - and returns that model's Solution; it does not need to know
/// presolve happened at all. This is shared infrastructure and stays centralized here
/// rather than duplicated per engine or per caller (#297's own Motivation section).
[[nodiscard]] PresolveOutcome run_with_presolve(
    const Model& model, const Options& options, Logger& logger, const Timer& timer,
    engine::ProblemClass problem_class,
    const std::function<Solution(const Model&)>& run_engine);

}  // namespace sankhya
