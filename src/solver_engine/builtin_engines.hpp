// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "solver_engine/solver_registry.hpp"

namespace sankhya::engine {

/// Registers every built-in engine solve()'s dispatcher already knows how to run: the
/// revised primal simplex, the dual simplex, restarted PDHG (CPU, and CUDA when this build
/// has SANKHYA_ENABLE_CUDA), the interior point, the convex QP engine and branch and bound.
/// Each wraps the EXISTING solve_*() free function (#297); none reimplements an algorithm.
void register_builtin_engines(SolverRegistry& registry);

}  // namespace sankhya::engine
