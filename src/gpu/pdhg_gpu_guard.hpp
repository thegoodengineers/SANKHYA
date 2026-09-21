// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the GPU PDHG safety gate (#281, #282, #383), shared by src/core/solve.cpp and
// the SolverEngine CUDA wrapper (src/solver_engine/builtin_engines.cpp) so the two cannot
// drift: a caller that reaches GPU PDHG through either path is refused the same way.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::gpu {

/// True when GPU PDHG may run on `model` under `options`. Applies, in order, the three
/// gates solve.cpp has always applied before committing to the GPU: deterministic mode is
/// refused (atomicAdd reductions are order-dependent, #383), the device's compute
/// capability must meet the minimum this build was compiled for (#282), and the estimated
/// device memory must fit in what is free (#281). Any failure logs why and returns false;
/// the caller then runs CPU PDHG instead. A query that itself fails (no device, driver
/// error) is not treated as a refusal here - the caller already knows whether a device is
/// present.
[[nodiscard]] bool gpu_pdhg_is_safe(const Model& model, const Options& options, Logger& logger);

}  // namespace sankhya::gpu
