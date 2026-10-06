// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU-accelerated restarted PDHG for LP: public declaration.
//
// Compiled only when SANKHYA_ENABLE_CUDA is ON.  src/core/solve.cpp is the sole caller;
// it guards the call site with the same ifdef.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::gpu {

/// GPU-accelerated restarted PDHG.
///
/// Falls back transparently to pdhg::solve_pdhg() when no CUDA device is found, so the caller
/// never needs to probe the device itself. `warm_start` (#913 part 2) is the same model-space
/// primal-dual pair pdhg::solve_pdhg takes: divided by this solve's own scaling and projected
/// into its own bounds before use, and ignored (a cold start runs) when its x is not sized for
/// `model`. Every internal fallback to the CPU engine below passes it on unchanged.
[[nodiscard]] Solution solve_pdhg_gpu(const Model& model, const Options& options,
                                      Logger& logger, SolveControl* control = nullptr,
                                      const pdhg::PdhgWarmStart* warm_start = nullptr);

}  // namespace sankhya::gpu
