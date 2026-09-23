// SPDX-License-Identifier: Apache-2.0
// GPU Feasibility Jump heuristic (#508).
//
// Algorithm (Corduk et al., arXiv:2510.20499):
//   Initialise: copy activities Ax and column weights to device.
//   Outer loop:
//     1. Score kernel: for each candidate integer column j, one warp computes
//        jump_value(j) = change in constraint violations from rounding x_j,
//        and score(j) = jump_value + weight * obj_delta.
//        One segmented CUB reduction per column over its nonzero rows.
//     2. Select: argmin score across candidates (CUB DeviceReduce).
//     3. Apply: atomic update of all activities for the chosen column's move.
//     4. Update weights on the device; no host sync inside a restart.
//   Several restarts run as independent CUDA blocks in parallel.
//   CPU FJ (#506) is the reference; both must pass offer_incumbent.
//
// Current state: stub returning nullopt.
// Activate with gpu_feasibility_jump=true once kernel is written and tested.

#include "device.hpp"
#include "gpu_fj.hpp"

namespace sankhya::gpu {

std::optional<FJSolution> feasibility_jump(const Model& /*model*/, const Options& options) {
  if (!options.get_bool("gpu_feasibility_jump") || !device_available(nullptr)) {
    return std::nullopt;
  }
  // TODO(#508): implement GPU Feasibility Jump kernels.
  return std::nullopt;
}

}  // namespace sankhya::gpu
