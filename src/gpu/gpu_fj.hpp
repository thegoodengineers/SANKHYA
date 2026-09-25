// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU Feasibility Jump for MILP (#508).
//
// Luteberget and Sartor, "Feasibility Jump: an LP-free Lagrangian MIP heuristic",
// Mathematical Programming Computation 15 (2023), on the device in the layout of Corduk,
// Sielski, Boucher & Aatish, "GPU-accelerated primal heuristics for mixed integer
// programming", arXiv:2510.20499. The CPU search in src/mip/feasibility_jump.cpp is the
// reference: each CUDA block runs that search, with its own point, activities and weights,
// and scores its candidate columns one warp per column. gpu_fj_device.cuh has the details.
//
// Compiled only with SANKHYA_ENABLE_CUDA. Off by default behind gpu_feasibility_jump; the
// branch and bound runs the CPU search when this does not run.
#pragma once

#include <string>
#include <vector>

#include "../mip/feasibility_jump.hpp"
#include "sankhya/model.hpp"

namespace sankhya::gpu {

struct FjDeviceResult {
  /// False when the device search never started (no device, a model too large for 32-bit
  /// indices, a failed allocation or upload): the caller then runs the CPU search.
  bool ran = false;
  /// Why it did not run, or the CUDA error that cut a started search short.
  std::string reason;
  /// Points in the order handed over, each strictly better than the one before and each
  /// verified by mip::feasibility_jump_point_is_feasible() against `model` on the host;
  /// work, moves and weight updates summed over the restarts.
  mip::FeasibilityJumpResult search;
  int restarts = 0;      ///< searches run in parallel, one per block
  Count launches = 0;    ///< kernel launches, each followed by one host poll
  Count rejected = 0;    ///< device points the host check refused; 0 unless something is wrong
  double seconds = 0.0;  ///< wall time of the whole call, context and transfers included
};

/// Run `restarts` independent Feasibility Jump searches on the device (0: two per
/// multiprocessor, fewer if the card's free memory cannot hold them), every one from `start`
/// with its own seed, each until it has spent settings.work_limit nonzero visits, or until
/// settings.should_stop() says so (polled between launches). Each point is verified against
/// `model` on the host before it is kept or passed to settings.on_point.
[[nodiscard]] FjDeviceResult feasibility_jump(const Model& model,
                                              const std::vector<double>& start,
                                              const mip::FeasibilityJumpSettings& settings,
                                              int restarts = 0);

}  // namespace sankhya::gpu
