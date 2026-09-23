// SPDX-License-Identifier: Apache-2.0
// GPU Feasibility Jump heuristic for MIP (#508).
// Reference: Corduk, Sielski, Boucher & Aatish, arXiv:2510.20499
#pragma once

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include <optional>
#include <vector>

namespace sankhya::gpu {

struct FJSolution {
  std::vector<double> x;
  double obj;
};

/// GPU Feasibility Jump: constraint activities and column weights kept on the
/// device; jump value and score of many candidate variables evaluated in
/// parallel (one warp per candidate, segmented reduction over its column);
/// best move applied with an atomic activity update; several independent
/// restarts run in parallel blocks.
///
/// Returns nullopt when gpu_feasibility_jump=false, no device, or no feasible
/// point is found within the iteration budget.
[[nodiscard]] std::optional<FJSolution> feasibility_jump(const Model& model,
                                                         const Options& options);

}  // namespace sankhya::gpu
