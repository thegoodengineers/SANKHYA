// SPDX-License-Identifier: Apache-2.0
// GPU MIP heuristics scaffold (#509).
//   gpu_pump      — GPU feasibility pump (PDHG as LP projection step)
//   gpu_fix_and_prop — GPU fix-and-propagate
// References:
//   Fischetti, Glover & Lodi, *The feasibility pump*, Mathematical Programming 104 (2005)
//   Mexi et al., *Combining Dual Simplex and GPU PDLP...*, arXiv:2307.03466
//   Corduk et al., *GPU Heuristics for MILP*, arXiv:2510.20499
#pragma once

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include <optional>
#include <vector>

namespace sankhya::gpu {

struct HeuristicSolution {
  std::vector<double> x;  // primal values (size = model.num_cols())
  double obj;
};

/// GPU feasibility pump: PDHG solves the LP projection step on the device.
/// Returns nullopt when gpu_pump=false, no device, or no feasible point found.
[[nodiscard]] std::optional<HeuristicSolution> feasibility_pump(const Model& model,
                                                                const Options& options);

/// GPU fix-and-propagate: fixes fractional variables by rounding, propagates
/// bounds on the GPU, solves the residual LP.
/// Returns nullopt when gpu_fix_and_prop=false, no device, or failure.
[[nodiscard]] std::optional<HeuristicSolution> fix_and_propagate(const Model& model,
                                                                 const Options& options);

}  // namespace sankhya::gpu
