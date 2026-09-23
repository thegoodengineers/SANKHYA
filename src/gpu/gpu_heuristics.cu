// SPDX-License-Identifier: Apache-2.0
// GPU MIP heuristics (#509): feasibility pump and fix-and-propagate.
//
// Feasibility pump algorithm (Fischetti, Glover & Lodi 2005; GPU variant: Mexi et al.
// arXiv:2307.03466, Corduk et al. arXiv:2510.20499):
//   Outer loop:
//     1. x^LP = PDHG(min ||x - x^INT||_1 over the integer columns s.t. Ax in [lo,hi],
//               x in [lb,ub]) - the L1 distance is what makes the projection an LP
//               (a squared norm would make it a QP); GPU PDHG solves it with the
//               pdhg_gpu.cu primitives, the distance linearised per column by its
//               position against x^INT (at a bound: one-sided; interior: split).
//     2. x^INT = round(x^LP) to nearest integer for each integer column.
//     3. If x^INT is feasible → return HeuristicSolution{x^INT, c'x^INT}.
//     4. If ||x^LP - x^INT||_1 is not improving → perturb and continue.
//   Stop after gpu_pump_max_iter rounds (default 50).
//
// Fix-and-propagate algorithm (Corduk et al. arXiv:2510.20499):
//   1. Solve LP relaxation (CPU, already done at the root node).
//   2. For each integer column j with |x^LP_j - round(x^LP_j)| < pump_round_tol:
//        fix lb_j = ub_j = round(x^LP_j).
//   3. Run GPU domain propagation (domain_prop.cu) to tighten remaining bounds.
//   4. Solve residual LP on CPU; if integer-feasible, return.
//
// Current state: stubs returning nullopt.
// Activate by setting gpu_pump=true or gpu_fix_and_prop=true (options added in
// src/util/options.cpp) once kernels are written and tested.

#include "gpu_heuristics.hpp"
#include "device.hpp"

namespace sankhya::gpu {

std::optional<HeuristicSolution>
feasibility_pump(const Model& model, const Options& options)
{
    if (!options.get_bool("gpu_pump") || !device_available(nullptr)) {
        return std::nullopt;
    }
    // TODO(#509): implement GPU feasibility pump kernel loop.
    (void)model;
    return std::nullopt;
}

std::optional<HeuristicSolution>
fix_and_propagate(const Model& model, const Options& options)
{
    if (!options.get_bool("gpu_fix_and_prop") || !device_available(nullptr)) {
        return std::nullopt;
    }
    // TODO(#509): implement GPU fix-and-propagate using domain_prop.cu.
    (void)model;
    return std::nullopt;
}

}  // namespace sankhya::gpu
