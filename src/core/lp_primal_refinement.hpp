// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one round of primal iterative refinement of an LP, for the certified gap's dual
// phase (#1082, src/core/lp_safe_bound.cpp).
//
// Gleixner, Steffy and Wolter, "Iterative refinement for linear programming", INFORMS J.
// Computing 28(3), 2016, sec. 3 (algorithm 1), with the dual residual left out: the costs
// are kept, so the refined LP has the same dual feasible bases and the same duals as the
// original. Only the primal side is shifted and scaled. A point x* (any point) is moved to
// the origin, and every bound is measured from it and multiplied by Delta = 2^k:
//
//   Delta (l - x*) <= xhat <= Delta (u - x*),
//   Delta (rl - A x*) <= A xhat <= Delta (ru - A x*)
//
// with A x* computed exactly and each difference rounded once. A basis has the same
// meaning in both LPs, and its basic solution in the refined one is Delta times its distance
// from x*, computed by the floating-point solve relative to that distance and not to |x*|.
// A violation of 1e-10 that the simplex cannot see at its 1e-7 tolerance in the original
// becomes one of 1e-10 Delta in the refined LP.
//
// WHY (#1082). On pilot87 the certified gap's dual phase, a warm-started dual simplex on the
// LP itself, took 129 iterations and then 0: both bases are exactly dual feasible, and both
// are still 5.4e-06 short of the optimum, because the remaining primal infeasibility is
// inside the 1e-7 tolerance. A 1e-12 tolerance on the same LP did not certify it either
// (#1138): double precision cannot place a basic value that finely relative to its size,
// while the refined LP computes only its distance from the point.
#pragma once

#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

struct PrimalRefinement {
  Model model;                 ///< the shifted and scaled LP; costs and matrix as given
  std::vector<double> shift;   ///< x*, one per column
  int scale_bits = 0;          ///< Delta = 2^scale_bits
  double max_violation = 0.0;  ///< the largest bound or row violation of x*, measured exactly
};

/// The refined LP of `model` about the point `x` (one value per column). Delta is the power
/// of two with Delta times the largest violation of x in [1, 2), kept between 1 and
/// 2^max_scale_bits; with no violation at all it is 2^max_scale_bits.
[[nodiscard]] PrimalRefinement refine_primal(const Model& model, const std::vector<double>& x,
                                             int max_scale_bits);

/// A solution of the refined LP taken back to `model`: x = x* + xhat / Delta and the row
/// activities A x. The basis and the duals are unchanged, since the costs were.
[[nodiscard]] Solution unrefine_primal(const Model& model, const PrimalRefinement& refinement,
                                       Solution refined);

}  // namespace sankhya
