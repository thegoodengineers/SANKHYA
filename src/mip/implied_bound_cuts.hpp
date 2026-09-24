// SPDX-License-Identifier: Apache-2.0
// SANKHYA - implied-bound cuts from two-variable rows with a binary (#499).
//
// THE STRUCTURE. A row with exactly two nonzeros, a_x x + a_y y <= b, y binary and x
// continuous, bounds x differently at y = 0 and at y = 1. With a_x > 0 it gives
//
//     y = 0:  x <= u0 = min(U_x, b / a_x)          y = 1:  x <= u1 = min(U_x, (b - a_y) / a_x)
//
// where U_x is x's own upper bound; with a_x < 0 it gives lower bounds l0, l1 the same way.
// The implied-bound cut is the line through the two cases,
//
//     x <= u0 + (u1 - u0) y        (or x >= l0 + (l1 - l0) y),
//
// valid because it holds at y = 0 and at y = 1 and y takes no other value (Savelsbergh,
// "Preprocessing and probing techniques for mixed integer programming problems", ORSA J.
// Computing 6 (1994); Achterberg, Constraint Integer Programming, thesis, TU Berlin 2007,
// sec. 8.3). When x's own bound caps neither case the cut IS the row, which the LP point
// already satisfies, so it separates exactly when the column bound makes one case tighter
// than the row says: x <= 3 with x + 2 y <= 4 gives x + y <= 3, which the LP point (3, 0.5)
// violates although it satisfies the row.
//
// Both sides of a ranged row are used. Each case bound is moved outward by the primal
// feasibility tolerance before the cut is built, so the rounding of the division cannot make
// the cut cut off a point the exact case admits.
#pragma once

#include <vector>

#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// The implied-bound cuts the LP point `x` violates, one per row side at most, each by more
/// than the cut violation tolerance and each with a strictly fractional binary.
[[nodiscard]] std::vector<Cut> implied_bound_cuts(const Model& model,
                                                  const std::vector<double>& x);

}  // namespace sankhya::mip
