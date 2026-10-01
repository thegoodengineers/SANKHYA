// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the safe bound from the EXACT duals of the reported basis (#763).
//
// A column whose reduced cost must be exactly zero - basic, with a bound on one side only or
// none, in an optimal face that is unbounded in its direction - has no floating-point y that
// prices it: every double y leaves c_j - a_j'y a rounding residue of either sign, and the
// residue times a missing bound is -inf. The rational y with B'y = c_B exactly makes every
// basic reduced cost exactly zero, and the Neumaier-Shcherbina bound (safe_bound.hpp) of
// that y, evaluated in exact arithmetic, is finite whenever the basis is exactly dual
// feasible over the column box. See lp_exact_dual.cpp for the method and its citations.
#pragma once

#include <limits>
#include <span>
#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

struct ExactDualBound {
  /// True when `value` is a proved bound.
  bool proved = false;
  /// A lower bound on min (cost . x) over the rows and the box (minimise space, no offset):
  /// the exact bound, rounded down to a double.
  double value = -std::numeric_limits<double>::infinity();
  /// The multipliers it was proved from, minimise space, as exact "numerator/denominator".
  std::vector<std::string> multipliers;
  /// Why no bound was proved; empty when `proved`.
  std::string message;
  /// The budget ran out before the duals were exact: a larger basis or budget, not a
  /// different basis, is what would change that.
  bool timed_out = false;
};

/// The exact duals of the basis `basis` carries (col_status / row_status, one basic variable
/// per row) for the minimise-space `dual_cost`, and their Neumaier-Shcherbina bound for the
/// minimise-space `cost` over the box [lower, upper], in exact arithmetic. The two costs
/// differ when the basis is that of a cost-perturbed re-solve (lp_safe_bound.cpp, stage 5):
/// any y gives a valid bound, whatever cost it was computed for. Gives up, unproved, after
/// `seconds`.
[[nodiscard]] ExactDualBound exact_dual_bound(const Model& model,
                                              std::span<const double> dual_cost,
                                              std::span<const double> cost,
                                              std::span<const double> lower,
                                              std::span<const double> upper,
                                              const Solution& basis, double seconds);

}  // namespace sankhya
