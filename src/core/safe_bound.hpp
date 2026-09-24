// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a rigorous LP lower bound from ANY dual vector (#519).
//
// Every pruning decision in branch and bound trusts a floating-point LP bound. The objective
// of the relaxation's primal point is only as good as the point's feasibility, and a dual
// vector that is slightly infeasible, or a first-order dual far from optimal, gives a
// "bound" that can sit ABOVE the true relaxation optimum. Pruning on it can discard the
// optimum, and nothing in the output says so.
//
// Neumaier and Shcherbina turn any y into a bound that is valid by construction. For
//
//     min c.x   subject to   row_lower <= A x <= row_upper,   l <= x <= u,
//
// every feasible x satisfies c.x = y.(A x) + r.x with r = c - A'y, and therefore
//
//     c.x >= sum_i min_{row_lower_i <= t <= row_upper_i} y_i t
//          + sum_j min_{l_j <= x_j <= u_j} r_j x_j.
//
// That is weak duality for the Lagrangian with the row multipliers fixed at y, and it holds
// for EVERY y: optimality of y only makes it tight. Computed with outward rounding - every
// quantity carried as an interval that contains its exact value - the right-hand side is a
// guaranteed lower bound on the exact relaxation optimum, so a node pruned on it contains
// no better point, whatever the LP solver's tolerances did.
//
// Reference: A. Neumaier and O. Shcherbina, "Safe bounds in linear and mixed-integer linear
// programming", Mathematical Programming 99 (2004) 283-296.
//
// Two details the formula leaves open, and how they are settled here:
//
//  * A multiplier that selects a side the row does not have (y_i > 0 on a row with no lower
//    side) makes the row term -inf. Such a y_i is replaced by 0 before anything is summed:
//    the formula holds for every y, so choosing a different y is always allowed, and this one
//    is at least as good. The count is reported.
//
//  * A column with an infinite bound on the side its reduced cost needs makes the column
//    term -inf, unless the reduced cost is exactly zero. A basic continuous column with no
//    upper bound has r_j = 0 only up to rounding, so in practice this would make almost every
//    bound -inf. Such a column is given a finite bound IMPLIED BY ONE ROW and the column
//    bounds of the others (Savelsbergh, "Preprocessing and probing for mixed integer
//    programming problems", ORSA J. Computing 6(4), 1994), itself computed with outward
//    rounding. Every feasible x satisfies it, so the minimum over the feasible set is
//    unchanged. This is our addition to the paper's formula, not part of it. A column no
//    single row bounds makes the bound -inf, and the count says why - unless scaling y by
//    (1 - e) for a tiny e fixes the sign of every such reduced cost, which it does when
//    each such column's cost has the needed sign (a basic column with a positive cost and
//    no upper bound, the usual case). Any y is allowed, so one retry with the scaled
//    vector is taken; also our addition.
//
// OUTWARD ROUNDING WITHOUT CHANGING THE ROUNDING MODE. Each operation is performed in the
// default round-to-nearest mode and its result is then stepped one ulp outward with
// std::nextafter. A correctly rounded result is within half an ulp of the exact value, so
// the stepped value is on the far side of it. This is looser than true directed rounding by
// one ulp per operation, and it needs neither std::fesetround nor -frounding-math, which the
// compiler would otherwise be free to reorder around. The function checks that the mode IS
// round-to-nearest and refuses (returns -inf) when it is not.
#pragma once

#include <span>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya {

/// The result of safe_dual_bound(). `value` is the bound; the counts explain it.
struct SafeBound {
  /// A guaranteed lower bound on min c.x over the relaxation (minimise space, no offset).
  /// -inf when no finite bound could be proved from this y.
  double value = -std::numeric_limits<double>::infinity();
  /// Multipliers set to zero because the row side they select is infinite.
  Index dropped_multipliers = 0;
  /// Columns that needed, and got, a bound implied by one row.
  Index implied_bounds = 0;
  /// Columns whose term is -inf: the reason `value` is -inf when it is.
  Index unbounded_columns = 0;
  /// When nonzero, the bound is that of (1 - shrink) y rather than y: see the retry in
  /// safe_bound.cpp. Zero when y itself gave the bound (or none was found).
  double shrink = 0.0;
};

/// The data safe_dual_bound() reads: an LP in minimise space. Spans, so the caller can
/// substitute a column box (a branch-and-bound node's) without copying the model.
struct SafeBoundProblem {
  const SparseMatrix* matrix = nullptr;
  /// Objective in minimise space: sense * c. Empty means all zero, which turns the bound
  /// into a Farkas test: a strictly positive result proves the rows and box infeasible.
  std::span<const double> cost;
  std::span<const double> row_lower;
  std::span<const double> row_upper;
  std::span<const double> col_lower;
  std::span<const double> col_upper;
};

/// Neumaier-Shcherbina bound for `problem` from the multipliers `y` (minimise space: y_i > 0
/// prices the row's lower side). `y` must have one entry per row. With `imply_bounds`,
/// columns whose reduced cost needs a missing bound get one from a row (see the file
/// header); without it they make the bound -inf.
[[nodiscard]] SafeBound safe_dual_bound(const SafeBoundProblem& problem,
                                        std::span<const double> y, bool imply_bounds = true);

/// The same for a whole Model with its own bounds, taking the duals exactly as a Solution
/// reports them (in the model's sense). The result is in minimise space without the offset,
/// which is what branch and bound compares.
[[nodiscard]] SafeBound safe_dual_bound(const Model& model, const std::vector<double>& row_dual,
                                        bool imply_bounds = true);

}  // namespace sankhya
