// SPDX-License-Identifier: Apache-2.0
// SANKHYA - complemented MIR with variable-bound substitution (c-MIR, #498).
//
// WHAT THE PLAIN MIR MISSES. mir_cuts.cpp substitutes every continuous column by its nearer
// SIMPLE bound. On a fixed-charge network, or a refinery unit that is either off or running
// between a minimum and a maximum rate, a continuous flow x is bounded by x <= u y with y a
// binary switch, and its simple bound [0, u] forgets y entirely: the cut never sees that the
// flow and the switch move together, which is the whole structure. Marchand & Wolsey,
// "Aggregation and mixed integer rounding to solve MIPs", Operations Research 49(3), 2001,
// sec. 3, substitute each continuous column by its closest simple OR variable bound:
//
//     x = u y + c - s  (variable upper bound x <= u y + c),   s >= 0 continuous,
//     x = l y + c + s  (variable lower bound x >= l y + c),   s >= 0 continuous,
//
// which moves u (or l) onto the integer y, where the rounding can use it. Then, on the
// resulting base inequality sum a_j y_j + sum h_k s_k <= b (y_j integer, bound-substituted
// to be non-negative; s_k >= 0):
//
//   - divisors: delta = |a_j| for each integer y_j strictly between its bounds at the LP
//     point, and 1; the best delta, then delta / 2, / 4 and / 8;
//   - complementation: each integer y_j with two finite bounds, nearest its midpoint first,
//     is complemented (y_j = u_j - x_j instead of x_j - l_j, or back) and kept if the cut
//     improves;
//   - the mixed-integer rounding inequality of mir_cuts.hpp on each trial, substituted back
//     to the model's columns; the most EFFICACIOUS (violation over the Euclidean norm, in the
//     model's columns) is kept.
//
// VALIDITY. Every step is an identity or a valid relaxation: a variable bound is a row of
// the model (two nonzeros, one continuous and one integer column), so s >= 0 holds at every
// feasible point; the integer substitutions use integral bounds (a column whose bound is not
// integral is not substituted and the base is abandoned); the MIR inequality is valid for
// any divisor delta > 0 and any complementation. Nemhauser & Wolsey, "Integer and
// Combinatorial Optimization", Wiley 1988, ch. II.1.7 for the inequality.
#pragma once

#include <optional>
#include <vector>

#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// x <= slope * y + constant (an upper variable bound) or x >= slope * y + constant (a
/// lower one), for a continuous x and an integer y, read off a model row.
struct VariableBound {
  Index indicator = -1;
  double slope = 0.0;
  double constant = 0.0;
};

/// Per column, every variable bound the model's two-nonzero rows give it.
struct VariableBounds {
  std::vector<std::vector<VariableBound>> upper;
  std::vector<std::vector<VariableBound>> lower;
  int rows = 0;  ///< rows that gave at least one bound
};

/// Read the variable bounds off `model`: every row with exactly two nonzeros, one on a
/// continuous column x and one on an integer column y, and a finite bound. a x + b y <= U
/// gives x <= U / a - (b / a) y when a > 0 and x >= that when a < 0; a lower row bound the
/// same with the directions swapped.
[[nodiscard]] VariableBounds find_variable_bounds(const Model& model);

/// The c-MIR separation on one base inequality sum values_k x_{columns_k} <= rhs at the LP
/// point: the most efficacious cut over the substitutions, divisors and complementations
/// above, violated by at least `min_violation` relative to max(1, |rhs|), or nothing.
[[nodiscard]] std::optional<Cut> separate_cmir(const Model& model, const Solution& solution,
                                               const std::vector<double>& col_lower,
                                               const std::vector<double>& col_upper,
                                               const std::vector<Index>& columns,
                                               const std::vector<double>& values, double rhs,
                                               const VariableBounds& bounds,
                                               double min_violation);

}  // namespace sankhya::mip
