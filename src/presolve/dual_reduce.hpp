// SPDX-License-Identifier: Apache-2.0
// SANKHYA - forcing rows and dominated columns by implied dual bounds: an LP presolve stage
// with a stack postsolve (presolve_dual_reductions), run on the model presolve.cpp hands over
// and only around a simplex solve (substitute.hpp says why the interior point and PDHG never
// see these stages).
//
// References, written from the literature:
//   Andersen & Andersen, "Presolving in linear programming", Math. Programming 71 (1995),
//     sec. 4 (forcing constraints: every column of the row fixed at the bound that attains
//     the activity bound, and the row's dual from the fixed columns' reduced costs) and sec. 7
//     (dominated columns: bounds on the duals from column singletons and free columns, and a
//     column whose reduced cost is positive for every dual feasible y fixed at its lower
//     bound).
//   Gondzio, "Presolve analysis of linear programs prior to applying an interior point
//     method", INFORMS J. Computing 9 (1997), sec. 2-3 - the same two reductions and the
//     reduced costs of the removed columns in postsolve.
//
// FORCING ROW. With every column of row r inside its box, the activity sum a_rj x_j lies in
// [inf_r, sup_r]. If sup_r equals the row's lower bound, the only feasible way to satisfy the
// row is every column at the bound that attains sup_r: x_j = u_j when a_rj > 0, l_j when
// a_rj < 0. Those columns are fixed, their contribution folded into the other rows' bounds
// and the objective, and the row goes. Symmetrically when inf_r equals the upper bound. In
// postsolve each fixed column j wants its reduced cost of the sign its bound needs,
//     d_j = c_j - sum_{i != r} a_ij y_i - a_rj y_r,
// and writing t_j = (c_j - sum_{i != r} a_ij y_i) / a_rj every column asks y_r >= t_j in the
// first case (y_r <= t_j in the second), so y_r is the largest t_j (smallest), or zero when
// the row's other side is infinite and zero is further that way; the column attaining it has
// d_j = 0 and is made basic, the row's logical nonbasic at the bound (basic when zero won).
// The rows still alive at the time hold every y_i the formula reads: a row removed earlier
// by this stage took all its columns with it, so none of them is in row r.
//
// DOMINATED COLUMN. Every optimal y is dual feasible: y_i >= 0 on a row bounded below only,
// <= 0 on one bounded above only, and for every column with u_j = +inf, sum a_ij y_i <= c_j
// (with l_j = -inf, >= c_j; both for a free column). Propagating the last through the column
// singletons and the free columns, as bound propagation on the dual, gives a box [y_lo, y_hi]
// containing every dual feasible y. A column j with l_j finite whose reduced cost is positive
// over the whole box, c_j - max sum a_ij y_i > 0, has d_j > 0 at every optimum, so x_j = l_j
// there by complementary slackness; symmetrically x_j = u_j when the reduced cost is negative
// over the box. Only singletons and free columns feed the box, and the free columns are
// never removed. Postsolve puts the column at its bound, nonbasic; the reduced costs of the
// stage's input are then c - A^T y.
//
// WHAT MAKES THE FIXING SAFE (#1337). Fixing x_j removes its dual constraint, so the claim
// "the reduced cost keeps its sign over the box" has to hold over a box that does not rest
// on that constraint. Then the constraint was implied by the others, the set of dual
// feasible y is the same with the column fixed, and an LP that had no optimum (unbounded or
// infeasible) still has none. Two things can break it, and both did:
//   * The box can come out empty, y_lo > y_hi on some row. Then no y is dual feasible and
//     nothing read off the intervals is a range. No column is fixed.
//   * A singleton with a bound missing is itself a source of the box, and the free columns
//     carry its bound round and back onto its own row. It is tested against its row's bound
//     from the row type and the other singletons alone, so it is fixed only when another
//     source bounds the row strictly tighter than it does. That also means a fixing never
//     removes the tightest source of a row, so the bounds the reduced model's duals are held
//     to survive the removals.
//
// LP only, and nothing is done unless a reduction fires.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::presolve {

struct DualReduction {
  enum class Kind { kForcingRow, kRedundantRow, kDominatedColumn };
  Kind kind = Kind::kForcingRow;
  Index row = -1;     ///< the row removed (forcing, redundant)
  Index column = -1;  ///< the column fixed (dominated)
  double value = 0;   ///< the dominated column's bound
  /// Forcing row: whether the activity was forced to the row's LOWER bound (every column at
  /// the bound attaining the maximum activity) or to its upper bound.
  bool at_lower = true;
  bool equality = false;  ///< the forcing row is an equality: y_r takes any sign
  /// Forcing row: its entries at the time (column, a_rj) and the bound each was fixed at.
  std::vector<Index> columns;
  std::vector<double> coefficients;
  std::vector<double> values;
};

struct DualReduceResult {
  Model model;
  std::vector<DualReduction> records;
  std::vector<Index> col_to_input;
  std::vector<Index> row_to_input;
  Index forcing_rows = 0;
  Index redundant_rows = 0;
  Index dominated_columns = 0;
  [[nodiscard]] bool reduced_anything() const { return !records.empty(); }
};

/// The reductions to a fixed point. Leaves `model` alone.
[[nodiscard]] DualReduceResult dual_reduce(const Model& model, const Options& options,
                                           Logger& logger);

/// The solution of `input` from the solution of result.model, records in reverse.
[[nodiscard]] Solution undo_dual_reduce(const DualReduceResult& result, const Model& input,
                                        const Solution& reduced);

/// How many solves ran on a reduced model, process-wide (tests).
[[nodiscard]] std::int64_t dual_reductions_applied_for_testing();

}  // namespace sankhya::presolve
