// SPDX-License-Identifier: Apache-2.0
// SANKHYA - postsolve of propagated bounds (#485).
//
// Bound propagation from row activity (presolve.cpp, kImpliedBound) writes bounds into the
// reduced model's box that the original model does not have. The dual passes of postsolve
// move the reduced cost a column carries on such a bound onto the row that implied it; what
// is left is the basis. A column the engine left nonbasic on a propagated bound is strictly
// inside its ORIGINAL box, so in a basis of the original model it is basic, and one entry of
// the implying row leaves the basis for it: the row's own logical when it is basic, or a
// basic column of the row sitting on one of its original bounds. Both are on their bounds
// there, because a column on a bound propagated from row i means row i is active and every
// other column of the row is at the end of its range the bound was computed from.
//
// Andersen & Andersen, *Presolving in linear programming*, Math. Programming 71 (1995), for
// the dual of a bound implied by a row; the one-basic-entry-per-row rule is #341's.
#pragma once

#include <cstddef>
#include <vector>

#include "presolve/presolve.hpp"

namespace sankhya::presolve {

/// Zeroes the reduced cost of each column whose propagated bound handed its price to the
/// row (it is inside its original box, and the transfer chose the price to cancel it), and
/// restores a basis of the original model around every column left nonbasic on a propagated
/// bound. Statuses are only touched when every one is known (a simplex answer).
/// `column_removed_at[j]` is the position of the record that removed column j, or
/// `result.records.size()` for a column still in the reduced model; `implied_transfer[p]`
/// the price record p moved onto its row.
void restore_implied_bound_basis(const Result& result, const Model& original,
                                 const std::vector<std::size_t>& column_removed_at,
                                 const std::vector<double>& implied_transfer,
                                 Solution* solution);

}  // namespace sankhya::presolve
