// SPDX-License-Identifier: Apache-2.0
// SANKHYA - row aggregation for the cut separators (#221, #498, #1158).
//
// A single model row rarely yields a strong cut when it carries continuous columns that sit
// strictly inside their bounds at the LP point: a stock s_t between two periods of a
// lot-sizing model, a flow through an intermediate node of a network. Marchand & Wolsey,
// "Aggregation and mixed integer rounding to solve MIPs", Operations Research 49(3), 2001,
// sec. 2, ELIMINATE such a column by adding a multiple of another row that contains it, and
// separate on every intermediate aggregate up to a small depth. On a lot-sizing model that
// walks the balance rows along the stocks, and the aggregate of periods l..k,
//
//     s_{l-1} + sum_{t=l..k} x_t = sum_{t=l..k} d_t + s_k,
//
// is the single-node flow set whose flow cover is the (l, S) inequality of Barany, Van Roy
// and Wolsey, "Strong formulations for multi-item capacitated lot sizing", Management Science
// 30 (1984), and whose MIR is the c-MIR path cut - the two families that close these models
// (Van Roy and Wolsey, "Solving mixed integer programming problems using automatic
// reformulation", Operations Research 35 (1987)). The MIR separator (mir_cuts.cpp) and the
// flow cover separator (flow_cover_cuts.cpp) share the walk below and differ in what they
// separate on each aggregate and in which column they are willing to eliminate.
//
// VALIDITY. An aggregate is a valid `<=` inequality because each row is added with the sign
// its own bound permits: a multiple lambda > 0 of `sum a x <= u`, or lambda < 0 of
// `sum a x >= l`. The multipliers are kept for a certificate's derivation (#518).

#pragma once

#include <utility>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::mip {

/// One row of the model in row-wise form: (column, coefficient) pairs.
struct RowEntries {
  std::vector<Index> columns;
  std::vector<double> values;
};

[[nodiscard]] std::vector<RowEntries> rows_of(const Model& model);

/// A base inequality sum values_j x_j <= rhs over the model's columns, sparse.
struct BaseRow {
  std::vector<Index> columns;
  std::vector<double> values;
  double rhs = 0.0;
  /// The rows it is the sum of, in the `<=` orientation (w > 0 on a row's upper side).
  std::vector<std::pair<Index, double>> multipliers;
};

/// The continuous column of `base` that sits furthest inside its bounds at the LP point (by
/// more than tol::kRowAggregationInside), or -1 when there is none. A column `skip` marks
/// (non-null, one char per model column) is never chosen: the flow cover separator keeps the
/// switched flows, whose variable upper bound is the structure it separates on.
[[nodiscard]] Index column_to_eliminate(const Model& model, const Solution& solution,
                                        const std::vector<double>& col_lower,
                                        const std::vector<double>& col_upper,
                                        const BaseRow& base,
                                        const std::vector<char>* skip = nullptr);

/// Add lambda times row i (as `sum a x <= u` for lambda > 0, `sum a x >= l` for lambda < 0)
/// to `base`, so that column `eliminate` cancels. Returns false when row i has no finite
/// bound on the side the sign needs, or does not contain the column.
bool aggregate_row(const Model& model, const std::vector<RowEntries>& rows, Index i,
                   Index eliminate, BaseRow* base);

}  // namespace sankhya::mip
