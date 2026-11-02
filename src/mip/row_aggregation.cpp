// SPDX-License-Identifier: Apache-2.0
// SANKHYA - row aggregation for the cut separators. The method and the citations are on the
// declaration.

#include "row_aggregation.hpp"

#include <cmath>
#include <limits>
#include <unordered_map>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

std::vector<RowEntries> rows_of(const Model& model) {
  std::vector<RowEntries> rows(static_cast<std::size_t>(model.num_rows()));
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      RowEntries& row = rows[static_cast<std::size_t>(column.rows[k])];
      row.columns.push_back(j);
      row.values.push_back(column.values[k]);
    }
  }
  return rows;
}

Index column_to_eliminate(const Model& model, const Solution& solution,
                          const std::vector<double>& col_lower,
                          const std::vector<double>& col_upper, const BaseRow& base,
                          const std::vector<char>* skip) {
  Index chosen = -1;
  double deepest = tol::kRowAggregationInside;
  for (std::size_t k = 0; k < base.columns.size(); ++k) {
    if (std::fabs(base.values[k]) <= tol::kZeroDrop) continue;
    const Index j = base.columns[k];
    const auto uj = static_cast<std::size_t>(j);
    if (model.col_type[uj] == VarType::kInteger) continue;
    if (skip != nullptr && (*skip)[uj] != 0) continue;
    const double x = solution.col_value[uj];
    double slack = std::numeric_limits<double>::infinity();
    if (is_finite_bound(col_lower[uj])) slack = std::min(slack, x - col_lower[uj]);
    if (is_finite_bound(col_upper[uj])) slack = std::min(slack, col_upper[uj] - x);
    if (std::isinf(slack)) continue;  // a free continuous column: the base is unusable anyway
    if (slack > deepest) {
      deepest = slack;
      chosen = j;
    }
  }
  return chosen;
}

bool aggregate_row(const Model& model, const std::vector<RowEntries>& rows, Index i,
                   Index eliminate, BaseRow* base) {
  const RowEntries& row = rows[static_cast<std::size_t>(i)];
  double coef_in_row = 0.0;
  for (std::size_t k = 0; k < row.columns.size(); ++k) {
    if (row.columns[k] == eliminate) coef_in_row = row.values[k];
  }
  if (std::fabs(coef_in_row) <= tol::kZeroDrop) return false;
  double coef_in_base = 0.0;
  for (std::size_t k = 0; k < base->columns.size(); ++k) {
    if (base->columns[k] == eliminate) coef_in_base = base->values[k];
  }
  const double lambda = -coef_in_base / coef_in_row;
  const auto ui = static_cast<std::size_t>(i);
  const double bound = lambda > 0.0 ? model.row_upper[ui] : model.row_lower[ui];
  if (!is_finite_bound(bound)) return false;
  // Merge: base += lambda * row.
  std::unordered_map<Index, std::size_t> position;
  for (std::size_t k = 0; k < base->columns.size(); ++k) position[base->columns[k]] = k;
  for (std::size_t k = 0; k < row.columns.size(); ++k) {
    const Index j = row.columns[k];
    const double add = lambda * row.values[k];
    auto it = position.find(j);
    if (it == position.end()) {
      base->columns.push_back(j);
      base->values.push_back(add);
    } else {
      base->values[it->second] += add;
    }
  }
  base->rhs += lambda * bound;
  base->multipliers.emplace_back(i, lambda);
  // The eliminated column is exactly cancelled in exact arithmetic; make it so here.
  for (std::size_t k = 0; k < base->columns.size(); ++k) {
    if (base->columns[k] == eliminate) base->values[k] = 0.0;
  }
  return true;
}

}  // namespace sankhya::mip
