// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a returned solution checked against the model it was asked about. TESTS ONLY.

#include "oracles/solution_checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::oracle {

std::string point_violation(const Model& model, const Solution& s) {
  const auto cols = static_cast<std::size_t>(model.num_cols());
  const auto rows = static_cast<std::size_t>(model.num_rows());
  if (s.col_value.size() != cols) return "the solution carries no point";
  std::vector<double> activity(rows, 0.0);
  std::vector<double> row_scale(rows, 1.0);
  double objective = model.objective_offset;
  for (std::size_t j = 0; j < cols; ++j) {
    const double x = s.col_value[j];
    const std::string name = "column " + std::to_string(j) + " = " + std::to_string(x);
    if (!std::isfinite(x)) return name + " is not a number";
    const double room = tol::kPrimalFeasibility * std::max(1.0, std::fabs(x));
    if (x < model.col_lower[j] - room || x > model.col_upper[j] + room) {
      return name + " is outside its bounds";
    }
    if (model.col_type[j] == VarType::kInteger &&
        std::fabs(x - std::round(x)) > tol::kIntegrality) {
      return name + " is not integral";
    }
    objective += model.col_cost[j] * x;
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      const auto i = static_cast<std::size_t>(column.rows[k]);
      const double term = column.values[k] * x;
      activity[i] += term;
      row_scale[i] = std::max(row_scale[i], std::fabs(term));
    }
  }
  for (std::size_t i = 0; i < rows; ++i) {
    const double room = tol::kPrimalFeasibility * row_scale[i];
    if (activity[i] < model.row_lower[i] - room || activity[i] > model.row_upper[i] + room) {
      return "row " + std::to_string(i) + " has activity " + std::to_string(activity[i]) +
             ", outside its bounds";
    }
  }
  if (std::fabs(objective - s.objective) > 1e-6 * std::max(1.0, std::fabs(objective))) {
    return "the point is worth " + std::to_string(objective) + ", not the reported objective";
  }
  return "";
}

std::string dual_violation(const Model& model, const Solution& s, double expected) {
  const auto cols = static_cast<std::size_t>(model.num_cols());
  const auto rows = static_cast<std::size_t>(model.num_rows());
  const double sigma = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  double cost_scale = 1.0;
  for (const double c : model.col_cost) cost_scale = std::max(cost_scale, std::fabs(c));
  const double negligible = tol::kDualFeasibility * cost_scale;

  double bound = 0.0;
  for (std::size_t i = 0; i < rows; ++i) {
    const double y = sigma * s.row_dual[i];
    const double side = y > 0.0 ? model.row_lower[i] : model.row_upper[i];
    if (std::isfinite(side)) {
      bound += y * side;
    } else if (!(std::fabs(y) <= negligible)) {
      return "row " + std::to_string(i) + " has multiplier " + std::to_string(s.row_dual[i]) +
             " toward a side it has no bound on";
    }
  }
  for (std::size_t j = 0; j < cols; ++j) {
    double d = sigma * model.col_cost[j];
    double term_scale = cost_scale;
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      const double term =
          column.values[k] * sigma * s.row_dual[static_cast<std::size_t>(column.rows[k])];
      d -= term;
      term_scale = std::max(term_scale, std::fabs(term));
    }
    if (!(std::fabs(sigma * s.col_dual[j] - d) <= 1e-6 * term_scale)) {
      return "column " + std::to_string(j) + " reports reduced cost " +
             std::to_string(s.col_dual[j]) + ", but c - A'y is " + std::to_string(sigma * d);
    }
    const double side = d > 0.0 ? model.col_lower[j] : model.col_upper[j];
    if (std::isfinite(side)) {
      bound += d * side;
    } else if (!(std::fabs(d) <= negligible)) {
      return "column " + std::to_string(j) + " has reduced cost " + std::to_string(sigma * d) +
             " toward a side it has no bound on";
    }
  }
  const double certified = sigma * bound + model.objective_offset;
  if (!(std::fabs(certified - expected) <= 1e-6 * std::max(1.0, std::fabs(expected)))) {
    return "the duals certify " + std::to_string(certified) + ", not the optimum";
  }
  return "";
}

}  // namespace sankhya::oracle
