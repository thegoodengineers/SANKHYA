// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact equilibration of the model by powers of two (#792). See prescale.hpp.

#include "core/prescale.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sankhya {
namespace {

/// The nearest power of two, so that multiplying by it changes only the exponent.
double nearest_power_of_two(double value) {
  if (!(value > 0.0) || !std::isfinite(value)) return 1.0;
  return std::ldexp(1.0, static_cast<int>(std::lround(std::log2(value))));
}

/// Geometric mean passes before the Ruiz rounds; the iteration settles in a handful (Fourer
/// 1982 used up to 20 and saw little change after the first few).
constexpr int kGeometricPasses = 8;

double scaled_bound(double bound, double factor) {
  return is_finite_bound(bound) ? bound * factor : bound;
}

}  // namespace

PrescaledModel prescale_by_powers_of_two(const Model& model, int passes) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  const auto um = static_cast<std::size_t>(m);
  const auto un = static_cast<std::size_t>(n);
  std::vector<double> row(um, 1.0);
  std::vector<double> column(un, 1.0);
  std::vector<double> row_max(um);
  std::vector<double> column_max(un);
  // GEOMETRIC PASSES FIRST. Ruiz's infinity-norm iteration stops once every row and column
  // has its largest entry at one, and that fixed point depends on where it started: on
  // modszk1 it reaches 1.9e-4..0.74, on the same model with rows and columns scaled by up to
  // 2^20 (#762) it stops at 7e-10..1, and the simplex then pivots on 1e-9 and diverges. The
  // geometric mean scaling divides each row, then each column, by sqrt(min |a| * max |a|),
  // which works on log |a| the way Curtis and Reid's least squares does and so undoes a
  // diagonal rescaling instead of inheriting it (Fourer, "Solving staircase linear programs
  // by the simplex method, 1: Inversion", Math. Programming 23 (1982), sec. 4; Tomlin, "On
  // scaling linear programming problems", Math. Programming Study 4 (1975)). Ruiz then sets
  // the largest entry of each row and column to about one.
  std::vector<double> row_min(um);
  std::vector<double> column_min(un);
  const auto extremes = [&]() {
    std::fill(row_max.begin(), row_max.end(), 0.0);
    std::fill(column_max.begin(), column_max.end(), 0.0);
    std::fill(row_min.begin(), row_min.end(), HUGE_VAL);
    std::fill(column_min.begin(), column_min.end(), HUGE_VAL);
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const ColumnView view = model.matrix.column(j);
      for (Index k = 0; k < view.size; ++k) {
        const auto i = static_cast<std::size_t>(view.rows[k]);
        const double a = std::fabs(view.values[k] * row[i] * column[u]);
        if (a == 0.0) continue;
        row_max[i] = std::max(row_max[i], a);
        row_min[i] = std::min(row_min[i], a);
        column_max[u] = std::max(column_max[u], a);
        column_min[u] = std::min(column_min[u], a);
      }
    }
  };
  for (int pass = 0; pass < kGeometricPasses; ++pass) {
    extremes();
    for (std::size_t i = 0; i < um; ++i) {
      if (row_max[i] > 0.0) row[i] /= std::sqrt(row_min[i] * row_max[i]);
    }
    extremes();
    for (std::size_t j = 0; j < un; ++j) {
      if (column_max[j] > 0.0) column[j] /= std::sqrt(column_min[j] * column_max[j]);
    }
  }
  for (int pass = 0; pass < passes; ++pass) {
    std::fill(row_max.begin(), row_max.end(), 0.0);
    std::fill(column_max.begin(), column_max.end(), 0.0);
    for (Index j = 0; j < n; ++j) {
      const ColumnView view = model.matrix.column(j);
      for (Index k = 0; k < view.size; ++k) {
        const auto i = static_cast<std::size_t>(view.rows[k]);
        const double a =
            std::fabs(view.values[k] * row[i] * column[static_cast<std::size_t>(j)]);
        row_max[i] = std::max(row_max[i], a);
        column_max[static_cast<std::size_t>(j)] =
            std::max(column_max[static_cast<std::size_t>(j)], a);
      }
    }
    for (std::size_t i = 0; i < um; ++i) {
      if (row_max[i] > 0.0) row[i] /= std::sqrt(row_max[i]);
    }
    for (std::size_t j = 0; j < un; ++j) {
      if (column_max[j] > 0.0) column[j] /= std::sqrt(column_max[j]);
    }
  }
  for (double& r : row) r = nearest_power_of_two(r);
  for (double& s : column) s = nearest_power_of_two(s);

  PrescaledModel out;
  out.row = row;
  out.column = column;
  Model& scaled = out.model;
  scaled = model;
  scaled.matrix.reset(m, n);
  scaled.matrix.reserve(static_cast<std::size_t>(model.matrix.num_nonzeros()));
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const ColumnView view = model.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) {
      scaled.matrix.add_entry(
          view.rows[k], j,
          view.values[k] * row[static_cast<std::size_t>(view.rows[k])] * column[u]);
    }
    scaled.col_cost[u] = model.col_cost[u] * column[u];
    scaled.col_lower[u] = scaled_bound(model.col_lower[u], 1.0 / column[u]);
    scaled.col_upper[u] = scaled_bound(model.col_upper[u], 1.0 / column[u]);
  }
  // Exact products of data and powers of two: nothing here is rounding, nothing is dropped.
  scaled.matrix.finalize(0.0);
  for (std::size_t i = 0; i < um; ++i) {
    scaled.row_lower[i] = scaled_bound(model.row_lower[i], row[i]);
    scaled.row_upper[i] = scaled_bound(model.row_upper[i], row[i]);
  }
  return out;
}

void unscale_solution(const PrescaledModel& prescaled, Solution* solution) {
  const auto& row = prescaled.row;
  const auto& column = prescaled.column;
  if (solution->col_value.size() == column.size()) {
    for (std::size_t j = 0; j < column.size(); ++j) solution->col_value[j] *= column[j];
  }
  if (solution->col_dual.size() == column.size()) {
    for (std::size_t j = 0; j < column.size(); ++j) solution->col_dual[j] /= column[j];
  }
  if (solution->primal_ray.size() == column.size()) {
    for (std::size_t j = 0; j < column.size(); ++j) solution->primal_ray[j] *= column[j];
  }
  if (solution->row_activity.size() == row.size()) {
    for (std::size_t i = 0; i < row.size(); ++i) solution->row_activity[i] /= row[i];
  }
  if (solution->row_dual.size() == row.size()) {
    for (std::size_t i = 0; i < row.size(); ++i) solution->row_dual[i] *= row[i];
  }
  if (solution->farkas_dual.size() == row.size()) {
    for (std::size_t i = 0; i < row.size(); ++i) solution->farkas_dual[i] *= row[i];
  }
}

}  // namespace sankhya
