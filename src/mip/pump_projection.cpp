// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the feasibility pump's projection LP; see pump_projection.hpp.

#include "mip/pump_projection.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "sankhya/sparse.hpp"

namespace sankhya::mip {

PumpProjection pump_projection(const Model& model, const std::vector<Index>& integer_columns,
                               double integrality_tolerance) {
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  PumpProjection p;
  p.kind.assign(static_cast<std::size_t>(n), -2);
  std::vector<Index> generals;  // the column of each auxiliary, by auxiliary index
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = is_finite_bound(model.col_lower[u])
                          ? std::ceil(model.col_lower[u] - integrality_tolerance)
                          : -kInfinity;
    const double hi = is_finite_bound(model.col_upper[u])
                          ? std::floor(model.col_upper[u] + integrality_tolerance)
                          : kInfinity;
    if (std::isfinite(lo) && std::isfinite(hi) && hi <= lo) continue;  // one value, or none
    if (std::isfinite(lo) && std::isfinite(hi) && hi - lo == 1.0 && model.col_lower[u] == lo &&
        model.col_upper[u] == hi) {
      p.kind[u] = -1;
      continue;
    }
    p.kind[u] = static_cast<Index>(generals.size());
    generals.push_back(j);
  }
  const auto g = static_cast<Index>(generals.size());

  Model& lp = p.lp;
  lp.name = model.name;
  lp.sense = ObjSense::kMinimize;
  lp.col_cost.assign(static_cast<std::size_t>(n + g), 0.0);
  lp.col_lower = model.col_lower;
  lp.col_upper = model.col_upper;
  for (Index k = 0; k < g; ++k) {
    lp.col_cost[static_cast<std::size_t>(n + k)] = 1.0;
    lp.col_lower.push_back(0.0);
    lp.col_upper.push_back(kInfinity);
  }
  lp.col_type.assign(static_cast<std::size_t>(n + g), VarType::kContinuous);
  lp.row_lower = model.row_lower;
  lp.row_upper = model.row_upper;
  lp.row_lower.resize(static_cast<std::size_t>(m + 2 * g), 0.0);
  lp.row_upper.resize(static_cast<std::size_t>(m + 2 * g), kInfinity);
  std::vector<Index> starts;
  std::vector<Index> rows;
  std::vector<double> values;
  starts.reserve(static_cast<std::size_t>(n + g) + 1);
  starts.push_back(0);
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      rows.push_back(column.rows[k]);
      values.push_back(column.values[k]);
    }
    const Index aux = p.kind[static_cast<std::size_t>(j)];
    if (aux >= 0) {
      rows.push_back(m + 2 * aux);
      values.push_back(-1.0);
      rows.push_back(m + 2 * aux + 1);
      values.push_back(1.0);
    }
    starts.push_back(static_cast<Index>(rows.size()));
  }
  for (Index k = 0; k < g; ++k) {
    rows.push_back(m + 2 * k);
    values.push_back(1.0);
    rows.push_back(m + 2 * k + 1);
    values.push_back(1.0);
    starts.push_back(static_cast<Index>(rows.size()));
  }
  lp.matrix.assign_columns(m + 2 * g, n + g, std::move(starts), std::move(rows),
                           std::move(values));
  lp.hessian.reset(n + g, n + g);
  lp.hessian.finalize();
  return p;
}

void set_pump_target(PumpProjection* projection, const Model& model,
                     const std::vector<Index>& integer_columns,
                     const std::vector<double>& rounding) {
  const Index m = model.num_rows();
  Model& lp = projection->lp;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    const Index aux = projection->kind[u];
    if (aux == -1) {
      lp.col_cost[u] = rounding[u] <= model.col_lower[u] ? 1.0 : -1.0;
    } else if (aux >= 0) {
      lp.row_lower[static_cast<std::size_t>(m + 2 * aux)] = -rounding[u];
      lp.row_lower[static_cast<std::size_t>(m + 2 * aux + 1)] = rounding[u];
    }
  }
}

}  // namespace sankhya::mip
