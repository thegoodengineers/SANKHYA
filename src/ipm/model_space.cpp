// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's iterate measured in the model's own units (#582).
//
// Solution::recompute_quality (src/core/model.cpp) is the definition this file mirrors,
// term by term, on the scaled iterate. A change to one without the other brings back the
// disagreement #582 was filed about, and test_ipm_model_space.cpp checks the two against
// each other on randomly scaled models.

#include "ipm/model_space.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sankhya::ipm {
namespace {

double nearest_bound_distance(double value, double lo, double hi) {
  double distance = kInfinity;
  if (is_finite_bound(lo)) distance = std::min(distance, std::fabs(value - lo));
  if (is_finite_bound(hi)) distance = std::min(distance, std::fabs(value - hi));
  return distance;
}

}  // namespace

ModelSpaceMeasure measure_in_model_space(const ScaledIterate& it) {
  const SparseMatrix& a = *it.matrix;
  const Index m = a.num_rows();
  const Index n = a.num_cols();
  const std::vector<double>& x = *it.x;
  const std::vector<double>& y = *it.y;
  const std::vector<double>& lower = *it.lower;
  const std::vector<double>& upper = *it.upper;
  const auto dr = [&](Index i) {
    return it.scaling == nullptr ? 1.0 : it.scaling->row[static_cast<std::size_t>(i)];
  };
  const auto dc = [&](Index j) {
    return it.scaling == nullptr ? 1.0 : it.scaling->column[static_cast<std::size_t>(j)];
  };
  const auto M = static_cast<std::size_t>(m);

  ModelSpaceMeasure out;
  out.row_violation.assign(M, 0.0);
  out.dual_violation.assign(static_cast<std::size_t>(n + m), 0.0);

  // ROWS. The activity the guard recomputes from x is (Ahat xhat)_i / Dr_i, and its scale is
  // the largest term of that sum, max(1, |Ahat_ij xhat_j| / Dr_i). Everything is kept in
  // scaled units until the one division by Dr at the end.
  std::vector<double> activity(M, 0.0);
  std::vector<double> row_scale(M, 1.0);
  for (Index j = 0; j < n; ++j) {
    const double xj = x[static_cast<std::size_t>(j)];
    if (xj == 0.0) continue;
    const ColumnView column = a.column(j);
    for (Index q = 0; q < column.size; ++q) {
      const auto r = static_cast<std::size_t>(column.rows[q]);
      const double term = column.values[q] * xj;
      activity[r] += term;
      row_scale[r] = std::max(row_scale[r], std::fabs(term) / dr(column.rows[q]));
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const auto k = static_cast<std::size_t>(n + i);
    double violation = 0.0;
    if (is_finite_bound(lower[k])) violation = std::max(violation, lower[k] - activity[u]);
    if (is_finite_bound(upper[k])) violation = std::max(violation, activity[u] - upper[k]);
    out.row_violation[u] = violation / dr(i) / row_scale[u];
    out.primal = std::max(out.primal, out.row_violation[u]);
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    double violation = 0.0;
    if (is_finite_bound(lower[u])) violation = std::max(violation, lower[u] - x[u]);
    if (is_finite_bound(upper[u])) violation = std::max(violation, x[u] - upper[u]);
    if (violation <= 0.0) continue;
    out.primal =
        std::max(out.primal, violation * dc(j) / std::max(1.0, std::fabs(x[u] * dc(j))));
  }

  // COLUMNS. The model's cost is chat / Dc, a term A_ij y_i is Ahat_ij yhat_i / Dc, and the
  // reported reduced cost (zl - zu) / Dc. The consistency residual |c - A^T y - d| is then
  // the scaled one over Dc, and the complementarity product |d| * distance is invariant.
  const std::vector<double>& cost = *it.cost;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (lower[u] == upper[u] && is_finite_bound(lower[u])) continue;  // any d is admissible
    const double factor = dc(j);
    double implied = cost[u];
    double terms = std::fabs(cost[u]);
    const ColumnView column = a.column(j);
    for (Index q = 0; q < column.size; ++q) {
      const double term = column.values[q] * y[static_cast<std::size_t>(column.rows[q])];
      implied -= term;
      terms = std::max(terms, std::fabs(term));
    }
    const double d_hat = (*it.zl)[u] - (*it.zu)[u];
    const double scale = std::max(1.0, terms / factor);
    double worst = std::fabs(implied - d_hat) / factor / scale;
    const double d = d_hat / factor;
    if (d > 0.0 && !is_finite_bound(lower[u])) worst = std::max(worst, d / scale);
    if (d < 0.0 && !is_finite_bound(upper[u])) worst = std::max(worst, -d / scale);
    const double distance = nearest_bound_distance(x[u], lower[u], upper[u]);
    if (d_hat != 0.0 && is_finite_bound(distance)) {
      const double product = std::fabs(d_hat) * distance;
      out.complementarity = std::max(out.complementarity, product);
      worst = std::max(
          worst, product / std::max(1.0, scale * std::max(1.0, std::fabs(x[u] * factor))));
    }
    out.dual_violation[u] = worst;
    out.dual = std::max(out.dual, worst);
  }

  // ROW PRICES. y_i = Dr_i yhat_i against the whole price vector's size; the distance of the
  // activity to its nearest row bound is the scaled one over Dr, so the product is again
  // invariant and only its denominator - the price scale times the row's term scale - moves.
  double dual_norm = 0.0;
  for (Index i = 0; i < m; ++i) {
    dual_norm = std::max(dual_norm, std::fabs(dr(i) * y[static_cast<std::size_t>(i)]));
  }
  const double price_scale = std::max(1.0, dual_norm);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const auto k = static_cast<std::size_t>(n + i);
    if (lower[k] == upper[k]) continue;  // equality row: any multiplier is admissible
    const double price = dr(i) * y[u];
    double worst = 0.0;
    if (price > 0.0 && !is_finite_bound(lower[k])) worst = price / price_scale;
    if (price < 0.0 && !is_finite_bound(upper[k])) worst = -price / price_scale;
    const double distance = nearest_bound_distance(activity[u], lower[k], upper[k]);
    if (y[u] != 0.0 && is_finite_bound(distance)) {
      const double product = std::fabs(y[u]) * distance;
      out.complementarity = std::max(out.complementarity, product);
      worst = std::max(worst, product / std::max(1.0, price_scale * row_scale[u]));
    }
    out.dual_violation[k] = worst;
    out.dual = std::max(out.dual, worst);
  }
  return out;
}

}  // namespace sankhya::ipm
