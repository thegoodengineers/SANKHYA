// SPDX-License-Identifier: Apache-2.0
// SANKHYA - model features for engine selection (#477). See the header for the definitions
// and the citations.

#include "engine_features.hpp"

#include "ipm/dense_columns.hpp"
#include "la/ldl.hpp"
#include "la/normal_pattern.hpp"
#include "sankhya/tolerances.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

namespace sankhya {
namespace {

/// The symbolic pass (see the header): the pattern of A A^T over the columns the interior
/// point keeps in its normal equations, then SparseLdl::analyze() under the feature budgets.
/// The columns left out are the ones its own find_dense_columns() picks at the default
/// options, and the fixed columns, which contribute nothing to A Theta A^T.
void symbolic_fill(const Model& model, EngineFeatures* f) {
  const SparseMatrix& a = model.matrix;
  const auto n = static_cast<std::size_t>(f->columns);
  std::vector<char> eligible(n, 1);
  for (std::size_t j = 0; j < n; ++j) {
    if (model.col_lower[j] == model.col_upper[j]) eligible[j] = 0;
  }
  std::vector<char> skip(n, 0);
  for (std::size_t j = 0; j < n; ++j) skip[j] = eligible[j] != 0 ? 0 : 1;
  for (const Index j : ipm::find_dense_columns(a, eligible, tol::kIpmDenseColumnFactor,
                                               tol::kIpmMaxDenseColumns)) {
    skip[static_cast<std::size_t>(j)] = 1;
  }
  const double nnz = static_cast<double>(f->nonzeros);
  const auto capped = [&](std::int64_t at_least) {
    f->cholesky_capped = true;
    f->cholesky_nonzeros = static_cast<double>(at_least);
    f->cholesky_fill_ratio = nnz > 0.0 ? f->cholesky_nonzeros / nnz : 0.0;
  };
  // L holds the lower triangle of A A^T and more, so a count over the budget settles it.
  const NormalPrediction normal = predict_normal_nonzeros(a, skip, kFeatureNormalBudget);
  if (normal.over_cap) {
    capped(normal.nonzeros);
    return;
  }
  std::vector<double> theta(n, 1.0);
  for (std::size_t j = 0; j < n; ++j) {
    if (skip[j] != 0) theta[j] = 0.0;
  }
  SparseMatrix lower;
  if (!normal_equations_lower(a, theta, {}, 1.0, &lower)) {
    capped(kFeatureNormalBudget);
    return;
  }
  SparseLdl ldl;
  // The ordering's live storage is bounded too: a few times the matrix it orders.
  ldl.set_ordering_budget(static_cast<std::size_t>(4 * kFeatureNormalBudget));
  ldl.set_factor_budget(kFeatureFactorBudget);
  if (!ldl.analyze(lower)) {
    capped(kFeatureFactorBudget);
    return;
  }
  f->cholesky_nonzeros =
      static_cast<double>(ldl.factor_nonzeros()) + static_cast<double>(ldl.dimension());
  f->cholesky_fill_ratio = nnz > 0.0 ? f->cholesky_nonzeros / nnz : 0.0;
}

}  // namespace

EngineFeatures compute_engine_features(const Model& model, bool symbolic) {
  EngineFeatures f;
  f.rows = model.num_rows();
  f.columns = model.num_cols();
  f.nonzeros = model.num_nonzeros();
  const double m = static_cast<double>(f.rows);
  const double n = static_cast<double>(f.columns);
  const double nnz = static_cast<double>(f.nonzeros);
  if (f.rows > 0 && f.columns > 0) f.density = nnz / (m * n);
  if (f.columns > 0) f.rows_per_column = m / n;

  const std::vector<Index>& starts = model.matrix.column_starts();
  // The interior point's test (find_dense_columns, src/ipm/dense_columns.cpp).
  const double dense_floor = tol::kIpmDenseColumnFactor * std::sqrt(m);
  double outer_products = 0.0;
  for (Index j = 0; j < f.columns; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const Index count = starts.size() > u + 1 ? starts[u + 1] - starts[u] : 0;
    f.max_column_count = std::max(f.max_column_count, count);
    if (static_cast<double>(count) > dense_floor) ++f.dense_columns;
    outer_products += static_cast<double>(count) * static_cast<double>(count);
  }
  f.normal_equations_nnz_bound = outer_products;
  if (f.nonzeros > 0) f.normal_equations_ratio = outer_products / nnz;

  Index equalities = 0;
  for (Index i = 0; i < f.rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (is_finite_bound(model.row_lower[u]) && model.row_lower[u] == model.row_upper[u]) {
      ++equalities;
    }
  }
  if (f.rows > 0) f.equality_row_share = static_cast<double>(equalities) / m;

  Index boxed = 0;
  Index free_cols = 0;
  Index fixed = 0;
  for (Index j = 0; j < f.columns; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const bool lo = is_finite_bound(model.col_lower[u]);
    const bool hi = is_finite_bound(model.col_upper[u]);
    if (lo && hi && model.col_lower[u] == model.col_upper[u]) {
      ++fixed;
    } else if (lo && hi) {
      ++boxed;
    } else if (!lo && !hi) {
      ++free_cols;
    }
  }
  if (f.columns > 0) {
    f.integer_share = static_cast<double>(model.num_integer_columns()) / n;
    f.boxed_column_share = static_cast<double>(boxed) / n;
    f.free_column_share = static_cast<double>(free_cols) / n;
    f.fixed_column_share = static_cast<double>(fixed) / n;
  }
  if (symbolic && f.rows > 0 && f.columns > 0) symbolic_fill(model, &f);
  return f;
}

std::string format_engine_features_json(const EngineFeatures& f) {
  return fmt::format(
      "{{\"rows\": {}, \"columns\": {}, \"nonzeros\": {}, \"density\": {:.17g}, "
      "\"max_column_count\": {}, \"dense_columns\": {}, \"rows_per_column\": {:.17g}, "
      "\"equality_row_share\": {:.17g}, \"integer_share\": {:.17g}, "
      "\"boxed_column_share\": {:.17g}, \"free_column_share\": {:.17g}, "
      "\"fixed_column_share\": {:.17g}, \"normal_equations_nnz_bound\": {:.17g}, "
      "\"normal_equations_ratio\": {:.17g}, \"cholesky_nonzeros\": {:.17g}, "
      "\"cholesky_fill_ratio\": {:.17g}, \"cholesky_capped\": {}}}",
      f.rows, f.columns, f.nonzeros, f.density, f.max_column_count, f.dense_columns,
      f.rows_per_column, f.equality_row_share, f.integer_share, f.boxed_column_share,
      f.free_column_share, f.fixed_column_share, f.normal_equations_nnz_bound,
      f.normal_equations_ratio, f.cholesky_nonzeros, f.cholesky_fill_ratio,
      f.cholesky_capped ? "true" : "false");
}

}  // namespace sankhya
