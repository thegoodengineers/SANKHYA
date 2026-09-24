// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the equality-and-bounds form and the quasi-definite KKT matrix of the proximal
// interior point for convex QP (#490). See qp_ipm.cpp for the method and its references.

#include "qp_ipm_system.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "sankhya/types.hpp"

namespace sankhya::qp::ipm_detail {

Standard standardize(const Model& model) {
  Standard s;
  const double sense = model.sense_multiplier();
  s.n = model.num_cols();
  const Index m = model.num_rows();
  s.column_of.assign(static_cast<std::size_t>(s.n), -1);
  s.fixed_value.assign(static_cast<std::size_t>(s.n), 0.0);
  for (Index j = 0; j < s.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] == model.col_upper[u]) {
      s.fixed_value[u] = model.col_lower[u];
    } else {
      s.column_of[u] = s.free_n++;
    }
  }
  s.row_of.assign(static_cast<std::size_t>(m), -1);
  std::vector<Index> slack_of_row(static_cast<std::size_t>(m), -1);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const bool lo = is_finite_bound(model.row_lower[u]);
    const bool hi = is_finite_bound(model.row_upper[u]);
    if (!lo && !hi) continue;  // a free row constrains nothing and gets a zero dual
    s.row_of[u] = s.rows++;
    if (!(lo && hi && model.row_lower[u] == model.row_upper[u])) {
      slack_of_row[u] = s.slacks++;
    }
  }
  s.cols = s.free_n + s.slacks;
  s.slack_row.assign(static_cast<std::size_t>(s.slacks), -1);
  const auto ucols = static_cast<std::size_t>(s.cols);
  s.g.assign(ucols, 0.0);
  s.lower.assign(ucols, -kInfinity);
  s.upper.assign(ucols, kInfinity);
  s.b.assign(static_cast<std::size_t>(s.rows), 0.0);

  for (Index j = 0; j < s.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const Index k = s.column_of[u];
    if (k < 0) continue;
    s.g[static_cast<std::size_t>(k)] = sense * model.col_cost[u];
    s.lower[static_cast<std::size_t>(k)] = model.col_lower[u];
    s.upper[static_cast<std::size_t>(k)] = model.col_upper[u];
  }
  // The Hessian: a fixed column's terms move into the gradient of the kept ones and drop out.
  s.h.reset(s.free_n, s.free_n);
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    for (Index p = 0; p < column.size; ++p) {
      const Index i = column.rows[p];
      const double v = sense * column.values[p];
      const Index ki = s.column_of[static_cast<std::size_t>(i)];
      const Index kj = s.column_of[static_cast<std::size_t>(j)];
      if (ki >= 0 && kj >= 0) {
        s.h.add_entry(std::max(ki, kj), std::min(ki, kj), v);
      } else if (ki >= 0) {
        s.g[static_cast<std::size_t>(ki)] += v * s.fixed_value[static_cast<std::size_t>(j)];
      } else if (kj >= 0 && i != j) {
        s.g[static_cast<std::size_t>(kj)] += v * s.fixed_value[static_cast<std::size_t>(i)];
      }
    }
  }
  s.h.finalize(0.0);

  s.m.reset(s.rows, s.cols);
  for (Index j = 0; j < s.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const Index k = s.column_of[u];
    const ColumnView column = model.matrix.column(j);
    for (Index p = 0; p < column.size; ++p) {
      const Index r = s.row_of[static_cast<std::size_t>(column.rows[p])];
      if (r < 0) continue;
      if (k >= 0) {
        s.m.add_entry(r, k, column.values[p]);
      } else {
        s.b[static_cast<std::size_t>(r)] -= column.values[p] * s.fixed_value[u];
      }
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const Index r = s.row_of[u];
    if (r < 0) continue;
    const Index slack = slack_of_row[u];
    if (slack < 0) {
      s.b[static_cast<std::size_t>(r)] += model.row_lower[u];
    } else {
      const auto k = static_cast<std::size_t>(s.free_n + slack);
      s.m.add_entry(r, static_cast<Index>(k), -1.0);
      s.slack_row[static_cast<std::size_t>(slack)] = r;
      s.lower[k] = model.row_lower[u];
      s.upper[k] = model.row_upper[u];
    }
  }
  s.m.finalize(0.0);
  return s;
}

/// out = H v (H symmetric, from its lower triangle), over the first h.num_cols() entries.
void hessian_times(const SparseMatrix& h, const std::vector<double>& v,
                   std::vector<double>* out) {
  std::fill(out->begin(), out->end(), 0.0);
  for (Index j = 0; j < h.num_cols(); ++j) {
    const ColumnView column = h.column(j);
    const auto uj = static_cast<std::size_t>(j);
    for (Index p = 0; p < column.size; ++p) {
      const auto ui = static_cast<std::size_t>(column.rows[p]);
      (*out)[ui] += column.values[p] * v[uj];
      if (ui != uj) (*out)[uj] += column.values[p] * v[ui];
    }
  }
}

double inf_norm(const std::vector<double>& v) {
  double norm = 0.0;
  for (const double x : v) norm = std::max(norm, std::fabs(x));
  return norm;
}

KktMatrix::KktMatrix(const Standard& s) : s_(s) {
  const Index dim = s.cols + s.rows;
  starts_.assign(static_cast<std::size_t>(dim) + 1, 0);
  for (Index j = 0; j < dim; ++j) {
    starts_[static_cast<std::size_t>(j)] = static_cast<Index>(rows_.size());
    rows_.push_back(j);
    if (j >= s.cols) continue;
    if (j < s.free_n) {
      const ColumnView q = s.h.column(j);
      for (Index p = 0; p < q.size; ++p) {
        if (q.rows[p] > j) rows_.push_back(q.rows[p]);
      }
    }
    const ColumnView a = s.m.column(j);
    for (Index p = 0; p < a.size; ++p) rows_.push_back(s.cols + a.rows[p]);
  }
  starts_[static_cast<std::size_t>(dim)] = static_cast<Index>(rows_.size());
}

SparseMatrix KktMatrix::build(const std::vector<double>& theta_inverse, double rho,
                              double delta) const {
  const Index dim = s_.cols + s_.rows;
  std::vector<double> values;
  values.reserve(rows_.size());
  for (Index j = 0; j < dim; ++j) {
    if (j >= s_.cols) {
      values.push_back(delta);
      continue;
    }
    double diagonal = theta_inverse[static_cast<std::size_t>(j)] + rho;
    const ColumnView q = j < s_.free_n ? s_.h.column(j) : ColumnView{};
    for (Index p = 0; p < q.size; ++p) {
      if (q.rows[p] == j) diagonal += q.values[p];
    }
    values.push_back(-diagonal);
    for (Index p = 0; p < q.size; ++p) {
      if (q.rows[p] > j) values.push_back(-q.values[p]);
    }
    const ColumnView a = s_.m.column(j);
    for (Index p = 0; p < a.size; ++p) values.push_back(a.values[p]);
  }
  SparseMatrix k;
  k.assign_columns(dim, dim, starts_, rows_, std::move(values));
  return k;
}

void KktMatrix::multiply(const SparseMatrix& k, const std::vector<double>& x,
                         std::vector<double>* out) {
  std::fill(out->begin(), out->end(), 0.0);
  for (Index j = 0; j < k.num_cols(); ++j) {
    const ColumnView c = k.column(j);
    const auto uj = static_cast<std::size_t>(j);
    for (Index p = 0; p < c.size; ++p) {
      const auto ui = static_cast<std::size_t>(c.rows[p]);
      (*out)[ui] += c.values[p] * x[uj];
      if (ui != uj) (*out)[uj] += c.values[p] * x[ui];
    }
  }
}

}  // namespace sankhya::qp::ipm_detail
