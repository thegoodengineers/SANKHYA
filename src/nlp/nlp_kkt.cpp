// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the primal-dual system of the NLP interior point, and its inertia (NLP stage 2).
// See nlp_kkt.hpp.

#include "nlp/nlp_kkt.hpp"

#include <algorithm>
#include <cmath>

#include "sankhya/tolerances.hpp"

namespace sankhya::nlp {

bool NlpKkt::analyze(const BarrierNlp& p, const SparseLdl::ShouldStop& should_stop) {
  n_ = p.n();
  m_ = p.m();
  const auto dim = static_cast<std::size_t>(n_ + m_);
  // The Jacobian by column: for column j, the (row i, Jacobian slot) pairs in row order.
  std::vector<std::vector<std::pair<Index, Index>>> by_column(static_cast<std::size_t>(n_));
  for (Index i = 0; i < m_; ++i) {
    for (Index k = p.jac_starts()[static_cast<std::size_t>(i)];
         k < p.jac_starts()[static_cast<std::size_t>(i) + 1]; ++k) {
      by_column[static_cast<std::size_t>(p.jac_cols()[static_cast<std::size_t>(k)])]
          .emplace_back(i, k);
    }
  }
  starts_.assign(1, 0);
  rows_.clear();
  hess_slot_.assign(p.hess_rows().size(), 0);
  jac_slot_.assign(p.jac_cols().size(), 0);
  diag_slot_.assign(dim, 0);
  // Column j < n: W's rows (the diagonal first, every pattern holds it), then J's rows n + i.
  for (Index j = 0; j < n_; ++j) {
    diag_slot_[static_cast<std::size_t>(j)] = static_cast<Index>(rows_.size());
    for (Index k = p.hess_starts()[static_cast<std::size_t>(j)];
         k < p.hess_starts()[static_cast<std::size_t>(j) + 1]; ++k) {
      hess_slot_[static_cast<std::size_t>(k)] = static_cast<Index>(rows_.size());
      rows_.push_back(p.hess_rows()[static_cast<std::size_t>(k)]);
    }
    for (const auto& [i, k] : by_column[static_cast<std::size_t>(j)]) {
      jac_slot_[static_cast<std::size_t>(k)] = static_cast<Index>(rows_.size());
      rows_.push_back(n_ + i);
    }
    starts_.push_back(static_cast<Index>(rows_.size()));
  }
  // Column n + i: its diagonal only.
  for (Index i = 0; i < m_; ++i) {
    diag_slot_[static_cast<std::size_t>(n_ + i)] = static_cast<Index>(rows_.size());
    rows_.push_back(n_ + i);
    starts_.push_back(static_cast<Index>(rows_.size()));
  }
  signs_.assign(dim, 1);
  for (Index i = 0; i < m_; ++i) signs_[static_cast<std::size_t>(n_ + i)] = -1;
  values_.assign(rows_.size(), 1.0);
  matrix_.assign_columns(n_ + m_, n_ + m_, starts_, rows_, values_);
  return ldl_.analyze(matrix_, should_stop);
}

bool NlpKkt::factorize(const Vec& hessian, const Vec& jacobian, const Vec& sigma,
                       const std::vector<char>& fixed, double delta_w, double delta_c,
                       const SparseLdl::ShouldStop& should_stop) {
  delta_c_ = delta_c;
  std::fill(values_.begin(), values_.end(), 0.0);
  for (std::size_t k = 0; k < hessian.size(); ++k) {
    values_[static_cast<std::size_t>(hess_slot_[k])] += hessian[k];
  }
  for (std::size_t k = 0; k < jacobian.size(); ++k) {
    values_[static_cast<std::size_t>(jac_slot_[k])] = jacobian[k];
  }
  for (Index j = 0; j < n_; ++j) {
    values_[static_cast<std::size_t>(diag_slot_[static_cast<std::size_t>(j)])] +=
        sigma[static_cast<std::size_t>(j)] + delta_w;
  }
  // A fixed column: zero its row and column, 1 on its diagonal.
  if (!fixed.empty()) {
    for (Index j = 0; j < n_ + m_; ++j) {
      for (Index k = starts_[static_cast<std::size_t>(j)];
           k < starts_[static_cast<std::size_t>(j) + 1]; ++k) {
        const Index i = rows_[static_cast<std::size_t>(k)];
        const bool row_fixed = i < n_ && fixed[static_cast<std::size_t>(i)] != 0;
        const bool col_fixed = j < n_ && fixed[static_cast<std::size_t>(j)] != 0;
        if (row_fixed || col_fixed) values_[static_cast<std::size_t>(k)] = i == j ? 1.0 : 0.0;
      }
    }
  }
  const double dual = std::max(delta_c, tol::kNlpDualRegularization);
  for (Index i = 0; i < m_; ++i) {
    values_[static_cast<std::size_t>(diag_slot_[static_cast<std::size_t>(n_ + i)])] = -dual;
  }
  matrix_.assign_columns(n_ + m_, n_ + m_, starts_, rows_, values_);
  if (!ldl_.factorize_quasidefinite(matrix_, signs_, tol::kNlpPivotFloor, should_stop)) {
    return false;
  }
  // Restore the stated (2,2) block for the refinement's residuals.
  for (Index i = 0; i < m_; ++i) {
    values_[static_cast<std::size_t>(diag_slot_[static_cast<std::size_t>(n_ + i)])] = -delta_c;
  }
  return ldl_.regularized_pivots() == 0;
}

void NlpKkt::multiply(const Vec& x, Vec* y) const {
  y->assign(x.size(), 0.0);
  const Index dim = n_ + m_;
  for (Index j = 0; j < dim; ++j) {
    const double xj = x[static_cast<std::size_t>(j)];
    for (Index k = starts_[static_cast<std::size_t>(j)];
         k < starts_[static_cast<std::size_t>(j) + 1]; ++k) {
      const auto i = static_cast<std::size_t>(rows_[static_cast<std::size_t>(k)]);
      const double v = values_[static_cast<std::size_t>(k)];
      (*y)[i] += v * xj;
      if (i != static_cast<std::size_t>(j)) (*y)[static_cast<std::size_t>(j)] += v * x[i];
    }
  }
}

double NlpKkt::solve(Vec* rhs) const {
  const Vec b = *rhs;
  double scale = 1.0;
  for (const double v : b) scale = std::max(scale, std::fabs(v));
  ldl_.solve(rhs->data());
  Vec r, kx;
  double residual = 0.0;
  // Iterative refinement against the stated matrix (Golub and Van Loan, "Matrix
  // Computations", 4th ed., sec. 3.5.3): the factors are of a nearby matrix, the corrections
  // converge to the stated system's solution while it is well enough conditioned.
  for (int step = 0; step <= tol::kNlpRefinementSteps; ++step) {
    multiply(*rhs, &kx);
    r.assign(b.size(), 0.0);
    residual = 0.0;
    for (std::size_t k = 0; k < b.size(); ++k) {
      r[k] = b[k] - kx[k];
      residual = std::max(residual, std::fabs(r[k]));
    }
    residual /= scale;
    if (residual <= tol::kNlpRefinementTolerance || step == tol::kNlpRefinementSteps) break;
    ldl_.solve(r.data());
    for (std::size_t k = 0; k < b.size(); ++k) (*rhs)[k] += r[k];
  }
  return residual;
}

}  // namespace sankhya::nlp
