// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dense columns in the interior point's normal equations (#467).
// See dense_columns.hpp for the method and the references.

#include "ipm/dense_columns.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>

#include "sankhya/tolerances.hpp"

namespace sankhya::ipm {
namespace {

// The constants (kIpmPcg*, kIpmDenseSupportRatio) are in include/sankhya/tolerances.hpp,
// with the measurements behind them.
constexpr double kPcgRelativeResidual = tol::kIpmPcgTargetBackwardError;
constexpr int kPcgMaxIterations = tol::kIpmPcgMaxIterations;
constexpr double kPcgAccepted = tol::kIpmPcgAcceptedBackwardError;
constexpr int kPcgStagnation = tol::kIpmPcgStagnationSteps;
constexpr double kSupportRatio = tol::kIpmDenseSupportRatio;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

double norm(const std::vector<double>& a) {
  return std::sqrt(dot(a, a));
}

double inf_norm(const std::vector<double>& a) {
  double largest = 0.0;
  for (const double v : a) largest = std::max(largest, std::fabs(v));
  return largest;
}

}  // namespace

std::vector<Index> find_dense_columns(const SparseMatrix& a, const std::vector<char>& eligible,
                                      double factor, Index cap) {
  std::vector<Index> dense;
  if (factor <= 0.0 || cap <= 0) return dense;
  const double threshold = factor * std::sqrt(static_cast<double>(a.num_rows()));
  for (Index j = 0; j < a.num_cols(); ++j) {
    if (!eligible.empty() && eligible[static_cast<std::size_t>(j)] == 0) continue;
    if (static_cast<double>(a.column(j).size) > threshold) dense.push_back(j);
  }
  std::stable_sort(dense.begin(), dense.end(),
                   [&](Index x, Index y) { return a.column(x).size > a.column(y).size; });
  if (static_cast<Index>(dense.size()) > cap) dense.resize(static_cast<std::size_t>(cap));
  std::sort(dense.begin(), dense.end());
  return dense;
}

void DenseColumnCorrection::set_columns(const SparseMatrix& a, std::vector<Index> columns) {
  columns_ = std::move(columns);
  mask_.assign(static_cast<std::size_t>(a.num_cols()), 0);
  for (const Index j : columns_) mask_[static_cast<std::size_t>(j)] = 1;
}

void DenseColumnCorrection::sparse_theta(const std::vector<double>& theta,
                                         std::vector<double>* out) const {
  *out = theta;
  for (const Index j : columns_) (*out)[static_cast<std::size_t>(j)] = 0.0;
}

Index DenseColumnCorrection::preconditioner_shift(const SparseMatrix& a,
                                                  const std::vector<double>& theta,
                                                  const std::vector<double>& row_shift,
                                                  double delta,
                                                  std::vector<double>* shift) const {
  const auto m = static_cast<std::size_t>(a.num_rows());
  const bool have_shift = row_shift.size() == m;
  std::vector<double> sparse_diag(m, delta);
  std::vector<double> dense_diag(m, 0.0);
  for (Index j = 0; j < a.num_cols(); ++j) {
    const double t = theta[static_cast<std::size_t>(j)];
    if (t == 0.0) continue;
    std::vector<double>& target =
        mask_[static_cast<std::size_t>(j)] != 0 ? dense_diag : sparse_diag;
    const ColumnView column = a.column(j);
    for (Index q = 0; q < column.size; ++q) {
      target[static_cast<std::size_t>(column.rows[q])] +=
          t * column.values[q] * column.values[q];
    }
  }
  shift->assign(m, 0.0);
  Index supported = 0;
  for (std::size_t i = 0; i < m; ++i) {
    const double own = have_shift ? row_shift[i] : 0.0;
    (*shift)[i] = own;
    if (sparse_diag[i] + own < kSupportRatio * dense_diag[i]) {
      (*shift)[i] += dense_diag[i];
      ++supported;
    }
  }
  return supported;
}

bool DenseColumnCorrection::prepare(const SparseLdl& ldl, const SparseMatrix& a,
                                    const std::vector<double>& theta,
                                    const std::vector<double>& row_shift, double delta) {
  ldl_ = &ldl;
  a_ = &a;
  theta_ = theta;
  row_shift_ = row_shift;
  delta_ = delta;
  const auto k = columns_.size();
  const auto m = static_cast<std::size_t>(a.num_rows());
  sqrt_theta_d_.resize(k);
  for (std::size_t c = 0; c < k; ++c) {
    sqrt_theta_d_[c] = std::sqrt(std::max(0.0, theta[static_cast<std::size_t>(columns_[c])]));
  }
  // S = I + V^T M_s^-1 V, one solve per dense column: w = M_s^-1 v_c, then column c of S is
  // the k dot products v_i^T w, each over the entries of the sparse column i.
  woodbury_ = true;
  schur_.assign(k * k, 0.0);
  std::vector<double> w(m);
  for (std::size_t c = 0; c < k; ++c) {
    std::fill(w.begin(), w.end(), 0.0);
    const ColumnView column = a.column(columns_[c]);
    for (Index q = 0; q < column.size; ++q) {
      w[static_cast<std::size_t>(column.rows[q])] = column.values[q] * sqrt_theta_d_[c];
    }
    ldl.solve(w.data());
    for (std::size_t i = 0; i < k; ++i) {
      const ColumnView other = a.column(columns_[i]);
      double sum = 0.0;
      for (Index q = 0; q < other.size; ++q) {
        sum += other.values[q] * w[static_cast<std::size_t>(other.rows[q])];
      }
      schur_[c * k + i] = sqrt_theta_d_[i] * sum;
    }
  }
  // Symmetrize (the two halves differ by the rounding of the solves), add I, and factor
  // S = L L^T in place, column-major lower.
  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t i = c + 1; i < k; ++i) {
      const double mean = 0.5 * (schur_[c * k + i] + schur_[i * k + c]);
      schur_[c * k + i] = mean;
      schur_[i * k + c] = mean;
    }
    schur_[c * k + c] += 1.0;
  }
  for (std::size_t j = 0; j < k; ++j) {
    double pivot = schur_[j * k + j];
    for (std::size_t p = 0; p < j; ++p) pivot -= schur_[p * k + j] * schur_[p * k + j];
    // Every eigenvalue of S is at least 1 in exact arithmetic. A pivot below
    // kIpmDenseSchurMinPivot (one half) means the solves it was built from were not
    // accurate - M_s is nearly singular, in a direction no diagonal support reaches (a
    // rank-deficient A_s) - and the product form would carry that error into every
    // direction. solve() then uses the sparse factor alone.
    if (!(pivot > tol::kIpmDenseSchurMinPivot) || !std::isfinite(pivot)) {
      woodbury_ = false;
      return true;
    }
    const double root = std::sqrt(pivot);
    schur_[j * k + j] = root;
    for (std::size_t i = j + 1; i < k; ++i) {
      double value = schur_[j * k + i];
      for (std::size_t p = 0; p < j; ++p) value -= schur_[p * k + i] * schur_[p * k + j];
      schur_[j * k + i] = value / root;
    }
  }
  return true;
}

void DenseColumnCorrection::apply_preconditioner(const std::vector<double>& r,
                                                 std::vector<double>* out) const {
  const auto k = columns_.size();
  const auto m = r.size();
  *out = r;
  ldl_->solve(out->data());  // w = M_s^-1 r
  if (k == 0) return;
  // t = V^T w, then S u = t by the Cholesky factor: L y = t, L^T u = y.
  std::vector<double> t(k, 0.0);
  for (std::size_t c = 0; c < k; ++c) {
    const ColumnView column = a_->column(columns_[c]);
    double sum = 0.0;
    for (Index q = 0; q < column.size; ++q) {
      sum += column.values[q] * (*out)[static_cast<std::size_t>(column.rows[q])];
    }
    t[c] = sqrt_theta_d_[c] * sum;
  }
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t p = 0; p < j; ++p) t[j] -= schur_[p * k + j] * t[p];
    t[j] /= schur_[j * k + j];
  }
  for (std::size_t j = k; j-- > 0;) {
    for (std::size_t i = j + 1; i < k; ++i) t[j] -= schur_[j * k + i] * t[i];
    t[j] /= schur_[j * k + j];
  }
  // out -= M_s^-1 V u.
  std::vector<double> vu(m, 0.0);
  for (std::size_t c = 0; c < k; ++c) {
    const ColumnView column = a_->column(columns_[c]);
    const double scale = sqrt_theta_d_[c] * t[c];
    for (Index q = 0; q < column.size; ++q) {
      vu[static_cast<std::size_t>(column.rows[q])] += column.values[q] * scale;
    }
  }
  ldl_->solve(vu.data());
  for (std::size_t i = 0; i < m; ++i) (*out)[i] -= vu[i];
}

double DenseColumnCorrection::terms_of_product(const std::vector<double>& v) const {
  // || |A| Theta |A|^T |v| + (shift + delta) |v| ||_inf: every term of M v in absolute value.
  const auto n = static_cast<std::size_t>(a_->num_cols());
  const auto m = v.size();
  std::vector<double> atv(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    const ColumnView column = a_->column(static_cast<Index>(j));
    double sum = 0.0;
    for (Index q = 0; q < column.size; ++q) {
      sum +=
          std::fabs(column.values[q]) * std::fabs(v[static_cast<std::size_t>(column.rows[q])]);
    }
    atv[j] = theta_[j] * sum;
  }
  std::vector<double> out(m, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    const ColumnView column = a_->column(static_cast<Index>(j));
    for (Index q = 0; q < column.size; ++q) {
      out[static_cast<std::size_t>(column.rows[q])] += std::fabs(column.values[q]) * atv[j];
    }
  }
  const bool have_shift = row_shift_.size() == m;
  double largest = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    out[i] += (delta_ + (have_shift ? row_shift_[i] : 0.0)) * std::fabs(v[i]);
    largest = std::max(largest, out[i]);
  }
  return largest;
}

void DenseColumnCorrection::multiply_full(const std::vector<double>& v,
                                          std::vector<double>* out) const {
  // M v = A (Theta (A^T v)) + (shift + delta) v, from A itself: M is never formed.
  const auto n = static_cast<std::size_t>(a_->num_cols());
  std::vector<double> atv(n, 0.0);
  a_->transpose_multiply_add(v.data(), atv.data());
  for (std::size_t j = 0; j < n; ++j) atv[j] *= theta_[j];
  std::fill(out->begin(), out->end(), 0.0);
  a_->multiply_add(atv.data(), out->data());
  const bool have_shift = row_shift_.size() == v.size();
  for (std::size_t i = 0; i < v.size(); ++i) {
    (*out)[i] += (delta_ + (have_shift ? row_shift_[i] : 0.0)) * v[i];
  }
}

PcgReport DenseColumnCorrection::conjugate_gradients(const std::vector<double>& b,
                                                     bool woodbury, int max_iterations,
                                                     int patience,
                                                     std::vector<double>* x) const {
  PcgReport report;
  const std::size_t m = b.size();
  const auto precondition = [&](const std::vector<double>& r, std::vector<double>* z) {
    if (woodbury) {
      apply_preconditioner(r, z);
    } else {
      *z = r;
      ldl_->solve(z->data());
    }
  };
  precondition(b, x);
  // The scale of the terms, from the first approximation: close enough to the answer's for
  // a stopping test, and one pair of products instead of one per step.
  const double scale = inf_norm(b) + terms_of_product(*x);
  std::vector<double> r(m), q(m), z(m);
  multiply_full(*x, &q);
  for (std::size_t i = 0; i < m; ++i) r[i] = b[i] - q[i];
  double residual = inf_norm(r);
  double checkpoint = residual;
  int since_checkpoint = 0;
  precondition(r, &z);
  std::vector<double> p = z;
  double rz = dot(r, z);
  while (residual > kPcgRelativeResidual * scale && report.iterations < max_iterations) {
    multiply_full(p, &q);
    const double curvature = dot(p, q);
    if (!(curvature > 0.0) || !(rz > 0.0)) {
      // Not positive definite as computed: the preconditioner (or M, at its rounding floor)
      // has lost definiteness. What is in hand is kept, and not called converged.
      report.broke_down = true;
      break;
    }
    const double alpha = rz / curvature;
    for (std::size_t i = 0; i < m; ++i) {
      (*x)[i] += alpha * p[i];
      r[i] -= alpha * q[i];
    }
    ++report.iterations;
    residual = inf_norm(r);
    if (residual <= 0.5 * checkpoint) {
      checkpoint = residual;
      since_checkpoint = 0;
    } else if (++since_checkpoint >= kPcgStagnation && report.iterations > patience) {
      break;
    }
    precondition(r, &z);
    const double rz_next = dot(r, z);
    const double beta = rz_next / rz;
    rz = rz_next;
    for (std::size_t i = 0; i < m; ++i) p[i] = z[i] + beta * p[i];
  }
  // The recurrence's residual drifts from the true one; the report states the true one.
  multiply_full(*x, &q);
  for (std::size_t i = 0; i < m; ++i) r[i] = b[i] - q[i];
  report.relative_residual = inf_norm(r) / (inf_norm(b) + terms_of_product(*x));
  if (!std::isfinite(report.relative_residual)) {
    report.relative_residual = std::numeric_limits<double>::infinity();
  }
  report.converged = !report.broke_down && report.relative_residual <= kPcgAccepted;
  return report;
}

PcgReport DenseColumnCorrection::solve(double* rhs) const {
  const auto m = static_cast<std::size_t>(a_->num_rows());
  const std::vector<double> b(rhs, rhs + m);
  if (norm(b) == 0.0) {
    PcgReport report;
    report.converged = true;
    return report;
  }
  // First the Woodbury preconditioner, which is M^-1 up to rounding when M_s is well
  // conditioned. When it is not - or when the Schur complement could not be factored - the
  // factor of M_s alone. Without diagonal support, M_s^-1 M = I + M_s^-1 V V^T has every
  // eigenvalue but k equal to one, so conjugate gradients finish in k + 1 steps in exact
  // arithmetic however singular M_s is (Golub and Van Loan, sec. 11.5). With b rows
  // supported the factor is of M_s + B, and (M_s + B)^-1 M = I + (M_s + B)^-1 (V V^T - B)
  // has up to k + b eigenvalues away from one, so the budget below is not a guarantee:
  // a solve that runs out of it is reported unconverged, and the caller must not use it.
  std::vector<double> x;
  PcgReport report;
  if (woodbury_) {
    report = conjugate_gradients(b, true, kPcgMaxIterations, 0, &x);
    report.preconditioner = PcgReport::Preconditioner::kWoodbury;
    if (report.converged) {
      std::copy(x.begin(), x.end(), rhs);
      return report;
    }
  }
  std::vector<double> y;
  PcgReport plain =
      conjugate_gradients(b, false, static_cast<int>(columns_.size()) + kPcgMaxIterations,
                          static_cast<int>(columns_.size()) + 1, &y);
  plain.preconditioner = PcgReport::Preconditioner::kSparseFactor;
  plain.iterations += report.iterations;
  if (!woodbury_ || plain.relative_residual <= report.relative_residual) {
    std::copy(y.begin(), y.end(), rhs);
    return plain;
  }
  std::copy(x.begin(), x.end(), rhs);
  return report;
}

}  // namespace sankhya::ipm
