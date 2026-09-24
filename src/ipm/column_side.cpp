// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's normal equations solved on the n x n side (#469).
// See column_side.hpp for the identity and the references.

#include "ipm/column_side.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace sankhya::ipm {
namespace {

/// Conjugate gradients aim at this normwise backward error (Higham 2002, sec. 7.1) ...
constexpr double kTargetBackwardError = 1e-16;
/// ... report convergence at this one ...
constexpr double kAcceptedBackwardError = 1e-12;
/// ... and give up after this many steps, or when the residual has not halved in
/// kStagnation of them: the preconditioner's floor is reached.
constexpr int kMaxIterations = 50;
constexpr int kStagnation = 3;
/// D in the preconditioner is at least this times the row's diagonal of A Theta A^T.
/// Measured with the interior point on the column side over nine Netlib instances (afiro,
/// adlittle, sc50a, sc50b, blend, share2b, scagr7, stocfor1, israel): at 1e-8 all nine end
/// optimal at about two conjugate-gradient steps per solve; 1e-6 takes up to three per
/// solve; at 1e-4 share2b runs to the iteration limit. Without the floor none converged.
constexpr double kDiagonalFloor = 1e-8;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

double inf_norm(const std::vector<double>& a) {
  double largest = 0.0;
  for (const double v : a) largest = std::max(largest, std::fabs(v));
  return largest;
}

}  // namespace

void ColumnSide::set_matrix(const SparseMatrix& a, const std::vector<bool>& fixed) {
  a_ = &a;
  fixed_ = fixed;
  const Index m = a.num_rows();
  const Index n = a.num_cols();
  // A^T by counting: column i of A^T holds row i of A, over the unfixed columns only.
  std::vector<Index> starts(static_cast<std::size_t>(m) + 1, 0);
  for (Index j = 0; j < n; ++j) {
    if (fixed[static_cast<std::size_t>(j)]) continue;
    const ColumnView column = a.column(j);
    for (Index q = 0; q < column.size; ++q) {
      ++starts[static_cast<std::size_t>(column.rows[q]) + 1];
    }
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(m); ++i) starts[i + 1] += starts[i];
  std::vector<Index> next(starts.begin(), starts.end() - 1);
  std::vector<Index> rows(static_cast<std::size_t>(starts.back()));
  std::vector<double> values(static_cast<std::size_t>(starts.back()));
  for (Index j = 0; j < n; ++j) {  // ascending j keeps every column of A^T sorted
    if (fixed[static_cast<std::size_t>(j)]) continue;
    const ColumnView column = a.column(j);
    for (Index q = 0; q < column.size; ++q) {
      const auto slot =
          static_cast<std::size_t>(next[static_cast<std::size_t>(column.rows[q])]++);
      rows[slot] = j;
      values[slot] = column.values[q];
    }
  }
  transpose_.assign_columns(n, m, std::move(starts), std::move(rows), std::move(values));
}

bool ColumnSide::assemble(const std::vector<double>& theta,
                          const std::vector<double>& row_shift, double delta,
                          SparseMatrix* lower, const SparseLdl::ShouldStop& should_stop) {
  const auto n = static_cast<std::size_t>(a_->num_cols());
  const auto m = static_cast<std::size_t>(a_->num_rows());
  theta_.assign(n, 0.0);
  std::vector<double> inverse_theta(n, 1.0);  // a fixed column: an identity row of N
  for (std::size_t j = 0; j < n; ++j) {
    if (fixed_[j] || !(theta[j] > 0.0)) continue;
    theta_[j] = theta[j];
    inverse_theta[j] = 1.0 / theta[j];
  }
  // THE FLOOR ON D IN THE PRECONDITIONER. An equality row's logical is fixed, so its D is
  // the regularization alone, 1e-10, and N = Theta^-1 + A^T D^-1 A would carry 1e10 A_E^T
  // A_E: measured on afiro, the unfloored form handed conjugate gradients a preconditioner
  // with backward error 0.6 and the interior point never converged. The preconditioner is
  // built with D floored at kDiagonalFloor times the row's own diagonal of A Theta A^T,
  // which changes it by a relative kDiagonalFloor on those rows and nowhere else; the
  // conjugate gradients multiply by the true D.
  std::vector<double> row_diagonal(m, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    if (theta_[j] == 0.0) continue;
    const ColumnView column = a_->column(static_cast<Index>(j));
    for (Index q = 0; q < column.size; ++q) {
      row_diagonal[static_cast<std::size_t>(column.rows[q])] +=
          theta_[j] * column.values[q] * column.values[q];
    }
  }
  d_.assign(m, delta);
  preconditioner_d_.assign(m, delta);
  std::vector<double> inverse_d(m);
  for (std::size_t i = 0; i < m; ++i) {
    if (row_shift.size() == m) d_[i] += row_shift[i];
    preconditioner_d_[i] = std::max(d_[i], kDiagonalFloor * row_diagonal[i]);
    inverse_d[i] = 1.0 / preconditioner_d_[i];
  }
  // N = Theta^-1 + A^T D^-1 A is the normal-equations product of A^T with weights D^-1 and
  // a row shift Theta^-1: the assembly the m side uses, on the transpose.
  return normal_equations_lower(transpose_, inverse_d, inverse_theta, 0.0, lower, should_stop);
}

namespace {

/// Whether the lower triangle of B B^T (diagonal included) holds more than `cap` entries,
/// B given twice: `outer` column i lists the columns of B in row i, `inner` column k lists
/// the rows of B in column k. Rows `skip` marks count their diagonal only. Stops as soon as
/// the count passes the cap; false when `should_stop` fires first.
bool lower_product_exceeds(const SparseMatrix& outer, const SparseMatrix& inner,
                           const std::vector<bool>* skip, std::int64_t cap,
                           const SparseLdl::ShouldStop& should_stop) {
  const Index size = outer.num_cols();
  std::vector<Index> mark(static_cast<std::size_t>(size), -1);
  std::int64_t count = 0;
  std::size_t work = 0;
  for (Index i = 0; i < size; ++i) {
    std::int64_t row_count = 1;  // the diagonal
    mark[static_cast<std::size_t>(i)] = i;
    if (skip == nullptr || !(*skip)[static_cast<std::size_t>(i)]) {
      const ColumnView row = outer.column(i);
      for (Index p = 0; p < row.size; ++p) {
        const ColumnView column = inner.column(row.rows[p]);
        for (Index q = 0; q < column.size; ++q) {
          const Index r = column.rows[q];
          if (r <= i || mark[static_cast<std::size_t>(r)] == i) continue;
          mark[static_cast<std::size_t>(r)] = i;
          ++row_count;
        }
        work += static_cast<std::size_t>(column.size);
        if (work >= (std::size_t{1} << 16)) {
          work = 0;
          if (should_stop && should_stop()) return false;
        }
      }
    }
    count += row_count;
    if (count > cap) return true;
  }
  return false;
}

}  // namespace

bool ColumnSide::row_side_exceeds(std::int64_t cap,
                                  const SparseLdl::ShouldStop& should_stop) const {
  // M = A A^T: row i's columns are column i of transpose_ (fixed columns already out), and
  // column j's rows are column j of A.
  return lower_product_exceeds(transpose_, *a_, nullptr, cap, should_stop);
}

bool ColumnSide::column_side_exceeds(std::int64_t cap,
                                     const SparseLdl::ShouldStop& should_stop) const {
  // N = A^T A: column j's rows are column j of A, and row i's columns are column i of
  // transpose_. A fixed column is an identity row of N.
  return lower_product_exceeds(*a_, transpose_, &fixed_, cap, should_stop);
}

void ColumnSide::multiply_m(const std::vector<double>& v, std::vector<double>* out) const {
  const auto n = static_cast<std::size_t>(a_->num_cols());
  std::vector<double> atv(n, 0.0);
  a_->transpose_multiply_add(v.data(), atv.data());
  for (std::size_t j = 0; j < n; ++j) atv[j] *= theta_[j];
  std::fill(out->begin(), out->end(), 0.0);
  a_->multiply_add(atv.data(), out->data());
  for (std::size_t i = 0; i < v.size(); ++i) (*out)[i] += d_[i] * v[i];
}

double ColumnSide::terms_of_m(const std::vector<double>& v) const {
  // || |A| Theta |A|^T |v| + D |v| ||_inf: the size of the terms M v is summed from.
  const auto n = static_cast<std::size_t>(a_->num_cols());
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
  std::vector<double> out(v.size(), 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    const ColumnView column = a_->column(static_cast<Index>(j));
    for (Index q = 0; q < column.size; ++q) {
      out[static_cast<std::size_t>(column.rows[q])] += std::fabs(column.values[q]) * atv[j];
    }
  }
  double largest = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    largest = std::max(largest, out[i] + d_[i] * std::fabs(v[i]));
  }
  return largest;
}

void ColumnSide::precondition(const SparseLdl& ldl, const std::vector<double>& r,
                              std::vector<double>* z) const {
  // z = D^-1 (r - A u), N u = A^T D^-1 r.
  const auto n = static_cast<std::size_t>(a_->num_cols());
  const std::size_t m = r.size();
  std::vector<double> scaled(m);
  for (std::size_t i = 0; i < m; ++i) scaled[i] = r[i] / preconditioner_d_[i];
  std::vector<double> u(n, 0.0);
  a_->transpose_multiply_add(scaled.data(), u.data());
  for (std::size_t j = 0; j < n; ++j) {
    if (fixed_[j]) u[j] = 0.0;
  }
  ldl.solve(u.data());
  for (std::size_t j = 0; j < n; ++j) {
    if (fixed_[j]) u[j] = 0.0;
  }
  std::vector<double> au(m, 0.0);
  a_->multiply_add(u.data(), au.data());
  z->assign(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) (*z)[i] = (r[i] - au[i]) / preconditioner_d_[i];
}

ColumnSideReport ColumnSide::solve(const SparseLdl& ldl, double* rhs) const {
  ColumnSideReport report;
  const auto m = static_cast<std::size_t>(a_->num_rows());
  const std::vector<double> b(rhs, rhs + m);
  if (inf_norm(b) == 0.0) {
    report.converged = true;
    return report;
  }
  std::vector<double> x;
  precondition(ldl, b, &x);
  const double scale = inf_norm(b) + terms_of_m(x);
  std::vector<double> r(m), q(m), z(m);
  multiply_m(x, &q);
  for (std::size_t i = 0; i < m; ++i) r[i] = b[i] - q[i];
  double residual = inf_norm(r);
  double checkpoint = residual;
  int since_checkpoint = 0;
  bool broke_down = false;
  precondition(ldl, r, &z);
  std::vector<double> p = z;
  double rz = dot(r, z);
  while (residual > kTargetBackwardError * scale && report.iterations < kMaxIterations) {
    multiply_m(p, &q);
    const double curvature = dot(p, q);
    if (!(curvature > 0.0) || !(rz > 0.0)) {
      broke_down = true;
      break;
    }
    const double alpha = rz / curvature;
    for (std::size_t i = 0; i < m; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * q[i];
    }
    ++report.iterations;
    residual = inf_norm(r);
    if (residual <= 0.5 * checkpoint) {
      checkpoint = residual;
      since_checkpoint = 0;
    } else if (++since_checkpoint >= kStagnation) {
      break;
    }
    precondition(ldl, r, &z);
    const double rz_next = dot(r, z);
    const double beta = rz_next / rz;
    rz = rz_next;
    for (std::size_t i = 0; i < m; ++i) p[i] = z[i] + beta * p[i];
  }
  multiply_m(x, &q);
  for (std::size_t i = 0; i < m; ++i) r[i] = b[i] - q[i];
  report.backward_error = inf_norm(r) / (inf_norm(b) + terms_of_m(x));
  report.converged = !broke_down && report.backward_error <= kAcceptedBackwardError;
  std::copy(x.begin(), x.end(), rhs);
  return report;
}

}  // namespace sankhya::ipm
