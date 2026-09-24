// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's normal equations solved on the n x n side (#469).
// See column_side.hpp for the identity and the references.

#include "ipm/column_side.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "sankhya/tolerances.hpp"

namespace sankhya::ipm {
namespace {

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
  // built with D floored at tol::kIpmColumnSideDiagonalFloor times the row's own diagonal
  // of A Theta A^T, which changes it by that relative amount on those rows and nowhere
  // else; the
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
    preconditioner_d_[i] = std::max(d_[i], tol::kIpmColumnSideDiagonalFloor * row_diagonal[i]);
    inverse_d[i] = 1.0 / preconditioner_d_[i];
  }
  // N = Theta^-1 + A^T D^-1 A is the normal-equations product of A^T with weights D^-1 and
  // a row shift Theta^-1: the assembly the m side uses, on the transpose.
  return normal_equations_lower(transpose_, inverse_d, inverse_theta, 0.0, lower, should_stop);
}

NormalPrediction ColumnSide::predict_row_side(std::int64_t cap,
                                              const SparseLdl::ShouldStop& stop) const {
  std::vector<char> skip(fixed_.size(), 0);
  for (std::size_t j = 0; j < fixed_.size(); ++j) skip[j] = fixed_[j] ? 1 : 0;
  return predict_normal_nonzeros(*a_, skip, cap, stop);
}

NormalPrediction ColumnSide::predict_column_side(std::int64_t cap,
                                                 const SparseLdl::ShouldStop& stop) const {
  // transpose_ is n x m and its column i is row i of A over the unfixed columns, so its
  // product with its own transpose has the pattern of A^T A; every one of the n diagonals
  // is counted, a fixed column's included, which is the identity row it is in N.
  return predict_normal_nonzeros(transpose_, {}, cap, stop);
}

SideChoice ColumnSide::choose(bool compare, std::int64_t factor_budget,
                              std::size_t ordering_budget, const SparseLdl::ShouldStop& stop) {
  const Index m = a_->num_rows();
  const Index n = a_->num_cols();
  SideChoice out;
  const auto decide = [&out](NormalSide side, std::string reason) {
    out.side = side;
    out.reason = std::move(reason);
    return out;
  };
  const auto stopped = [&stop] { return stop && stop(); };
  const std::string halted = "stopped before the choice was made";
  if (stopped()) return decide(NormalSide::kStopped, halted);

  // THE RESCUE: rows that do not outnumber the columns keep the row side unless it is
  // already known to be over the budget.
  bool rows_over_budget = false;
  if (!compare) {
    if (factor_budget < 0) {
      return decide(NormalSide::kRows,
                    "the rows do not outnumber the columns and there is no factor budget; "
                    "the row side is kept");
    }
    const NormalPrediction rows = predict_row_side(factor_budget, stop);
    if (rows.stopped) return decide(NormalSide::kStopped, halted);
    if (!rows.over_cap) {
      return decide(NormalSide::kRows,
                    fmt::format("the rows do not outnumber the columns and the row side's "
                                "normal equations fit the factor budget ({} {} nonzeros); the "
                                "row side is kept",
                                rows.exact ? "exactly" : "at most", rows.nonzeros));
    }
    rows_over_budget = true;
  }

  // THE COLUMN SIDE, counted against the budget before it is built.
  if (factor_budget >= 0) {
    if (stopped()) return decide(NormalSide::kStopped, halted);
    const NormalPrediction columns = predict_column_side(factor_budget, stop);
    if (columns.stopped) return decide(NormalSide::kStopped, halted);
    if (columns.over_cap) {
      return decide(NormalSide::kRows,
                    fmt::format("the column side's N alone holds at least {} nonzeros, over "
                                "the factor budget {}; the row side is kept",
                                columns.nonzeros, factor_budget));
    }
  }
  if (stopped()) return decide(NormalSide::kStopped, halted);
  std::vector<double> ones_x(static_cast<std::size_t>(n), 1.0);
  for (std::size_t j = 0; j < ones_x.size(); ++j) {
    if (fixed_[j]) ones_x[j] = 0.0;
  }
  const std::vector<double> ones_m(static_cast<std::size_t>(m), 1.0);
  SparseMatrix pattern;
  if (!assemble(ones_x, ones_m, 0.0, &pattern, stop)) {
    return decide(NormalSide::kStopped, halted);
  }
  SparseLdl columns;
  columns.set_ordering_budget(ordering_budget);
  columns.set_factor_budget(factor_budget);
  if (!columns.analyze(pattern, stop)) {
    if (columns.stopped_early()) return decide(NormalSide::kStopped, halted);
    return decide(NormalSide::kRows,
                  "the column side's factor is over the factor budget or its ordering over "
                  "the ordering budget; the row side is kept");
  }
  const auto column_factor =
      static_cast<std::int64_t>(columns.factor_nonzeros()) + columns.dimension();
  if (factor_budget >= 0 && column_factor > factor_budget) {
    return decide(NormalSide::kRows,
                  fmt::format("the column side's factor ({} nonzeros) is over the factor "
                              "budget {}; the row side is kept",
                              column_factor, factor_budget));
  }
  if (rows_over_budget) {
    return decide(NormalSide::kColumns,
                  fmt::format("the row side's normal equations are over the factor budget {} "
                              "before assembly and the column side's factor fits ({} "
                              "nonzeros, {} x {}); the column side is taken",
                              factor_budget, column_factor, n, n));
  }

  // THE COMPARISON. M's lower triangle is inside its factor, so a count that passes the
  // column side's factor decides it without M ever being built.
  if (stopped()) return decide(NormalSide::kStopped, halted);
  const NormalPrediction rows = predict_row_side(column_factor, stop);
  if (rows.stopped) return decide(NormalSide::kStopped, halted);
  if (rows.over_cap) {
    return decide(NormalSide::kColumns,
                  fmt::format("the row side's normal equations alone hold at least {} "
                              "nonzeros, more than the column side's factor ({} nonzeros, {} x "
                              "{}); the column side is taken",
                              rows.nonzeros, column_factor, n, n));
  }
  if (stopped()) return decide(NormalSide::kStopped, halted);
  if (!normal_equations_lower(*a_, ones_x, ones_m, 0.0, &pattern, stop)) {
    return decide(NormalSide::kStopped, halted);
  }
  SparseLdl row_ldl;
  row_ldl.set_ordering_budget(ordering_budget);
  row_ldl.set_factor_budget(column_factor);
  if (!row_ldl.analyze(pattern, stop)) {
    if (row_ldl.stopped_early()) return decide(NormalSide::kStopped, halted);
    if (row_ldl.factor_too_large() || row_ldl.ordering_too_large()) {
      return decide(NormalSide::kColumns,
                    fmt::format("the row side's factor passes the column side's ({} "
                                "nonzeros, {} x {}), or its ordering the ordering budget; the "
                                "column side is taken",
                                column_factor, n, n));
    }
    return decide(NormalSide::kRows,
                  "the row side's pattern could not be analysed; the row side is kept");
  }
  const auto row_factor =
      static_cast<std::int64_t>(row_ldl.factor_nonzeros()) + row_ldl.dimension();
  const std::string sizes = fmt::format(
      "normal equations on the row side ({} x {}): factor {}; on the column side "
      "({} x {}): factor {}",
      m, m, row_factor, n, n, column_factor);
  return column_factor < row_factor
             ? decide(NormalSide::kColumns, sizes + "; the column side is taken")
             : decide(NormalSide::kRows, sizes + "; the row side is kept");
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
  while (residual > tol::kIpmPcgTargetBackwardError * scale &&
         report.iterations < tol::kIpmPcgMaxIterations) {
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
    } else if (++since_checkpoint >= tol::kIpmColumnSideStagnationSteps) {
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
  report.converged = !broke_down && report.backward_error <= tol::kIpmPcgAcceptedBackwardError;
  std::copy(x.begin(), x.end(), rhs);
  return report;
}

}  // namespace sankhya::ipm
