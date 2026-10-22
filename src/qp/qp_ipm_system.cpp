// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the equality-and-bounds form and the quasi-definite KKT matrix of the proximal
// interior point for convex QP (#490). See qp_ipm.cpp for the method and its references.

#include "qp_ipm_system.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "sankhya/tolerances.hpp"
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

/// Solve K x = rhs with the factors, then two steps of iterative refinement against the
/// regularized K itself: the factors belong to K up to rounding and any lifted pivot, and the
/// refinement recovers what that costs without changing the system being solved.
void solve_refined(const SparseLdl& ldl, const SparseMatrix& k, const std::vector<double>& rhs,
                   std::vector<double>* x) {
  *x = rhs;
  ldl.solve(x->data());
  std::vector<double> kx(rhs.size()), correction(rhs.size());
  for (int step = 0; step < 2; ++step) {
    KktMatrix::multiply(k, *x, &kx);
    for (std::size_t i = 0; i < rhs.size(); ++i) correction[i] = rhs[i] - kx[i];
    ldl.solve(correction.data());
    for (std::size_t i = 0; i < rhs.size(); ++i) (*x)[i] += correction[i];
  }
}

// Reference: Altman & Gondzio, "Regularized symmetric indefinite systems in interior point
// methods for linear and quadratic optimization", Optim. Methods Softw. 11 (1999): the
// primal-dual regularization makes the factorization safe, and iterative refinement against the
// unregularized system takes its perturbation back out, so the step solved is the Newton step
// of the problem and not of its proximal subproblem. Without it the second block row reads
// M dv + delta dy = b - M v, so a full step leaves a primal residual of delta * dy, and on a
// badly scaled model that error compounds until the iterate stops being finite: cvxqp3m and
// powell20 ended that way at 8d4b07b3 and reach the reference with the refinement, qship12l
// and qpilotno stay finite (measured, bench/results). The refinement does not cure the
// degenerate LISWET family (its correction gains under 10% a step once the active rows lose
// rank, so the loop stops where it started); that stays an open class.
void refine_unregularized(const SparseLdl& ldl, const SparseMatrix& k0,
                          const std::vector<double>& rhs, std::vector<double>* x) {
  const std::size_t n = rhs.size();
  // Each row's residual is judged against its own diagonal, max(1, |K0_ii|): a column whose
  // Theta^{-1} is 1e+8 carries a residual of 1e-8 |dv| from rounding alone, and measured
  // unscaled that row would end the refinement while the row block (diagonal 0, the primal
  // residual it exists to close) is still improving.
  std::vector<double> scale(n, 1.0);
  for (Index j = 0; j < k0.num_cols(); ++j) {
    const ColumnView column = k0.column(j);
    for (Index p = 0; p < column.size; ++p) {
      if (column.rows[p] == j) {
        scale[static_cast<std::size_t>(j)] = std::max(1.0, std::fabs(column.values[p]));
      }
    }
  }
  std::vector<double> kx(n), r(n), trial(n), trial_r(n);
  const auto residual = [&](const std::vector<double>& v, std::vector<double>* out) {
    KktMatrix::multiply(k0, v, &kx);
    double norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      (*out)[i] = rhs[i] - kx[i];
      norm = std::max(norm, std::fabs((*out)[i]) / scale[i]);
    }
    return norm;
  };
  double best = residual(*x, &r);
  const double floor = std::numeric_limits<double>::epsilon() * std::max(1.0, inf_norm(rhs));
  for (int step = 0; step < tol::kQpIpmRefinementSteps && best > floor; ++step) {
    trial = r;
    ldl.solve(trial.data());
    for (std::size_t i = 0; i < n; ++i) trial[i] += (*x)[i];
    const double next = residual(trial, &trial_r);
    if (!(next < best)) break;  // no better, or not finite: keep what is in hand
    std::swap(*x, trial);
    std::swap(r, trial_r);
    const bool stalled = !(next < tol::kQpIpmRefinementProgress * best);
    best = next;
    if (stalled) break;
  }
}

// Reference: Saad & Schultz, "GMRES: a generalized minimal residual algorithm for solving
// nonsymmetric linear systems", SIAM J. Sci. Stat. Comput. 7 (1986), right preconditioned
// (Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., 2003, sec. 9.3.2); used as
// refinement with the factors of a nearby matrix as the preconditioner after Arioli, Duff &
// Ruiz, SIAM J. Matrix Anal. Appl. 13 (1992) and Carson & Higham, SIAM J. Sci. Comput. 39
// (2017). On LISWET (#981) the stationary refinement above stops gaining once delta |dy|
// reaches |b - M v|, and the Newton step then leaves almost all of the primal residual
// in place; GMRES is not held to a contraction factor below one.
void refine_krylov(const SparseLdl& ldl, const SparseMatrix& k0, const std::vector<double>& rhs,
                   std::vector<double>* x) {
  const std::size_t n = rhs.size();
  const auto m = static_cast<std::size_t>(tol::kQpIpmKrylovDimension);
  std::vector<double> kx(n);
  const auto residual = [&](const std::vector<double>& at, std::vector<double>* out) {
    KktMatrix::multiply(k0, at, &kx);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      (*out)[i] = rhs[i] - kx[i];
      sum += (*out)[i] * (*out)[i];
    }
    return std::sqrt(sum);
  };
  std::vector<double> r(n);
  const double beta = residual(*x, &r);
  if (!(beta > 0.0) || !std::isfinite(beta)) return;
  std::vector<std::vector<double>> basis(m + 1, std::vector<double>(n));
  std::vector<std::vector<double>> hess(m + 1, std::vector<double>(m, 0.0));
  std::vector<double> cs(m), sn(m), g(m + 1, 0.0), w(n);
  for (std::size_t i = 0; i < n; ++i) basis[0][i] = r[i] / beta;
  g[0] = beta;
  std::size_t built = 0;
  for (std::size_t j = 0; j < m; ++j) {
    w = basis[j];
    ldl.solve(w.data());
    KktMatrix::multiply(k0, w, &kx);
    w = kx;
    for (std::size_t i = 0; i <= j; ++i) {  // modified Gram-Schmidt
      double dot = 0.0;
      for (std::size_t q = 0; q < n; ++q) dot += w[q] * basis[i][q];
      hess[i][j] = dot;
      for (std::size_t q = 0; q < n; ++q) w[q] -= dot * basis[i][q];
    }
    double norm = 0.0;
    for (std::size_t q = 0; q < n; ++q) norm += w[q] * w[q];
    norm = std::sqrt(norm);
    hess[j + 1][j] = norm;
    for (std::size_t i = 0; i < j; ++i) {  // the earlier Givens rotations
      const double a = hess[i][j], b = hess[i + 1][j];
      hess[i][j] = cs[i] * a + sn[i] * b;
      hess[i + 1][j] = -sn[i] * a + cs[i] * b;
    }
    const double radius = std::hypot(hess[j][j], hess[j + 1][j]);
    if (!(radius > 0.0) || !std::isfinite(radius)) break;
    cs[j] = hess[j][j] / radius;
    sn[j] = hess[j + 1][j] / radius;
    hess[j][j] = radius;
    hess[j + 1][j] = 0.0;
    g[j + 1] = -sn[j] * g[j];
    g[j] = cs[j] * g[j];
    built = j + 1;
    if (std::fabs(g[j + 1]) <= std::numeric_limits<double>::epsilon() * beta || !(norm > 0.0)) {
      break;
    }
    for (std::size_t q = 0; q < n; ++q) basis[j + 1][q] = w[q] / norm;
  }
  if (built == 0) return;
  std::vector<double> coef(built);
  for (std::size_t i = built; i-- > 0;) {
    double sum = g[i];
    for (std::size_t q = i + 1; q < built; ++q) sum -= hess[i][q] * coef[q];
    coef[i] = sum / hess[i][i];
  }
  std::vector<double> z(n, 0.0);
  for (std::size_t i = 0; i < built; ++i) {
    for (std::size_t q = 0; q < n; ++q) z[q] += coef[i] * basis[i][q];
  }
  ldl.solve(z.data());
  std::vector<double> trial(*x);
  for (std::size_t q = 0; q < n; ++q) trial[q] += z[q];
  std::vector<double> trial_r(n);
  const double next = residual(trial, &trial_r);
  if (next < beta) std::swap(*x, trial);
}

}  // namespace sankhya::qp::ipm_detail
