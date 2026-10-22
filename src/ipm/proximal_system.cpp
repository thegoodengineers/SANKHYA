// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the proximal primal-dual regularized Newton system of the LP interior point (#473).
// See proximal_system.hpp for the method and its references.

#include "ipm/proximal_system.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "sankhya/tolerances.hpp"

namespace sankhya::ipm {
namespace {

double inf_norm(const std::vector<double>& v) {
  double norm = 0.0;
  for (const double x : v) norm = std::max(norm, std::fabs(x));
  return norm;
}

}  // namespace

ProximalSystem::ProximalSystem(const SparseMatrix& a, const std::vector<bool>& fixed)
    : a_(a), fixed_(fixed), n_(a.num_cols()), m_(a.num_rows()) {
  const Index dim = n_ + m_;
  starts_.assign(static_cast<std::size_t>(dim) + 1, 0);
  signs_.assign(static_cast<std::size_t>(dim), 1);
  for (Index j = 0; j < dim; ++j) {
    starts_[static_cast<std::size_t>(j)] = static_cast<Index>(rows_.size());
    rows_.push_back(j);
    if (j >= n_) continue;
    signs_[static_cast<std::size_t>(j)] = -1;
    if (fixed_[static_cast<std::size_t>(j)]) continue;  // dx_j = 0: the column drops out
    const ColumnView column = a_.column(j);
    for (Index p = 0; p < column.size; ++p) rows_.push_back(n_ + column.rows[p]);
  }
  starts_[static_cast<std::size_t>(dim)] = static_cast<Index>(rows_.size());
  theta_s_.assign(static_cast<std::size_t>(m_), 0.0);
}

void ProximalSystem::assemble(const std::vector<double>& theta_inverse, double rho,
                              double delta) {
  theta_inverse_ = theta_inverse;
  const Index dim = n_ + m_;
  std::vector<double> values;
  values.reserve(rows_.size());
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (fixed_[u]) {
      values.push_back(-1.0);  // a placeholder pivot of the right sign; its row is empty
      continue;
    }
    values.push_back(-(theta_inverse[u] + rho));
    const ColumnView column = a_.column(j);
    for (Index p = 0; p < column.size; ++p) values.push_back(column.values[p]);
  }
  for (Index i = 0; i < m_; ++i) {
    const auto k = static_cast<std::size_t>(n_ + i);
    const double theta_s = fixed_[k] ? 0.0 : 1.0 / (theta_inverse[k] + rho);
    theta_s_[static_cast<std::size_t>(i)] = theta_s;
    values.push_back(theta_s + delta);
  }
  k_.assign_columns(dim, dim, starts_, rows_, std::move(values));
}

double ProximalSystem::next_regularization(double previous, double mu, double floor,
                                           bool refinement_missed) {
  if (refinement_missed)
    previous = std::max(floor, previous * tol::kIpmProximalRefinementShrink);
  const double target = std::isfinite(mu) ? tol::kIpmProximalShare * mu : previous;
  return std::max(floor, std::min(previous, target));
}

double ProximalSystem::recovery_floor(Count raises) {
  double floor = tol::kIpmProximalFloor;
  for (Count r = 0; r < raises && floor < tol::kIpmProximalRecoveryCap; ++r) {
    floor *= tol::kIpmProximalRecoveryRaise;
  }
  return std::min(floor, tol::kIpmProximalRecoveryCap);
}

bool ProximalSystem::factorize(SparseLdl& ldl, const std::vector<double>& theta_inverse,
                               double* reg, const SparseLdl::ShouldStop& should_stop,
                               int* factorizations) {
  *factorizations = 0;
  for (int attempt = 0; attempt < tol::kIpmProximalAttempts; ++attempt) {
    if (attempt > 0) *reg *= tol::kIpmProximalRaise;
    assemble(theta_inverse, *reg, *reg);
    if (!ldl.factorize_quasidefinite(k_, signs_, tol::kIpmProximalPivotShare * *reg,
                                     should_stop)) {
      return false;
    }
    ++*factorizations;
    if (ldl.regularized_pivots() == 0) break;
  }
  return true;
}

void ProximalSystem::apply_inverse(const SparseLdl& ldl, const std::vector<double>& p,
                                   const std::vector<double>& q, std::vector<double>* dx,
                                   std::vector<double>* dy) const {
  // The logical rows eliminated as in the header: z = (p_x, q - theta_s p_s), K z solved,
  // then dx_s = -theta_s (dy + p_s).
  work_.assign(static_cast<std::size_t>(n_ + m_), 0.0);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    work_[u] = fixed_[u] ? 0.0 : p[u];
  }
  for (Index i = 0; i < m_; ++i) {
    const auto r = static_cast<std::size_t>(i);
    const auto k = static_cast<std::size_t>(n_ + i);
    work_[k] = q[r] - theta_s_[r] * p[k];
  }
  ldl.solve(work_.data());
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    (*dx)[u] = fixed_[u] ? 0.0 : work_[u];
  }
  for (Index i = 0; i < m_; ++i) {
    const auto r = static_cast<std::size_t>(i);
    const auto k = static_cast<std::size_t>(n_ + i);
    (*dy)[r] = work_[k];
    (*dx)[k] = fixed_[k] ? 0.0 : -theta_s_[r] * ((*dy)[r] + p[k]);
  }
}

double ProximalSystem::residual(const std::vector<double>& g, const std::vector<double>& r_b,
                                const std::vector<double>& dx, const std::vector<double>& dy,
                                std::vector<double>* p, std::vector<double>* q) const {
  // K_0 (dx, dy) = (-Theta^-1 dx + Abar^T dy, Abar dx), Abar = [A | -I].
  const auto total = static_cast<std::size_t>(n_ + m_);
  std::fill(p->begin(), p->end(), 0.0);
  if (m_ > 0) a_.transpose_multiply_add(dy.data(), p->data());
  for (Index i = 0; i < m_; ++i) {
    (*p)[static_cast<std::size_t>(n_ + i)] = -dy[static_cast<std::size_t>(i)];
  }
  for (std::size_t k = 0; k < total; ++k) {
    (*p)[k] = fixed_[k] ? 0.0 : g[k] - ((*p)[k] - theta_inverse_[k] * dx[k]);
  }
  std::fill(q->begin(), q->end(), 0.0);
  if (m_ > 0) a_.multiply_add(dx.data(), q->data());
  for (Index i = 0; i < m_; ++i) {
    const auto r = static_cast<std::size_t>(i);
    (*q)[r] = r_b[r] - ((*q)[r] - dx[static_cast<std::size_t>(n_) + r]);
  }
  return std::max(inf_norm(*p), inf_norm(*q));
}

ProximalSystem::Refinement ProximalSystem::solve(
    const SparseLdl& ldl, const std::vector<double>& g, const std::vector<double>& r_b,
    int max_steps, std::vector<double>* dx, std::vector<double>* dy, int krylov_steps) const {
  Refinement report;
  apply_inverse(ldl, g, r_b, dx, dy);
  std::vector<double> p(g.size()), q(r_b.size());
  double best = residual(g, r_b, *dx, *dy, &p, &q);
  report.first_residual = best;
  report.final_residual = best;
  report.scale = std::max({1.0, inf_norm(g), inf_norm(r_b)});
  if (!std::isfinite(best)) return report;
  const double floor = std::numeric_limits<double>::epsilon() * report.scale;
  std::vector<double> cx(dx->size()), cy(dy->size());
  std::vector<double> tx(dx->size()), ty(dy->size());
  std::vector<double> tp(p.size()), tq(q.size());
  for (int step = 0; step < max_steps && best > floor; ++step) {
    apply_inverse(ldl, p, q, &cx, &cy);
    for (std::size_t k = 0; k < tx.size(); ++k) tx[k] = (*dx)[k] + cx[k];
    for (std::size_t i = 0; i < ty.size(); ++i) ty[i] = (*dy)[i] + cy[i];
    const double next = residual(g, r_b, tx, ty, &tp, &tq);
    if (!(next < best)) break;  // no better, or not finite: keep what is in hand
    std::swap(*dx, tx);
    std::swap(*dy, ty);
    std::swap(p, tp);
    std::swap(q, tq);
    const bool stalled = !(next < tol::kIpmProximalRefinementProgress * best);
    best = next;
    ++report.steps;
    if (stalled) break;
  }
  if (best > tol::kIpmProximalRefinementTarget * report.scale && krylov_steps > 0) {
    best = gmres(ldl, g, r_b, krylov_steps, best, dx, dy, &p, &q, &report.krylov_steps);
  }
  report.final_residual = best;
  return report;
}

// GMRES, RIGHT-PRECONDITIONED BY K_reg^-1 (Saad & Schultz, "GMRES: a generalized minimal
// residual algorithm for solving nonsymmetric linear systems", SIAM J. Sci. Stat. Comput. 7
// (1986); Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed., SIAM (2003), 9.3.2).
// The stationary refinement above contracts by the spectral radius of I - K_reg^-1 K_0, which
// approaches 1 on every component where Theta^-1 << rho - the basic variables late in a run -
// so it stalls there. GMRES minimises the residual over the Krylov space of the same operator
// instead, and needs no contraction. One iteration costs what one refinement step does: one
// solve with the factors and one product with K_0. (p, q) holds the residual of (dx, dy) on
// entry; on return (dx, dy) is the best iterate seen and its residual norm is returned.
double ProximalSystem::gmres(const SparseLdl& ldl, const std::vector<double>& g,
                             const std::vector<double>& r_b, int steps, double best,
                             std::vector<double>* dx, std::vector<double>* dy,
                             std::vector<double>* p, std::vector<double>* q, int* taken) const {
  const std::size_t nx = dx->size();
  const std::size_t ny = dy->size();
  const std::size_t dim = nx + ny;
  const auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t k = 0; k < dim; ++k) s += a[k] * b[k];
    return s;
  };
  const auto k = static_cast<std::size_t>(steps);
  std::vector<std::vector<double>> v(k + 1, std::vector<double>(dim, 0.0));
  std::vector<std::vector<double>> z(k, std::vector<double>(dim, 0.0));
  std::vector<std::vector<double>> h(k + 1, std::vector<double>(k, 0.0));
  std::vector<double> cs(k, 0.0), sn(k, 0.0), e(k + 1, 0.0);
  for (std::size_t i = 0; i < nx; ++i) v[0][i] = (*p)[i];
  for (std::size_t i = 0; i < ny; ++i) v[0][nx + i] = (*q)[i];
  const double beta = std::sqrt(dot(v[0], v[0]));
  if (!(beta > 0.0) || !std::isfinite(beta)) return best;
  for (double& x : v[0]) x /= beta;
  e[0] = beta;
  std::vector<double> zx(nx), zy(ny), wx(nx), wy(ny), zero_x(nx, 0.0), zero_y(ny, 0.0);
  std::vector<double> sx(nx), sy(ny), tp(nx), tq(ny);
  std::size_t used = 0;
  for (std::size_t j = 0; j < k; ++j) {
    // z_j = K_reg^-1 v_j; w = K_0 z_j (residual() of a zero right-hand side is -K_0 z_j).
    std::vector<double> vx(v[j].begin(), v[j].begin() + static_cast<std::ptrdiff_t>(nx));
    std::vector<double> vy(v[j].begin() + static_cast<std::ptrdiff_t>(nx), v[j].end());
    apply_inverse(ldl, vx, vy, &zx, &zy);
    (void)residual(zero_x, zero_y, zx, zy, &wx, &wy);
    std::vector<double>& w = v[j + 1];
    for (std::size_t i = 0; i < nx; ++i) w[i] = -wx[i];
    for (std::size_t i = 0; i < ny; ++i) w[nx + i] = -wy[i];
    for (std::size_t i = 0; i < nx; ++i) z[j][i] = zx[i];
    for (std::size_t i = 0; i < ny; ++i) z[j][nx + i] = zy[i];
    for (std::size_t i = 0; i <= j; ++i) {  // modified Gram-Schmidt
      h[i][j] = dot(w, v[i]);
      for (std::size_t t = 0; t < dim; ++t) w[t] -= h[i][j] * v[i][t];
    }
    h[j + 1][j] = std::sqrt(dot(w, w));
    for (std::size_t i = 0; i < j; ++i) {  // the earlier Givens rotations
      const double a = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
      h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
      h[i][j] = a;
    }
    const double r = std::hypot(h[j][j], h[j + 1][j]);
    if (!(r > 0.0) || !std::isfinite(r)) break;
    cs[j] = h[j][j] / r;
    sn[j] = h[j + 1][j] / r;
    h[j][j] = r;
    h[j + 1][j] = 0.0;
    e[j + 1] = -sn[j] * e[j];
    e[j] = cs[j] * e[j];
    used = j + 1;
    const double next_h = std::sqrt(dot(w, w));
    if (!(next_h > 0.0)) break;  // the Krylov space is exhausted: the solution is exact
    for (double& x : w) x /= next_h;
  }
  if (used == 0) return best;
  // y = R^-1 e over the first `used` columns; correction = sum_i y_i z_i.
  std::vector<double> y(used, 0.0);
  for (std::size_t i = used; i-- > 0;) {
    double s = e[i];
    for (std::size_t t = i + 1; t < used; ++t) s -= h[i][t] * y[t];
    y[i] = s / h[i][i];
  }
  for (std::size_t i = 0; i < nx; ++i) sx[i] = (*dx)[i];
  for (std::size_t i = 0; i < ny; ++i) sy[i] = (*dy)[i];
  for (std::size_t c = 0; c < used; ++c) {
    for (std::size_t i = 0; i < nx; ++i) sx[i] += y[c] * z[c][i];
    for (std::size_t i = 0; i < ny; ++i) sy[i] += y[c] * z[c][nx + i];
  }
  // Measured in the infinity norm the target is on; kept only if it is better, and only if
  // the correction is no larger than the direction it corrects: on a nearly singular K_0 a
  // lower residual can come from a large step along a near-null vector, which is not a
  // better direction (Netlib share2b stalled at 193 iterations without this guard).
  double size = 0.0;
  double moved = 0.0;
  for (std::size_t i = 0; i < nx; ++i) {
    size = std::max(size, std::fabs((*dx)[i]));
    moved = std::max(moved, std::fabs(sx[i] - (*dx)[i]));
  }
  for (std::size_t i = 0; i < ny; ++i) {
    size = std::max(size, std::fabs((*dy)[i]));
    moved = std::max(moved, std::fabs(sy[i] - (*dy)[i]));
  }
  if (!(moved <= size)) return best;
  const double next = residual(g, r_b, sx, sy, &tp, &tq);
  if (!(next < best)) return best;
  std::swap(*dx, sx);
  std::swap(*dy, sy);
  std::swap(*p, tp);
  std::swap(*q, tq);
  *taken = static_cast<int>(used);
  return next;
}

}  // namespace sankhya::ipm
