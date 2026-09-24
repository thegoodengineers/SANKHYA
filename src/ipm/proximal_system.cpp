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

/// A correction is kept whenever it lowers the residual; the refinement stops after one that
/// gains less than a tenth, which has stalled at what the regularization lets it reach.
constexpr double kRefinementProgress = 0.9;

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

double ProximalSystem::next_regularization(double previous, double mu, double floor) {
  const double target = std::isfinite(mu) ? tol::kIpmProximalShare * mu : previous;
  return std::max(floor, std::min(previous, target));
}

bool ProximalSystem::factorize(SparseLdl& ldl, const std::vector<double>& theta_inverse,
                               double* reg, const SparseLdl::ShouldStop& should_stop,
                               int* factorizations) {
  *factorizations = 0;
  for (int attempt = 0; attempt < tol::kIpmProximalAttempts; ++attempt) {
    if (attempt > 0) *reg *= tol::kQpIpmRegularizationRaise;
    assemble(theta_inverse, *reg, *reg);
    if (!ldl.factorize_quasidefinite(k_, signs_, tol::kQpIpmPivotShare * *reg, should_stop)) {
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

ProximalSystem::Refinement ProximalSystem::solve(const SparseLdl& ldl,
                                                 const std::vector<double>& g,
                                                 const std::vector<double>& r_b, int max_steps,
                                                 std::vector<double>* dx,
                                                 std::vector<double>* dy) const {
  Refinement report;
  apply_inverse(ldl, g, r_b, dx, dy);
  std::vector<double> p(g.size()), q(r_b.size());
  double best = residual(g, r_b, *dx, *dy, &p, &q);
  report.first_residual = best;
  report.final_residual = best;
  if (!std::isfinite(best)) return report;
  const double floor =
      std::numeric_limits<double>::epsilon() * std::max({1.0, inf_norm(g), inf_norm(r_b)});
  std::vector<double> cx(dx->size()), cy(dy->size());
  std::vector<double> tx(dx->size()), ty(dy->size());
  for (int step = 0; step < max_steps && best > floor; ++step) {
    apply_inverse(ldl, p, q, &cx, &cy);
    for (std::size_t k = 0; k < tx.size(); ++k) tx[k] = (*dx)[k] + cx[k];
    for (std::size_t i = 0; i < ty.size(); ++i) ty[i] = (*dy)[i] + cy[i];
    std::vector<double> tp(p.size()), tq(q.size());
    const double next = residual(g, r_b, tx, ty, &tp, &tq);
    if (!(next < best)) break;  // no better, or not finite: keep what is in hand
    std::swap(*dx, tx);
    std::swap(*dy, ty);
    std::swap(p, tp);
    std::swap(q, tq);
    const bool stalled = !(next < kRefinementProgress * best);
    best = next;
    ++report.steps;
    if (stalled) break;
  }
  report.final_residual = best;
  return report;
}

}  // namespace sankhya::ipm
