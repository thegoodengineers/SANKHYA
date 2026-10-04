// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the homogeneous self-dual embedding of the LP interior point (#475, ipm_hsd).
//
// References:
//   Ye, Todd & Mizuno, "An O(sqrt(n)L)-iteration homogeneous and self-dual linear programming
//     algorithm", Mathematics of Operations Research 19(1) (1994) - the embedding.
//   Xu, Hung & Ye, "A simplified homogeneous and self-dual linear programming algorithm and
//     its implementation", Annals of Operations Research 62 (1996) - the simplified form
//     implemented here (no extra artificial row and column), and what tau and kappa say at
//     the limit.
//   Andersen & Andersen, "The MOSEK interior point optimizer for linear programming: an
//     implementation of the homogeneous algorithm", in High Performance Optimization, Kluwer
//     (2000) - the embedding with bounds, the Newton system by two solves with one
//     factorization of the normal equations, Mehrotra's predictor-corrector on it.
//   Mehrotra (1992) and Wright, Primal-Dual Interior-Point Methods (1997), as in ipm.cpp.
//
// THE EMBEDDING. The form is ipm.cpp's: Abar = [A | -I] over the n structurals and m
// logicals, Abar x = 0, l <= x <= u. A fixed variable is the constant x = l tau. With
// b = -Abar_fixed l_fixed, a slack s_l = x - l tau >= 0 per finite lower bound and
// s_u = u tau - x >= 0 per finite upper bound, multipliers z_l, z_u >= 0, the embedding is
//
//     Abar x                                   = 0     (fixed columns carry b tau)
//     x - s_l - l tau                          = 0     (each finite lower bound)
//     x + s_u - u tau                          = 0     (each finite upper bound)
//     Abar^T y + z_l - z_u - c tau             = 0     (each unfixed variable)
//     b'y + l'z_l - u'z_u - c'x - kappa        = 0
//
// with s, z, tau, kappa >= 0. Its linear part is skew-symmetric, so a Newton step of length
// alpha with target (1 - eta) shrinks every residual and, to first order, mu = (s'z + tau
// kappa) / (bounds + 1) by the same factor 1 - alpha eta (Xu, Hung & Ye, Theorem 1). At the
// limit tau kappa = 0: tau > 0 makes (x, y, z) / tau an optimum; kappa > 0 makes
// b'y + l'z_l - u'z_u > 0 with Abar^T y + z_l - z_u = 0, which is a Farkas certificate of
// primal infeasibility in y, or c'x < 0 with Abar x = 0 inside the recession cone of the
// bounds, which is a primal ray (or both).
//
// THE NEWTON SYSTEM. With D = z_l/s_l + z_u/s_u + rho and Theta = D^-1, eliminating the
// slacks and multipliers leaves the normal equations of ipm.cpp, M = Abar Theta Abar^T, for
// dy at a FIXED dtau; dtau is the one extra unknown, and the direction is affine in it:
// (dx, dy) = (dx1, dy1) + dtau (dx2, dy2), where the second pair solves the same M against
// b - Abar Theta (p - c), p = (z_l/s_l) l + (z_u/s_u) u, and does not depend on the target.
// The gap row and the tau-kappa product then give dtau from one scalar equation (Andersen &
// Andersen, sec. 3). That is a rank-one Schur complement on the factor of M, which is the
// same factor, the same assembly (la/ldl.hpp normal_equations_lower) and the same SparseLdl
// the default path uses; nothing here factors anything of its own.
//
// THE CERTIFICATES. When kappa > tau the iterate leans to the ray side. y is then offered to
// farkas_proves_infeasible() and x (structurals) to ray_proves_unbounded(), each against the
// model this method iterates on, and reported only when the checker accepts it. Unbounded
// also needs a feasible point to start the ray from: the embedding is solved a second time
// with c = 0, whose limit is that point (tau > 0) or a Farkas certificate (kappa > 0). The
// rays are cleaned before they are offered (hsd_certificate.cpp), and when the model is a
// scaled copy each one must also hold on the caller's model, mapped back.
//
// WHAT IT DOES NOT HAVE is the default loop's recovery machinery near an optimum. When the
// embedding ends with no verdict of its own, solve_scaled() (ipm.cpp) hands the model to the
// default loop on the time that is left; see default_loop_after_the_embedding().

#include "ipm/hsd_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "ipm/model_space.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::ipm::hsd {

void Homogeneous::constraint_times(const std::vector<double>& v,
                                   std::vector<double>* out) const {
  std::fill(out->begin(), out->end(), 0.0);
  if (m_ == 0) return;
  model_.matrix.multiply_add(v.data(), out->data());
  for (Index i = 0; i < m_; ++i) {
    (*out)[static_cast<std::size_t>(i)] -= v[static_cast<std::size_t>(n_ + i)];
  }
}

void Homogeneous::constraint_transpose_times(const std::vector<double>& w,
                                             std::vector<double>* out) const {
  std::fill(out->begin(), out->end(), 0.0);
  if (m_ == 0) return;
  model_.matrix.transpose_multiply_add(w.data(), out->data());
  for (Index i = 0; i < m_; ++i) {
    (*out)[static_cast<std::size_t>(n_ + i)] = -w[static_cast<std::size_t>(i)];
  }
}

void Homogeneous::build() {
  n_ = model_.num_cols();
  m_ = model_.num_rows();
  total_ = n_ + m_;
  const auto T = static_cast<std::size_t>(total_);
  const double sense = model_.sense_multiplier();
  cost_.assign(T, 0.0);
  lower_.resize(T);
  upper_.resize(T);
  has_lower_.assign(T, 0);
  has_upper_.assign(T, 0);
  fixed_.assign(T, 0);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (!feasibility_only_) cost_[u] = sense * model_.col_cost[u];
    lower_[u] = model_.col_lower[u];
    upper_[u] = model_.col_upper[u];
  }
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(n_ + i);
    lower_[u] = model_.row_lower[static_cast<std::size_t>(i)];
    upper_[u] = model_.row_upper[static_cast<std::size_t>(i)];
  }
  c_norm_ = 0.0;
  bound_count_ = 0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    c_norm_ = std::max(c_norm_, std::fabs(cost_[u]));
    if (lower_[u] == upper_[u] && is_finite_bound(lower_[u])) {
      fixed_[u] = 1;
      continue;
    }
    has_lower_[u] = is_finite_bound(lower_[u]) ? 1 : 0;
    has_upper_[u] = is_finite_bound(upper_[u]) ? 1 : 0;
    bound_count_ += has_lower_[u] + has_upper_[u];
  }

  // THE STARTING POINT is ipm.cpp's (Mehrotra's shifted slacks, multipliers from the costs,
  // balanced), with tau = 1 and kappa the mean product, so the pair starts as centred as the
  // rest. The embedding needs no feasible start: every residual shrinks at one rate.
  x_.assign(T, 0.0);
  sl_.assign(T, 0.0);
  su_.assign(T, 0.0);
  zl_.assign(T, 0.0);
  zu_.assign(T, 0.0);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u] != 0) {
      x_[u] = lower_[u];
    } else if (has_lower_[u] != 0 && has_upper_[u] != 0) {
      x_[u] = 0.5 * (lower_[u] + upper_[u]);
    } else if (has_lower_[u] != 0) {
      x_[u] = lower_[u] + 1.0;
    } else if (has_upper_[u] != 0) {
      x_[u] = upper_[u] - 1.0;
    }
  }
  if (m_ > 0) {
    std::vector<double> activity(static_cast<std::size_t>(m_), 0.0);
    model_.matrix.multiply_add(x_.data(), activity.data());
    for (Index i = 0; i < m_; ++i) {
      const auto u = static_cast<std::size_t>(n_ + i);
      if (fixed_[u] == 0) x_[u] = activity[static_cast<std::size_t>(i)];
    }
  }
  double worst = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (has_lower_[u] != 0) worst = std::min(worst, x_[u] - lower_[u]);
    if (has_upper_[u] != 0) worst = std::min(worst, upper_[u] - x_[u]);
  }
  const double shift = std::max(1.0, -1.5 * worst);
  double products = 0.0, slacks = 0.0, duals = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (has_lower_[u] != 0) {
      sl_[u] = x_[u] - lower_[u] + shift;
      zl_[u] = std::max(cost_[u], 1.0);
      products += sl_[u] * zl_[u];
      slacks += sl_[u];
      duals += zl_[u];
    }
    if (has_upper_[u] != 0) {
      su_[u] = upper_[u] - x_[u] + shift;
      zu_[u] = std::max(-cost_[u], 1.0);
      products += su_[u] * zu_[u];
      slacks += su_[u];
      duals += zu_[u];
    }
  }
  double mean = 1.0;
  if (products > 0.0) {
    const double slack_balance = 0.5 * products / duals;
    const double dual_balance = 0.5 * products / slacks;
    products = 0.0;
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (has_lower_[u] != 0) {
        sl_[u] += slack_balance;
        zl_[u] += dual_balance;
        products += sl_[u] * zl_[u];
      }
      if (has_upper_[u] != 0) {
        su_[u] += slack_balance;
        zu_[u] += dual_balance;
        products += su_[u] * zu_[u];
      }
    }
    mean = products / static_cast<double>(bound_count_);
  }
  tau_ = 1.0;
  kappa_ = mean;
  y_.assign(static_cast<std::size_t>(m_), 0.0);

  // b = -Abar_fixed l_fixed: what the fixed variables put on the rows, per unit of tau.
  b_.assign(static_cast<std::size_t>(m_), 0.0);
  std::vector<double> fixed_part(T, 0.0);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u] != 0) fixed_part[u] = lower_[u];
  }
  constraint_times(fixed_part, &b_);
  data_norm_ = 0.0;
  for (double& v : b_) {
    v = -v;
    data_norm_ = std::max(data_norm_, std::fabs(v));
  }
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (has_lower_[u] != 0) data_norm_ = std::max(data_norm_, std::fabs(lower_[u]));
    if (has_upper_[u] != 0) data_norm_ = std::max(data_norm_, std::fabs(upper_[u]));
  }

  for (std::vector<double>* v : {&r_l_, &r_u_, &r_d_, &theta_, &dx_, &dx2_, &dsl_, &dzl_, &dsu_,
                                 &dzu_, &rmu_l_, &rmu_u_}) {
    v->assign(T, 0.0);
  }
  for (std::vector<double>* v : {&r_p_, &dy_, &dy2_})
    v->assign(static_cast<std::size_t>(m_), 0.0);
}

void Homogeneous::residuals() {
  constraint_times(x_, &r_p_);
  double row_worst = 0.0;
  for (double& v : r_p_) {
    v = -v;
    row_worst = std::max(row_worst, std::fabs(v));
  }
  std::vector<double> aty(static_cast<std::size_t>(total_));
  constraint_transpose_times(y_, &aty);
  double products = 0.0, x_norm = 0.0, dual_worst = 0.0;
  double by = 0.0, lz = 0.0, uz = 0.0;
  cx_ = 0.0;
  max_product_ = 0.0;
  for (std::size_t i = 0; i < b_.size(); ++i) by += b_[i] * y_[i];
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    x_norm = std::max(x_norm, std::fabs(x_[u]));
    if (fixed_[u] != 0) {
      r_d_[u] = 0.0;
      continue;
    }
    cx_ += cost_[u] * x_[u];
    r_d_[u] = cost_[u] * tau_ - aty[u] - zl_[u] + zu_[u];
    dual_worst = std::max(dual_worst, std::fabs(r_d_[u]));
    const double scale = tau_ * tau_ * (1.0 + std::fabs(x_[u]) / tau_);
    if (has_lower_[u] != 0) {
      r_l_[u] = lower_[u] * tau_ + sl_[u] - x_[u];
      row_worst = std::max(row_worst, std::fabs(r_l_[u]));
      products += sl_[u] * zl_[u];
      lz += lower_[u] * zl_[u];
      max_product_ = std::max(max_product_, sl_[u] * zl_[u] / scale);
    }
    if (has_upper_[u] != 0) {
      r_u_[u] = upper_[u] * tau_ - su_[u] - x_[u];
      row_worst = std::max(row_worst, std::fabs(r_u_[u]));
      products += su_[u] * zu_[u];
      uz += upper_[u] * zu_[u];
      max_product_ = std::max(max_product_, su_[u] * zu_[u] / scale);
    }
  }
  dual_ray_ = by + lz - uz;
  r_g_ = kappa_ + cx_ - dual_ray_;
  mu_ = (products + tau_ * kappa_) / static_cast<double>(bound_count_ + 1);
  primal_inf_ = row_worst / tau_ / (1.0 + x_norm / tau_);
  primal_data_inf_ = row_worst / tau_ / (1.0 + data_norm_);
  dual_inf_ = dual_worst / tau_ / (1.0 + c_norm_);
  const double primal_objective = cx_ / tau_;
  gap_ = std::fabs(cx_ - dual_ray_) / tau_ / (1.0 + std::fabs(primal_objective));
}

bool Homogeneous::model_space_holds() {
  // The status guard's own measurement (#582, ipm/model_space.hpp) on the point (x, y, z)/tau.
  std::vector<double> x(x_), y(y_), zl(zl_), zu(zu_);
  for (double& v : x) v /= tau_;
  for (double& v : y) v /= tau_;
  for (double& v : zl) v /= tau_;
  for (double& v : zu) v /= tau_;
  ScaledIterate it;
  it.matrix = &model_.matrix;
  it.cost = &cost_;
  it.lower = &lower_;
  it.upper = &upper_;
  it.x = &x;
  it.y = &y;
  it.zl = &zl;
  it.zu = &zu;
  it.scaling = scaling_;
  const ModelSpaceMeasure measure = measure_in_model_space(it);
  model_primal_ = measure.primal;
  model_dual_ = measure.dual_residual;
  return model_primal_ <= tol::kPrimalFeasibility &&
         (feasibility_only_ || model_dual_ <= tol::kDualFeasibility);
}

bool Homogeneous::factorize() {
  if (m_ == 0) return true;
  const auto T = static_cast<std::size_t>(total_);
  std::vector<double> theta_x(static_cast<std::size_t>(n_));
  std::vector<double> row_shift(static_cast<std::size_t>(m_));
  for (std::size_t u = 0; u < T; ++u) {
    double inverse = kPrimalRegularization;
    if (has_lower_[u] != 0) inverse += zl_[u] / sl_[u];
    if (has_upper_[u] != 0) inverse += zu_[u] / su_[u];
    theta_[u] = fixed_[u] != 0 ? 0.0 : 1.0 / inverse;
    if (u < static_cast<std::size_t>(n_)) {
      theta_x[u] = theta_[u];
    } else {
      row_shift[u - static_cast<std::size_t>(n_)] = theta_[u];
    }
  }
  if (!normal_equations_lower(model_.matrix, theta_x, row_shift, delta_, &normal_lower_,
                              should_stop_)) {
    return false;
  }
  if (!analyzed_) {
    ldl_.set_factor_budget(max_factor_nonzeros_);
    if (!ldl_.analyze(normal_lower_, should_stop_)) return false;
    analyzed_ = true;
  }
  return ldl_.factorize(normal_lower_, delta_, should_stop_);
}

void Homogeneous::solve_normal(std::vector<double>* rhs) {
  // ipm.cpp's solve: the factors, then kRefinementSteps of refinement against the matrix
  // assembled (stored as its lower triangle).
  if (m_ == 0) return;
  const std::vector<double> b = *rhs;
  ldl_.solve(rhs->data());
  std::vector<double> residual(static_cast<std::size_t>(m_));
  for (int step = 0; step < kRefinementSteps; ++step) {
    std::fill(residual.begin(), residual.end(), 0.0);
    for (Index j = 0; j < m_; ++j) {
      const ColumnView column = normal_lower_.column(j);
      const double vj = (*rhs)[static_cast<std::size_t>(j)];
      double dot = 0.0;
      for (Index p = 0; p < column.size; ++p) {
        const Index i = column.rows[p];
        residual[static_cast<std::size_t>(i)] += column.values[p] * vj;
        if (i != j) dot += column.values[p] * (*rhs)[static_cast<std::size_t>(i)];
      }
      residual[static_cast<std::size_t>(j)] += dot;
    }
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = b[i] - residual[i];
    ldl_.solve(residual.data());
    for (std::size_t i = 0; i < residual.size(); ++i) (*rhs)[i] += residual[i];
  }
}

void Homogeneous::second_system() {
  // (dx2, dy2): the direction per unit of dtau, against b - Abar Theta (p - c).
  const auto T = static_cast<std::size_t>(total_);
  std::vector<double> h(T, 0.0), theta_h(T, 0.0);
  for (std::size_t u = 0; u < T; ++u) {
    if (fixed_[u] != 0) continue;
    double p = 0.0;
    if (has_lower_[u] != 0) p += zl_[u] / sl_[u] * lower_[u];
    if (has_upper_[u] != 0) p += zu_[u] / su_[u] * upper_[u];
    h[u] = p - cost_[u];
    theta_h[u] = theta_[u] * h[u];
  }
  constraint_times(theta_h, &dy2_);
  for (std::size_t i = 0; i < dy2_.size(); ++i) dy2_[i] = b_[i] - dy2_[i];
  solve_normal(&dy2_);
  constraint_transpose_times(dy2_, &dx2_);
  g2_ = 0.0;
  for (std::size_t i = 0; i < dy2_.size(); ++i) g2_ += b_[i] * dy2_[i];
  for (std::size_t u = 0; u < T; ++u) {
    if (fixed_[u] != 0) {
      dx2_[u] = 0.0;
      continue;
    }
    dx2_[u] = theta_[u] * (dx2_[u] + h[u]);
    g2_ -= cost_[u] * dx2_[u];
    if (has_lower_[u] != 0) g2_ -= lower_[u] * zl_[u] * (dx2_[u] - lower_[u]) / sl_[u];
    if (has_upper_[u] != 0) g2_ += upper_[u] * zu_[u] * (upper_[u] - dx2_[u]) / su_[u];
  }
}

void Homogeneous::direction(double eta, double r_mu_t) {
  // (dx1, dy1) for the target in rmu_l_, rmu_u_ and eta, then dtau from the gap row and the
  // tau-kappa product, then the whole direction (the file comment has the algebra).
  const auto T = static_cast<std::size_t>(total_);
  std::vector<double> g(T, 0.0), theta_g(T, 0.0);
  for (std::size_t u = 0; u < T; ++u) {
    if (fixed_[u] != 0) continue;
    double q = -eta * r_d_[u];
    if (has_lower_[u] != 0) q += (rmu_l_[u] + zl_[u] * eta * r_l_[u]) / sl_[u];
    if (has_upper_[u] != 0) q += (-rmu_u_[u] + zu_[u] * eta * r_u_[u]) / su_[u];
    g[u] = q;
    theta_g[u] = theta_[u] * q;
  }
  constraint_times(theta_g, &dy_);
  for (std::size_t i = 0; i < dy_.size(); ++i) dy_[i] = eta * r_p_[i] - dy_[i];
  solve_normal(&dy_);
  constraint_transpose_times(dy_, &dx_);
  double g1 = 0.0;
  for (std::size_t i = 0; i < dy_.size(); ++i) g1 += b_[i] * dy_[i];
  for (std::size_t u = 0; u < T; ++u) {
    if (fixed_[u] != 0) {
      dx_[u] = 0.0;
      continue;
    }
    dx_[u] = theta_[u] * (dx_[u] + g[u]);
    g1 -= cost_[u] * dx_[u];
    if (has_lower_[u] != 0) {
      g1 += lower_[u] * (rmu_l_[u] - zl_[u] * (dx_[u] - eta * r_l_[u])) / sl_[u];
    }
    if (has_upper_[u] != 0) {
      g1 -= upper_[u] * (rmu_u_[u] - zu_[u] * (-dx_[u] + eta * r_u_[u])) / su_[u];
    }
  }
  dtau_ = (eta * r_g_ + r_mu_t / tau_ - g1) / (g2_ + kappa_ / tau_);
  dkappa_ = (r_mu_t - kappa_ * dtau_) / tau_;
  for (std::size_t i = 0; i < dy_.size(); ++i) dy_[i] += dtau_ * dy2_[i];
  for (std::size_t u = 0; u < T; ++u) {
    if (fixed_[u] != 0) {
      dx_[u] = lower_[u] * dtau_;  // a fixed variable is l tau
      continue;
    }
    dx_[u] += dtau_ * dx2_[u];
    if (has_lower_[u] != 0) {
      dsl_[u] = dx_[u] - lower_[u] * dtau_ - eta * r_l_[u];
      dzl_[u] = (rmu_l_[u] - zl_[u] * dsl_[u]) / sl_[u];
    }
    if (has_upper_[u] != 0) {
      dsu_[u] = -dx_[u] + upper_[u] * dtau_ + eta * r_u_[u];
      dzu_[u] = (rmu_u_[u] - zu_[u] * dsu_[u]) / su_[u];
    }
  }
}

bool Homogeneous::direction_is_finite() const {
  const auto finite = [](const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
  };
  return std::isfinite(dtau_) && std::isfinite(dkappa_) && std::isfinite(g2_) && finite(dx_) &&
         finite(dy_) && finite(dsl_) && finite(dzl_) && finite(dsu_) && finite(dzu_);
}

double Homogeneous::max_step() const {
  double alpha = 1.0 / kStepToBoundary;  // so that the cap below is exactly 1
  const auto limit = [&alpha](double v, double dv) {
    if (dv < 0.0) alpha = std::min(alpha, -v / dv);
  };
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (has_lower_[u] != 0) {
      limit(sl_[u], dsl_[u]);
      limit(zl_[u], dzl_[u]);
    }
    if (has_upper_[u] != 0) {
      limit(su_[u], dsu_[u]);
      limit(zu_[u], dzu_[u]);
    }
  }
  limit(tau_, dtau_);
  limit(kappa_, dkappa_);
  return alpha;
}

}  // namespace sankhya::ipm::hsd
