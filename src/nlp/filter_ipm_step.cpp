// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one iteration of the filter interior point: the inertia-corrected Newton step and
// the filter line search (NLP stage 2). Wachter and Biegler, Math. Programming 106(1)
// (2006): Algorithm IC (sec. 3.1), the step (eq. 11-13), the fraction to the boundary (eq.
// 15), the multiplier safeguard (eq. 16), the filter line search (sec. 2.3, eq. 18-23).

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "nlp/filter_ipm_method.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::nlp::detail {

bool FilterMethod::factorize_with_inertia_correction(const Vec& sigma) {
  // IC-1: no correction. A certified factorization ends it, and delta_w_last stays.
  if (kkt_.factorize(hess_, jac_, sigma, fixed_, 0.0, 0.0, s_.should_stop)) return true;
  if (kkt_.stopped_early()) return false;
  // IC-2..IC-6. The factorization cannot tell a singular Jacobian from wrong inertia, so
  // delta_c is applied from the first correction on (it is kNlpDeltaC mu^kappa_c: tiny).
  const double delta_c = tol::kNlpDeltaC * std::pow(mu_, tol::kNlpKappaC);
  double delta_w = delta_w_last_ == 0.0
                       ? tol::kNlpDeltaWInit
                       : std::max(tol::kNlpDeltaWMin, tol::kNlpKappaWMinus * delta_w_last_);
  for (;;) {
    if (kkt_.factorize(hess_, jac_, sigma, fixed_, delta_w, delta_c, s_.should_stop)) {
      delta_w_last_ = delta_w;
      return true;
    }
    if (kkt_.stopped_early()) return false;
    delta_w *= delta_w_last_ == 0.0 ? tol::kNlpKappaWPlusFirst : tol::kNlpKappaWPlus;
    if (delta_w > tol::kNlpDeltaWMax) return false;
  }
}

double FilterMethod::fraction_to_boundary(const Vec& dw) const {
  double alpha = 1.0;
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (has_lo_[u] != 0 && dw[u] < 0.0) {
      alpha = std::min(alpha, tau_ * (w_[u] - lo_[u]) / -dw[u]);
    }
    if (has_up_[u] != 0 && dw[u] > 0.0) {
      alpha = std::min(alpha, tau_ * (up_[u] - w_[u]) / dw[u]);
    }
  }
  return alpha;
}

double FilterMethod::fraction_to_boundary_dual(const Vec& dzl, const Vec& dzu) const {
  double alpha = 1.0;
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (has_lo_[u] != 0 && dzl[u] < 0.0) alpha = std::min(alpha, tau_ * zl_[u] / -dzl[u]);
    if (has_up_[u] != 0 && dzu[u] < 0.0) alpha = std::min(alpha, tau_ * zu_[u] / -dzu[u]);
  }
  return alpha;
}

void FilterMethod::safeguard_multipliers() {
  // Eq. (16): keep each z within a factor kappa_Sigma of its primal-dual value mu / s.
  const double k = tol::kNlpKappaSigma;
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (has_lo_[u] != 0) {
      const double s = w_[u] - lo_[u];
      zl_[u] = std::max(std::min(zl_[u], k * mu_ / s), mu_ / (k * s));
    }
    if (has_up_[u] != 0) {
      const double s = up_[u] - w_[u];
      zu_[u] = std::max(std::min(zu_[u], k * mu_ / s), mu_ / (k * s));
    }
  }
}

FilterMethod::StepOutcome FilterMethod::step() {
  Evaluation failure;
  if (!p_.hessian(w_, 1.0, lam_, &hess_, &failure)) return StepOutcome::kRestoration;
  const auto n = static_cast<std::size_t>(n_);
  Vec sigma(n, 0.0), grad_phi(n, 0.0);
  for (std::size_t u = 0; u < n; ++u) {
    if (fixed_[u] != 0) continue;
    grad_phi[u] = grad_[u];
    if (has_lo_[u] != 0) {
      sigma[u] += zl_[u] / (w_[u] - lo_[u]);
      grad_phi[u] -= mu_ / (w_[u] - lo_[u]);
    }
    if (has_up_[u] != 0) {
      sigma[u] += zu_[u] / (up_[u] - w_[u]);
      grad_phi[u] += mu_ / (up_[u] - w_[u]);
    }
  }
  if (!factorize_with_inertia_correction(sigma)) {
    return kkt_.stopped_early() ? StepOutcome::kStopped : StepOutcome::kRestoration;
  }
  // The right-hand side of eq. (13): -(grad phi + J^T lambda) and -c.
  Vec jt_lambda;
  jacobian_transpose_times(lam_, &jt_lambda);
  Vec d(n + static_cast<std::size_t>(m_), 0.0);
  for (std::size_t u = 0; u < n; ++u)
    d[u] = fixed_[u] != 0 ? 0.0 : -(grad_phi[u] + jt_lambda[u]);
  for (Index i = 0; i < m_; ++i)
    d[n + static_cast<std::size_t>(i)] = -c_[static_cast<std::size_t>(i)];
  kkt_.solve(&d);
  const Vec dw(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(n));
  const Vec dl(d.begin() + static_cast<std::ptrdiff_t>(n), d.end());
  // The bound multipliers' steps, eliminated from the full system (eq. 12).
  Vec dzl(n, 0.0), dzu(n, 0.0);
  for (std::size_t u = 0; u < n; ++u) {
    if (has_lo_[u] != 0) {
      const double s = w_[u] - lo_[u];
      dzl[u] = mu_ / s - zl_[u] - zl_[u] / s * dw[u];
    }
    if (has_up_[u] != 0) {
      const double s = up_[u] - w_[u];
      dzu[u] = mu_ / s - zu_[u] + zu_[u] / s * dw[u];
    }
  }
  const double alpha_max = fraction_to_boundary(dw);
  const double alpha_z = fraction_to_boundary_dual(dzl, dzu);

  // ---- The filter line search (sec. 2.3).
  const double theta_k = theta(c_);
  const double phi_k = barrier(w_, f_);
  double slope = 0.0;  // grad phi_mu' dw
  double tiny = 0.0;
  for (std::size_t u = 0; u < n; ++u) {
    slope += grad_phi[u] * dw[u];
    tiny = std::max(tiny, std::fabs(dw[u]) / (1.0 + std::fabs(w_[u])));
  }
  const double gt = tol::kNlpGammaTheta, gp = tol::kNlpGammaPhi;
  double alpha_min = gt;  // eq. (23)
  if (slope < 0.0) {
    alpha_min = std::min(gt, gp * theta_k / -slope);
    if (theta_k <= theta_min_) {
      alpha_min =
          std::min(alpha_min, tol::kNlpSwitchDelta * std::pow(theta_k, tol::kNlpSwitchSTheta) /
                                  std::pow(-slope, tol::kNlpSwitchSPhi));
    }
  }
  alpha_min *= tol::kNlpGammaAlpha;

  double alpha = alpha_max;
  bool f_type = false;
  Vec trial(n), trial_c;
  double trial_f = 0.0;
  for (;;) {
    for (std::size_t u = 0; u < n; ++u) trial[u] = w_[u] + alpha * dw[u];
    bool accepted = false;
    Evaluation e;
    if (p_.objective(trial, &trial_f, &e) && p_.constraints(trial, &trial_c, &e)) {
      const double theta_t = theta(trial_c);
      const double phi_t = barrier(trial, trial_f);
      if (tiny <= tol::kNlpTinyStep) {
        accepted = true;  // the step is rounding-sized; the tests can only mis-judge it
        f_type = true;
      } else if (std::isfinite(phi_t) && theta_t < theta_max_ && !in_filter(theta_t, phi_t)) {
        const bool switching =
            slope < 0.0 && alpha * std::pow(-slope, tol::kNlpSwitchSPhi) >
                               tol::kNlpSwitchDelta * std::pow(theta_k, tol::kNlpSwitchSTheta);
        if (theta_k <= theta_min_ && switching) {
          // Armijo on the barrier objective (eq. 20): an f-type step.
          accepted = phi_t <= phi_k + tol::kNlpEtaPhi * alpha * slope;
          f_type = accepted;
        } else {
          // Sufficient progress in feasibility or the barrier objective (eq. 18).
          accepted = theta_t <= (1.0 - gt) * theta_k || phi_t <= phi_k - gp * theta_k;
          f_type = false;
        }
      }
    }
    if (accepted) break;
    alpha *= tol::kNlpBacktrack;
    if (alpha < alpha_min) return StepOutcome::kRestoration;
  }

  // ---- Accept (A-6, A-7).
  if (!f_type) filter_.emplace_back((1.0 - gt) * theta_k, phi_k - gp * theta_k);
  w_ = trial;
  for (Index i = 0; i < m_; ++i) {
    lam_[static_cast<std::size_t>(i)] += alpha * dl[static_cast<std::size_t>(i)];
  }
  for (std::size_t u = 0; u < n; ++u) {
    zl_[u] += alpha_z * dzl[u];
    zu_[u] += alpha_z * dzu[u];
  }
  safeguard_multipliers();
  ++iterations_;
  if (!evaluate_all(&failure)) return StepOutcome::kRestoration;
  if (h_.log) {
    Vec r;
    dual_residual(&r);
    double dual = 0.0;
    for (const double v : r) dual = std::max(dual, std::fabs(v));
    h_.log(iterations_, f_, theta(c_), dual, mu_, alpha, false);
  }
  return StepOutcome::kAccepted;
}

}  // namespace sankhya::nlp::detail
