// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the feasibility restoration phase of the filter interior point (NLP stage 2).
//
// Wachter and Biegler, Math. Programming 106(1) (2006), sec. 3.3: when the line search finds
// no acceptable step, the method looks for a point that is less infeasible by solving
//
//     minimize   rho sum (p + n) + zeta/2 || D_R (w - w_R) ||^2
//     subject to c(w) - p + n = 0,  w_L <= w <= w_U,  p, n >= 0        (eq. 29)
//
// with zeta = sqrt(mu) and D_R = diag(min(1, 1 / |w_R|)) (eq. 30), starting from p and n
// given in closed form by eq. (33), with this same interior point (without a restoration of
// its own). It returns as soon as a point is acceptable to the ORIGINAL filter and has
// reduced the violation by kappa_resto (sec. 3.3). If the restoration problem is instead
// solved - a stationary point of the l1 violation plus the proximity term - while the
// original violation is still positive, the model is LOCALLY infeasible there: the method
// has found a local minimizer of the infeasibility, which proves nothing about the rest of
// the domain, and the result says exactly that.

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <fmt/format.h>

#include "nlp/filter_ipm_method.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::nlp::detail {

IpmExit FilterMethod::restoration(std::string* message) {
  if (!s_.allow_restoration) {
    *message = "the line search failed inside the restoration phase";
    return IpmExit::kRestorationFailed;
  }
  const double theta_r = theta(c_);
  const double phi_r = barrier(w_, f_);
  // A-9: the current point enters the filter before restoration leaves it (sec. 2.3).
  filter_.emplace_back((1.0 - tol::kNlpGammaTheta) * theta_r,
                       phi_r - tol::kNlpGammaPhi * theta_r);

  const double rho = tol::kNlpRestorationRho;
  double c_max = 0.0;
  for (const double v : c_) c_max = std::max(c_max, std::fabs(v));
  const double mu_r = std::max(mu_, c_max);
  const RestorationNlp problem(&p_, w_, std::sqrt(mu_), rho);

  // Eq. (33): p and n that satisfy c - p + n = 0 and the restoration's complementarity.
  const auto m = static_cast<std::size_t>(m_);
  const auto n = static_cast<std::size_t>(n_);
  IpmIterate start;
  start.w = w_;
  start.w.resize(n + 2 * m, 0.0);
  start.lambda.assign(m, 0.0);
  start.zl.assign(n + 2 * m, 0.0);
  start.zu.assign(n + 2 * m, 0.0);
  for (std::size_t u = 0; u < n; ++u) {
    start.zl[u] = has_lo_[u] != 0 ? std::min(rho, zl_[u]) : 0.0;
    start.zu[u] = has_up_[u] != 0 ? std::min(rho, zu_[u]) : 0.0;
  }
  for (std::size_t i = 0; i < m; ++i) {
    const double c = c_[i];
    const double a = (mu_r - rho * c) / (2.0 * rho);
    const double neg = a + std::sqrt(a * a + mu_r * c / (2.0 * rho));
    const double pos = c + neg;
    start.w[n + i] = pos;
    start.w[n + m + i] = neg;
    start.zl[n + i] = mu_r / pos;
    start.zl[n + m + i] = mu_r / neg;
  }

  IpmSettings settings = s_;
  settings.allow_restoration = false;
  settings.push_start_into_bounds = false;
  settings.least_squares_multipliers = false;
  settings.mu_init = mu_r;
  settings.max_iterations = tol::kNlpMaxRestorationIterations;

  const FilterMethod& self = *this;
  Vec probe_c;
  IpmHooks hooks;
  hooks.accept = [&](const IpmIterate& it) {
    // Sec. 3.3: back to the original problem once the violation fell by kappa_resto and the
    // point is acceptable to the original filter.
    const Vec w(it.w.begin(), it.w.begin() + static_cast<std::ptrdiff_t>(n));
    double f = 0.0;
    Evaluation e;
    if (!p_.objective(w, &f, &e) || !p_.constraints(w, &probe_c, &e)) return false;
    const double t = self.theta(probe_c);
    if (t > tol::kNlpKappaResto * theta_r || t >= theta_max_) return false;
    const double phi = self.barrier(w, f);
    return std::isfinite(phi) && !self.in_filter(t, phi);
  };
  if (h_.log) {
    hooks.log = [&](Count it, double f, double t, double dual, double mu, double alpha, bool) {
      h_.log(iterations_ + it, f, t, dual, mu, alpha, true);
    };
  }
  const IpmResult result = run_filter_ipm(problem, start, settings, hooks);
  iterations_ += result.iterations;

  const Vec w(result.at.w.begin(), result.at.w.begin() + static_cast<std::ptrdiff_t>(n));
  if (result.exit == IpmExit::kAccepted) {
    w_ = w;
    zl_.assign(result.at.zl.begin(), result.at.zl.begin() + static_cast<std::ptrdiff_t>(n));
    zu_.assign(result.at.zu.begin(), result.at.zu.begin() + static_cast<std::ptrdiff_t>(n));
    safeguard_multipliers();
    Evaluation failure;
    if (!evaluate_all(&failure)) {
      *message = "after the restoration phase: " + failure.message;
      return IpmExit::kRestorationFailed;
    }
    least_squares_multipliers();
    return IpmExit::kAccepted;
  }
  if (result.exit == IpmExit::kStopped) {
    *message = "stopped in the restoration phase";
    return IpmExit::kStopped;
  }
  if (result.exit == IpmExit::kConverged) {
    // The restoration problem is solved and the violation did not fall enough: a local
    // minimizer of the infeasibility. Report the point it found.
    w_ = w;
    Evaluation failure;
    (void)evaluate_all(&failure);
    *message = fmt::format(
        "the restoration phase converged to a point of constraint violation {:.3e} (l1), "
        "a local minimizer of the infeasibility; this is not a proof that the model has "
        "no feasible point",
        theta(c_));
    return IpmExit::kLocallyInfeasible;
  }
  *message =
      fmt::format("the restoration phase ended without a usable point ({}{}{})",
                  to_string(result.exit), result.message.empty() ? "" : ": ", result.message);
  return result.exit == IpmExit::kIterationLimit ? IpmExit::kIterationLimit
                                                 : IpmExit::kRestorationFailed;
}

}  // namespace sankhya::nlp::detail
