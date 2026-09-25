// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior-point filter line-search method (NLP stage 2). See filter_ipm.hpp
// for the method and its simplifications; equation and section numbers are those of Wachter
// and Biegler, Math. Programming 106(1) (2006). The restoration phase is in
// filter_ipm_restoration.cpp.

#include "nlp/filter_ipm.hpp"

#include <algorithm>
#include <cmath>

#include "nlp/filter_ipm_method.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::nlp {

const char* to_string(IpmExit exit) noexcept {
  switch (exit) {
    case IpmExit::kConverged: return "converged";
    case IpmExit::kAccepted: return "accepted";
    case IpmExit::kIterationLimit: return "iteration_limit";
    case IpmExit::kStopped: return "stopped";
    case IpmExit::kLocallyInfeasible: return "locally_infeasible";
    case IpmExit::kRestorationFailed: return "restoration_failed";
    case IpmExit::kEvaluationFailed: return "evaluation_failed";
    case IpmExit::kFactorizationFailed: return "factorization_failed";
  }
  return "unknown";
}

namespace detail {

FilterMethod::FilterMethod(const BarrierNlp& p, const IpmSettings& s, const IpmHooks& h)
    : p_(p), s_(s), h_(h), n_(p.n()), m_(p.m()) {
  lo_ = p.lower();
  up_ = p.upper();
  has_lo_.assign(static_cast<std::size_t>(n_), 0);
  has_up_.assign(static_cast<std::size_t>(n_), 0);
  fixed_.assign(static_cast<std::size_t>(n_), 0);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const bool lo = std::isfinite(lo_[u]);
    const bool up = std::isfinite(up_[u]);
    if (lo && up && lo_[u] == up_[u]) {
      fixed_[u] = 1;
    } else {
      has_lo_[u] = lo ? 1 : 0;
      has_up_[u] = up ? 1 : 0;
    }
  }
}

double FilterMethod::theta(const Vec& c) const {
  double s = 0.0;
  for (const double v : c) s += std::fabs(v);
  return s;
}

double FilterMethod::barrier(const Vec& w, double f) const {
  double phi = f;
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (has_lo_[u] != 0) phi -= mu_ * std::log(w[u] - lo_[u]);
    if (has_up_[u] != 0) phi -= mu_ * std::log(up_[u] - w[u]);
  }
  return phi;
}

void FilterMethod::jacobian_transpose_times(const Vec& y, Vec* out) const {
  out->assign(static_cast<std::size_t>(n_), 0.0);
  for (Index i = 0; i < m_; ++i) {
    const double yi = y[static_cast<std::size_t>(i)];
    if (yi == 0.0) continue;
    for (Index k = p_.jac_starts()[static_cast<std::size_t>(i)];
         k < p_.jac_starts()[static_cast<std::size_t>(i) + 1]; ++k) {
      (*out)[static_cast<std::size_t>(p_.jac_cols()[static_cast<std::size_t>(k)])] +=
          jac_[static_cast<std::size_t>(k)] * yi;
    }
  }
}

void FilterMethod::dual_residual(Vec* r) const {
  jacobian_transpose_times(lam_, r);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    (*r)[u] = fixed_[u] != 0 ? 0.0 : grad_[u] + (*r)[u] - zl_[u] + zu_[u];
  }
}

double FilterMethod::error(double mu) const {
  // E_mu of eq. (5), with the scalings s_d and s_c.
  Vec r;
  dual_residual(&r);
  double dual = 0.0, primal = 0.0, compl_error = 0.0, lam1 = 0.0, z1 = 0.0;
  Index bounds = 0;
  for (const double v : r) dual = std::max(dual, std::fabs(v));
  for (const double v : c_) primal = std::max(primal, std::fabs(v));
  for (const double v : lam_) lam1 += std::fabs(v);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (has_lo_[u] != 0) {
      compl_error = std::max(compl_error, std::fabs((w_[u] - lo_[u]) * zl_[u] - mu));
      z1 += std::fabs(zl_[u]);
      ++bounds;
    }
    if (has_up_[u] != 0) {
      compl_error = std::max(compl_error, std::fabs((up_[u] - w_[u]) * zu_[u] - mu));
      z1 += std::fabs(zu_[u]);
      ++bounds;
    }
  }
  const double s_max = tol::kNlpScalingMax;
  const double s_d =
      std::max(s_max, (lam1 + z1) / std::max(1.0, static_cast<double>(m_ + bounds))) / s_max;
  const double s_c = std::max(s_max, z1 / std::max(1.0, static_cast<double>(bounds))) / s_max;
  return std::max({dual / s_d, primal, compl_error / s_c});
}

bool FilterMethod::in_filter(double theta, double phi) const {
  for (const auto& [t, f] : filter_) {
    if (theta >= t && phi >= f) return true;
  }
  return false;
}

bool FilterMethod::evaluate_all(Evaluation* error) {
  return p_.gradient(w_, &f_, &grad_, error) && p_.constraints(w_, &c_, error) &&
         p_.jacobian(w_, &jac_, error);
}

void FilterMethod::push_interior() {
  // Sec. 3.6: move the start at least kappa_1 max(1, |bound|) inside each bound, and no more
  // than kappa_2 of the gap when both bounds are finite.
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (fixed_[u] != 0) {
      w_[u] = lo_[u];
      continue;
    }
    const double gap = up_[u] - lo_[u];
    if (has_lo_[u] != 0) {
      double push = tol::kNlpBoundPush * std::max(1.0, std::fabs(lo_[u]));
      if (has_up_[u] != 0) push = std::min(push, tol::kNlpBoundFraction * gap);
      w_[u] = std::max(w_[u], lo_[u] + push);
    }
    if (has_up_[u] != 0) {
      double push = tol::kNlpBoundPush * std::max(1.0, std::fabs(up_[u]));
      if (has_lo_[u] != 0) push = std::min(push, tol::kNlpBoundFraction * gap);
      w_[u] = std::min(w_[u], up_[u] - push);
    }
  }
}

void FilterMethod::least_squares_multipliers() {
  // Eq. (36): lambda minimising || grad f - z + J^T lambda ||, from [I J^T; J 0].
  lam_.assign(static_cast<std::size_t>(m_), 0.0);
  if (m_ == 0) return;
  const Vec zero_hessian(p_.hess_rows().size(), 0.0);
  const Vec ones(static_cast<std::size_t>(n_), 1.0);
  if (!kkt_.factorize(zero_hessian, jac_, ones, fixed_, 0.0, 0.0, s_.should_stop)) return;
  Vec rhs(static_cast<std::size_t>(n_ + m_), 0.0);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    rhs[u] = fixed_[u] != 0 ? 0.0 : -(grad_[u] - zl_[u] + zu_[u]);
  }
  kkt_.solve(&rhs);
  double largest = 0.0;
  for (Index i = 0; i < m_; ++i)
    largest = std::max(largest, std::fabs(rhs[static_cast<std::size_t>(n_ + i)]));
  if (!std::isfinite(largest) || largest > tol::kNlpLambdaMax) return;
  for (Index i = 0; i < m_; ++i)
    lam_[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(n_ + i)];
}

IpmResult FilterMethod::finish(IpmExit exit, std::string message) const {
  IpmResult r;
  r.exit = exit;
  r.at = IpmIterate{w_, lam_, zl_, zu_};
  r.iterations = iterations_;
  r.mu = mu_;
  r.theta = theta(c_);
  r.message = std::move(message);
  return r;
}

IpmResult FilterMethod::run(IpmIterate start) {
  w_ = start.w;
  w_.resize(static_cast<std::size_t>(n_), 0.0);
  if (s_.push_start_into_bounds) push_interior();
  if (n_ == 0) return finish(IpmExit::kConverged, "no variables");
  if (!kkt_.analyze(p_, s_.should_stop)) {
    return finish(kkt_.stopped_early() ? IpmExit::kStopped : IpmExit::kFactorizationFailed,
                  "the symbolic analysis of the KKT system did not complete");
  }
  Evaluation failure;
  if (!evaluate_all(&failure)) {
    return finish(IpmExit::kEvaluationFailed, "at the starting point: " + failure.message);
  }
  if (start.zl.size() == w_.size() && start.zu.size() == w_.size()) {
    zl_ = start.zl;
    zu_ = start.zu;
  } else {
    zl_.assign(w_.size(), 0.0);
    zu_.assign(w_.size(), 0.0);
    for (std::size_t u = 0; u < w_.size(); ++u) {
      if (has_lo_[u] != 0) zl_[u] = 1.0;  // z_0 = 1 (sec. 3.6)
      if (has_up_[u] != 0) zu_[u] = 1.0;
    }
  }
  if (start.lambda.size() == static_cast<std::size_t>(m_)) {
    lam_ = start.lambda;
  } else if (s_.least_squares_multipliers) {
    least_squares_multipliers();
  } else {
    lam_.assign(static_cast<std::size_t>(m_), 0.0);
  }
  mu_ = s_.mu_init > 0.0 ? s_.mu_init : tol::kNlpMuInit;
  tau_ = std::max(tol::kNlpTauMin, 1.0 - mu_);
  const double theta0 = theta(c_);
  theta_max_ = tol::kNlpThetaMaxFactor * std::max(1.0, theta0);
  theta_min_ = tol::kNlpThetaMinFactor * std::max(1.0, theta0);

  for (;;) {
    if (s_.should_stop && s_.should_stop()) return finish(IpmExit::kStopped, "stopped");
    const bool done = h_.converged ? h_.converged(IpmIterate{w_, lam_, zl_, zu_}, mu_)
                                   : error(0.0) <= s_.tolerance;
    if (done) return finish(IpmExit::kConverged, "");
    if (iterations_ >= s_.max_iterations) {
      return finish(IpmExit::kIterationLimit, "the iteration limit was reached");
    }
    // A-3: decrease mu while the barrier problem is solved to kappa_eps mu (eq. 7).
    while (mu_ > s_.mu_min && error(mu_) <= tol::kNlpKappaEpsilon * mu_) {
      mu_ = std::max(s_.mu_min,
                     std::min(tol::kNlpKappaMu * mu_, std::pow(mu_, tol::kNlpThetaMu)));
      tau_ = std::max(tol::kNlpTauMin, 1.0 - mu_);
      filter_.clear();  // the filter belongs to one barrier problem (sec. 2.3)
    }
    const StepOutcome outcome = step();
    if (outcome == StepOutcome::kStopped) return finish(IpmExit::kStopped, "stopped");
    if (outcome == StepOutcome::kRestoration) {
      std::string message;
      const IpmExit exit = restoration(&message);
      if (exit != IpmExit::kAccepted) return finish(exit, message);
    }
    if (outcome == StepOutcome::kAccepted && h_.accept &&
        h_.accept(IpmIterate{w_, lam_, zl_, zu_})) {
      return finish(IpmExit::kAccepted, "");
    }
  }
}

}  // namespace detail

IpmResult run_filter_ipm(const BarrierNlp& p, IpmIterate start, const IpmSettings& settings,
                         const IpmHooks& hooks) {
  detail::FilterMethod method(p, settings, hooks);
  return method.run(std::move(start));
}

}  // namespace sankhya::nlp
