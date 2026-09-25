// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a primal-dual interior-point method with a filter line search for smooth NLP
// (NLP stage 2).
//
// THE METHOD is Algorithm A of Wachter and Biegler, "On the implementation of an interior-
// point filter line-search algorithm for large-scale nonlinear programming", Mathematical
// Programming 106(1) (2006), written from the paper - no solver's source was read (see
// docs/PROVENANCE.md, judgement calls). For min f(w) s.t. c(w) = 0, w_L <= w <= w_U:
//
//   * the barrier problem  min phi_mu(w) = f(w) - mu sum log(w - w_L) - mu sum log(w_U - w)
//     s.t. c(w) = 0 is solved approximately for a decreasing sequence of mu (sec. 2.1,
//     eq. 7): mu+ = max(mu_min, min(kappa_mu mu, mu^theta_mu)) once E_mu <= kappa_eps mu;
//   * each step is the Newton step of the primal-dual equations (eq. 11-13), computed from
//     the system in nlp_kkt.hpp, with the inertia correction of Algorithm IC (sec. 3.1);
//   * the step length is chosen by a backtracking line search against a FILTER of (theta,
//     phi) pairs, theta = ||c||_1 (sec. 2.3): a trial point is accepted if it improves
//     feasibility or the barrier objective enough against the current point and is not
//     dominated by the filter; a switching condition decides when the Armijo condition on phi
//     must hold instead;
//   * the fraction-to-the-boundary rule keeps w and the bound multipliers z strictly interior
//     (eq. 15), and z is kept within kappa_Sigma of its primal-dual value (eq. 16);
//   * when the line search cannot find an acceptable step it enters the FEASIBILITY
//     RESTORATION PHASE (sec. 3.3): the same method, applied to the minimum-violation
//     problem of barrier_nlp.hpp, until a point acceptable to the filter with a violation
//     reduced by kappa_resto is found. If that problem instead CONVERGES while the
//     violation is still positive, the point is a local minimizer of the infeasibility: the
//     model is reported LOCALLY INFEASIBLE, which proves nothing about the whole domain.
//
// SIMPLIFICATIONS, stated. No second-order correction (sec. 2.4) - the Maratos effect can
// cost iterations, never correctness; no watchdog; no automatic problem scaling; the
// restoration phase is the paper's l1 problem solved by this same method without its own
// restoration (a restoration that needs restoring fails, as the paper's does). The inertia is
// certified only one way (nlp_kkt.hpp): an uncertified factorization is treated as wrong.
//
// WHAT "CONVERGED" MEANS is decided by the caller (IpmHooks::converged), because it is the
// project's KKT check at the project tolerances in the MODEL'S own terms, not this method's
// scaled error. Without a hook the scaled error E_0 of eq. 5 against `tolerance` is used (the
// restoration phase's own problem).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "nlp/barrier_nlp.hpp"
#include "sankhya/types.hpp"

namespace sankhya::nlp {

struct IpmIterate {
  Vec w, lambda, zl, zu;  ///< zl, zu are 0 where the bound is infinite or the column fixed
};

struct IpmSettings {
  double mu_init = 0.0;    ///< 0: kNlpMuInit
  double mu_min = 0.0;     ///< the smallest barrier parameter; set by the caller
  double tolerance = 0.0;  ///< E_0 target when no convergence hook is given
  Count max_iterations = 0;
  bool allow_restoration = true;
  bool push_start_into_bounds = true;  ///< sec. 3.6; the restoration phase starts inside
  bool least_squares_multipliers = true;
  std::function<bool()> should_stop;  ///< a deadline or an interrupt
};

enum class IpmExit {
  kConverged,  ///< the convergence test passed
  kAccepted,   ///< IpmHooks::accept ended it (the restoration phase's success)
  kIterationLimit,
  kStopped,            ///< should_stop()
  kLocallyInfeasible,  ///< the restoration phase converged with the violation positive
  kRestorationFailed,  ///< the restoration phase ended without a usable point
  kEvaluationFailed,   ///< the start is outside the functions' domain
  kFactorizationFailed,
};

[[nodiscard]] const char* to_string(IpmExit exit) noexcept;

struct IpmResult {
  IpmExit exit = IpmExit::kIterationLimit;
  IpmIterate at;
  Count iterations = 0;  ///< restoration iterations included
  double mu = 0.0;
  double theta = 0.0;  ///< ||c(w)||_1 at the returned point
  std::string message;
};

struct IpmHooks {
  std::function<bool(const IpmIterate&, double mu)> converged;
  std::function<bool(const IpmIterate&)> accept;
  /// iteration, f, ||c||_1, ||dual residual||_inf, mu, step, in restoration
  std::function<void(Count, double, double, double, double, double, bool)> log;
};

/// Run the method from `start` (w; the multipliers are initialised when their sizes do not
/// match p).
[[nodiscard]] IpmResult run_filter_ipm(const BarrierNlp& p, IpmIterate start,
                                       const IpmSettings& settings, const IpmHooks& hooks);

}  // namespace sankhya::nlp
