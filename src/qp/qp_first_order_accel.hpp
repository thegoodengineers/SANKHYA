// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted Halpern iteration and a PID primal weight for the Condat-Vu QP engine
// (#493, CPU). Both are off by default (qp_halpern, qp_primal_weight_pid); with both off
// src/qp/qp_condat_vu.cpp runs the iteration it ran before, operation for operation.
//
// References
//   [C13]   Condat, "A primal-dual splitting method for convex optimization involving
//           Lipschitzian, proximable and linear composite terms", JOTA 158 (2013), Theorem 3.1:
//           with 1/tau - sigma ||A||^2 > L/2 the step T is averaged, and the relaxed step
//           z + rho (T z - z) converges for rho < delta = 2 - (L/2) (1/tau - sigma ||A||^2)^-1.
//   [V13]   Vu, "A splitting algorithm for dual monotone inclusions involving cocoercive
//           operators", Adv. Comput. Math. 38 (2013).
//   [H67]   Halpern, "Fixed points of nonexpanding maps", Bull. AMS 73 (1967).
//   [LY24]  Lu & Yang, "Restarted Halpern PDHG for linear programming", arXiv:2407.16144:
//           the anchored iteration, the reflected form (1 + rho) T - rho I, and restarts on
//           the fixed-point residual.
//   [LY23]  Lu & Yang, "A practical and optimal first-order method for large-scale convex
//           quadratic programming" (PDQP), arXiv:2311.07710: the QP primal-dual operator
//           with the gradient Q x taken once per iteration, restarted.
//   [PDLP]  Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue & Schudy, "Practical large-scale
//           linear programming using primal-dual hybrid gradient", NeurIPS 2021, section 3.2:
//           the primal weight and its smoothed update, which is the proportional-only case
//           of the controller below.
//   [LPY25] Lu, Peng & Yang, "cuPDLPx: a further enhanced GPU-based first-order solver for
//           linear programming", arXiv:2507.14051: the primal weight driven by a PID
//           controller on the log of the primal-dual movement ratio.
// Papers only; no implementation of any of them was read.
#pragma once

#include <vector>

namespace sankhya::qp {

/// Condat-Vu step sizes. `reflection_max` is delta - 1 of [C13] at these steps: the largest
/// rho for which (1 + rho) T - rho I is still nonexpansive in the method's metric.
struct CondatVuSteps {
  double tau = 0.0;
  double sigma = 0.0;
  double reflection_max = 0.0;
};

/// The reflection bound at a primal step tau when sigma is tied to tau by the engine's rule
/// sigma ||A||^2 = (1/tau - L/2) / 2: then 1/tau - sigma ||A||^2 = (1/tau + L/2) / 2 and
/// delta - 1 = (1/tau - L/2) / (1/tau + L/2), in [0, 1), and 1 exactly for an LP (L = 0),
/// where T is firmly nonexpansive.
[[nodiscard]] double condat_vu_reflection_max(double lipschitz, double tau);

/// The steps at primal weight omega, meaning sigma / tau = omega^2 (the [PDLP] convention),
/// under the same coupling rule as the default steps: tau is the positive root of
/// 2 a2 omega^2 tau^2 + (L/2) tau - 1 = 0, taken in its cancellation-free form, and sigma is
/// then (1/tau - L/2) / (2 a2). `a2` is the engine's inflated ||A||^2 estimate.
[[nodiscard]] CondatVuSteps condat_vu_steps_at_weight(double lipschitz, double a2,
                                                      double omega);

/// The weight the engine's default steps correspond to, sqrt(sigma / tau).
[[nodiscard]] double condat_vu_weight_of(double tau, double sigma);

/// Halpern's anchored step in [LY24]'s reflected form, in place on z:
///     z <- w ((1 + rho) T z - rho z) + (1 - w) z_anchor,     w = (k + 1) / (k + 2)
/// where k counts the steps since the last restart and `tz` holds T z.
void halpern_blend(std::vector<double>* z, const std::vector<double>& tz,
                   const std::vector<double>& anchor, double k, double rho);

/// Gains and memory of the primal-weight controller.
struct PidGains {
  double kp = 0.0;
  double ki = 0.0;
  double kd = 0.0;
};
struct PidState {
  double integral = 0.0;
  double previous_error = 0.0;
  bool has_previous = false;
};

/// One controller step at a restart. The error is e = log(omega ||dx|| / ||dy||), zero when
/// the weighted primal and dual movements over the period balance ([PDLP] section 3.2), and
///     log omega <- log omega - (kp e + ki sum(e) + kd (e - e_previous)).
/// kp = 0.5 with ki = kd = 0 is exactly [PDLP]'s smoothing with theta = 0.5. The integral is
/// clamped (anti-windup) and omega is kept in [kQpPrimalWeightMin, kQpPrimalWeightMax].
/// Returns omega unchanged, and leaves the state alone, when either movement is below
/// kQpPrimalWeightMinMovement, where the ratio is noise.
[[nodiscard]] double pid_primal_weight(double omega, double dx_norm, double dy_norm,
                                       const PidGains& gains, PidState* state);

}  // namespace sankhya::qp
