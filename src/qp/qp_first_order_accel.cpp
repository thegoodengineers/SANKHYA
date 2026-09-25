// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted Halpern iteration and PID primal weight for the Condat-Vu QP engine
// (#493). See qp_first_order_accel.hpp for the method and its references ([C13], [H67],
// [LY24], [LY23], [PDLP], [LPY25]); written from the papers.

#include "qp_first_order_accel.hpp"

#include <algorithm>
#include <cmath>

#include "sankhya/tolerances.hpp"

namespace sankhya::qp {

double condat_vu_reflection_max(double lipschitz, double tau) {
  const double inverse = 1.0 / tau;
  const double half_l = 0.5 * lipschitz;
  return std::max(0.0, (inverse - half_l) / (inverse + half_l));
}

CondatVuSteps condat_vu_steps_at_weight(double lipschitz, double a2, double omega) {
  // Roots of A t^2 + B t - 1 = 0 with A = 2 a2 omega^2 > 0, B = L/2 >= 0: the positive one is
  // 2 / (B + sqrt(B^2 + 4A)), which has no cancellation. It satisfies 1/t - B = A t > 0, so
  // tau < 2/L and sigma below is positive.
  const double half_l = 0.5 * lipschitz;
  const double quadratic = 2.0 * a2 * omega * omega;
  CondatVuSteps steps;
  steps.tau = 2.0 / (half_l + std::sqrt(half_l * half_l + 4.0 * quadratic));
  steps.sigma = (1.0 / steps.tau - half_l) / (2.0 * a2);
  steps.reflection_max = condat_vu_reflection_max(lipschitz, steps.tau);
  return steps;
}

double condat_vu_weight_of(double tau, double sigma) {
  return std::sqrt(sigma / tau);
}

void halpern_blend(std::vector<double>* z, const std::vector<double>& tz,
                   const std::vector<double>& anchor, double k, double rho) {
  const double w = (k + 1.0) / (k + 2.0);
  const double anchor_share = 1.0 - w;
  for (std::size_t i = 0; i < z->size(); ++i) {
    const double reflected = (1.0 + rho) * tz[i] - rho * (*z)[i];
    (*z)[i] = w * reflected + anchor_share * anchor[i];
  }
}

double pid_primal_weight(double omega, double dx_norm, double dy_norm, const PidGains& gains,
                         PidState* state) {
  if (dx_norm <= tol::kQpPrimalWeightMinMovement ||
      dy_norm <= tol::kQpPrimalWeightMinMovement) {
    return omega;
  }
  const double error = std::log(omega * dx_norm / dy_norm);
  state->integral =
      std::clamp(state->integral + error, -tol::kQpPidIntegralLimit, tol::kQpPidIntegralLimit);
  const double derivative = state->has_previous ? error - state->previous_error : 0.0;
  state->previous_error = error;
  state->has_previous = true;
  const double log_omega =
      std::log(omega) - (gains.kp * error + gains.ki * state->integral + gains.kd * derivative);
  return std::clamp(std::exp(log_omega), tol::kQpPrimalWeightMin, tol::kQpPrimalWeightMax);
}

}  // namespace sankhya::qp
