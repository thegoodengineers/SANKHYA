// SPDX-License-Identifier: Apache-2.0
// Restarted Halpern PDHG for LP (#481), without the reflection step of [LY24].
//
// References:
//   [LY24]  Lu & Yang, "Restarted Halpern PDHG for linear programming",
//           arXiv:2407.16144, 2024.
//   [H67]   Halpern, "Fixed points of nonexpanding maps", Bull. AMS 73, 1967.
//   [CSY24] Chen, Sun, Yuan, Zhang & Zhao, "HPR-LP: an implementation of an HPR
//           method for solving linear programming", arXiv:2408.12179, 2024.
//
// The Halpern iteration replaces the running-average scheme of PDLP:
//
//   z_{k+1} = alpha_k * T(z_k) + (1 - alpha_k) * z_0,   alpha_k = (k+1)/(k+2)
//
// where T is one PDHG step and z_0 is the anchor set at each restart.
// No running averages are maintained. Restarts are triggered when the
// fixed-point residual r_k = ||T(z_k) - z_k||_P drops below 0.2 * r_0.
#pragma once

#include <cstdint>
#include <vector>

#include "sankhya/options.hpp"
#include "sankhya/types.hpp"

namespace sankhya::pdhg {

/// Per-restart state for the Halpern iteration.
struct HalpernState {
  std::vector<double> x_anchor;
  std::vector<double> y_anchor;
  double r0 = 0.0;           ///< fixed-point residual at the start of this period
  Count iter_in_period = 0;  ///< steps taken since the last restart/init
};

/// Result returned by pdhg_halpern_step.
struct HalpernResult {
  bool active = false;          ///< false when pdhg_halpern=false (no-op path)
  double fp_residual = 0.0;     ///< ||T(z) - z||_P before the Halpern blend
  double fp_x2 = 0.0;           ///< ||T(x) - x||^2, so the P-norm can be re-weighted
  double fp_y2 = 0.0;           ///< ||T(y) - y||^2
  bool should_restart = false;  ///< true when the restart criterion is met
};

/// Apply one Halpern combination step for the LP primal-dual iterate.
///
/// When pdhg_halpern=false: returns {false, 0.0, false}, (x_next, y_next) unchanged.
/// Otherwise: computes the fixed-point residual from (x_next-x, y_next-y), applies
///   x_next = alpha * x_next + (1-alpha) * x_anchor,  alpha = (k+1)/(k+2)
///   y_next = alpha * y_next + (1-alpha) * y_anchor
/// in-place, increments state.iter_in_period, and signals a restart when
///   fp_residual < 0.2 * state.r0  (or r0 == 0 on the first step).
///
/// omega is the current primal weight; the P-norm uses
///   ||dz||_P^2 = omega * ||dx||^2 + (1/omega) * ||dy||^2.
HalpernResult pdhg_halpern_step(const std::vector<double>& x, const std::vector<double>& y,
                                std::vector<double>& x_next, std::vector<double>& y_next,
                                HalpernState& state, double omega, const Options& options);

/// Reset the anchor to (x, y) and recompute r0 from the given raw PDHG residuals.
/// Call at the start of a Halpern period (initial setup or after each restart).
void halpern_reset(const std::vector<double>& x, const std::vector<double>& y,
                   double fp_residual, HalpernState& state);

}  // namespace sankhya::pdhg
