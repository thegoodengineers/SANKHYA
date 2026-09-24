// SPDX-License-Identifier: Apache-2.0
// Reflected restarted Halpern PDHG for LP (#481) — CPU-side implementation.
//
// References: [LY24] Lu & Yang arXiv:2407.16144; [H67] Halpern 1967; [CSY24] arXiv:2408.12179
// No reference implementation was read. See PROVENANCE.md row 22.
#include "pdhg_halpern.hpp"

#include <cmath>

namespace sankhya::pdhg {

HalpernResult pdhg_halpern_step(const std::vector<double>& x, const std::vector<double>& y,
                                std::vector<double>& x_next, std::vector<double>& y_next,
                                HalpernState& state, double omega,
                                [[maybe_unused]] const Options& options) {
  // Fixed-point residual before the Halpern blend: ||T(z) - z||_P where
  //   ||dz||_P^2 = omega * ||dx||^2 + (1/omega) * ||dy||^2
  double fp2 = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    const double d = x_next[j] - x[j];
    fp2 += omega * d * d;
  }
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double d = y_next[i] - y[i];
    fp2 += d * d / omega;
  }
  const double r_k = std::sqrt(fp2);

  // On the very first step after a reset, record r0 and set anchor.
  // (halpern_reset was called before the first step, so state.r0 is set there.)

  // Halpern blend: z_{k+1} = alpha * T(z_k) + (1-alpha) * z_0
  //   alpha = (k+1)/(k+2), k = iter_in_period (0-indexed count of steps in this period)
  const double k = static_cast<double>(state.iter_in_period);
  const double alpha = (k + 1.0) / (k + 2.0);
  const double one_minus = 1.0 - alpha;

  for (std::size_t j = 0; j < x_next.size(); ++j)
    x_next[j] = alpha * x_next[j] + one_minus * state.x_anchor[j];
  for (std::size_t i = 0; i < y_next.size(); ++i)
    y_next[i] = alpha * y_next[i] + one_minus * state.y_anchor[i];

  ++state.iter_in_period;

  // Restart when the fixed-point residual drops to 20 % of the period-start value,
  // matching the PDLP KKT-ratio criterion ([PDLP] section 4.3, [LY24] section 4).
  // Also restart on the very first step (r0 == 0 sentinel from halpern_reset).
  const bool should_restart = (state.r0 <= 0.0) || (r_k < 0.2 * state.r0);

  return {true, r_k, should_restart};
}

void halpern_reset(const std::vector<double>& x, const std::vector<double>& y,
                   double fp_residual, HalpernState& state) {
  state.x_anchor = x;
  state.y_anchor = y;
  state.r0 = fp_residual;
  state.iter_in_period = 0;
}

}  // namespace sankhya::pdhg
