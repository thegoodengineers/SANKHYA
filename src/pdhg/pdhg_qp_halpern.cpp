// SPDX-License-Identifier: Apache-2.0
// Halpern restarts and PID primal weight for the QP first-order engine (#493).
//
// Algorithm sketch (Lu & Yang PDQP, arXiv:2311.07710; HPR-QP, arXiv:2507.02470):
//   The plain Condat-Vu QP step is:
//     x_{t+1} = prox_{sigma f}(x_t - sigma (A^T y_t + Q x_t))
//     y_{t+1} = prox_{tau g*}(y_t + tau A(2 x_{t+1} - x_t))
//   Halpern modification:
//     z_{t+1} = standard step(z_t)
//     x_{t+1} = (1 - 1/(t+2)) * z_{t+1} + (1/(t+2)) * z_0   (Halpern anchor)
//   PID primal weight: sigma updated each restart by a PID controller on the
//   primal-dual gap ratio, avoiding manual tuning.
//   Q x computed once per outer iteration (SpMV on CPU; or cuSPARSE on GPU).
//   Restarts triggered by the normalised gap criterion (PDLP rule, #481).
//
// Current state: stub returning false immediately.
// Activate with qp_halpern=true once implemented.

#include "pdhg_qp_halpern.hpp"

namespace sankhya::pdhg {

bool qp_halpern_step(const double* /*qx_term*/, int /*n*/, const Options& options) {
  if (!options.get_bool("qp_halpern")) {
    return false;
  }
  // TODO(#493): implement Halpern restart and PID primal-weight update.
  return false;
}

}  // namespace sankhya::pdhg
