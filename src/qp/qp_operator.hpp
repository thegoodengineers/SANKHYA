// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the per-iteration arithmetic of the first-order QP engine, behind one interface
// with a host and a device implementation (#493).
//
// qp_condat_vu.cpp owns the ITERATION LOGIC: step sizes, Halpern restarts, the PID primal
// weight, the termination test, the reported point. What it does with the vectors every
// iteration is the same six pieces of arithmetic in the same order:
//
//     T z:   x' = proj_box( x - tau (c + Q x + A' y) ),  xbar = 2 x' - x,
//            y' = v - sigma proj_C(v / sigma),  v = y + sigma A xbar
//     the fixed-point residual ||T z - z|| in diag(1/tau, 1/sigma)
//     z <- T z, or the Halpern blend of T z, z and the anchor
//     the distance to, and the setting of, the restart point and the anchor
//
// That arithmetic is what an operator here provides. HostOperator is the arithmetic the
// engine has always run, moved into a class without a change to a single expression, and
// it is the reference: the device operator (src/gpu/qp_device.cu, option qp_gpu) has to
// match it to rounding on one step and to the tolerance on a whole solve
// (tests/unit/test_qp_device.cpp). The residual evaluation every 50 iterations stays on the
// host from a download of the evaluated point, so the two paths report through one code.
//
// Reference: Condat, JOTA 158 (2013); Vu, Adv. Comput. Math. 38 (2013); the LP case is
// Chambolle & Pock, JMIV 40 (2011).
#pragma once

#include <memory>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::qp {

/// Q x from the stored lower triangle, on the host. Shared by the engine's residual
/// evaluation and the final reduced costs.
void hessian_multiply(const Model& model, const std::vector<double>& x,
                      std::vector<double>* out);

class QpOperator {
 public:
  virtual ~QpOperator() = default;

  /// (x', y') = T(x, y) at the given step sizes. False on a device failure.
  [[nodiscard]] virtual bool step(double tau, double sigma) = 0;
  /// ||T z - z|| in the diagonal metric diag(I / tau, I / sigma), after step().
  [[nodiscard]] virtual double fixed_point_residual(double tau, double sigma) = 0;
  /// z <- T z when rho < 0 (plain), else the Halpern blend
  ///   z <- w ((1 + rho) T z - rho z) + (1 - w) anchor,  w = (k + 1) / (k + 2).
  [[nodiscard]] virtual bool advance(double k, double rho) = 0;
  /// z <- T z regardless (a Halpern restart lands on the evaluated point).
  [[nodiscard]] virtual bool take_tz() = 0;
  [[nodiscard]] virtual bool set_anchor() = 0;   // anchor <- z
  [[nodiscard]] virtual bool set_restart() = 0;  // restart point <- z
  /// ||x - x_restart|| and ||y - y_restart||.
  [[nodiscard]] virtual bool restart_distance(double* dx, double* dy) = 0;
  /// The current z (tz = false) or T z (tz = true) to the host.
  [[nodiscard]] virtual bool download(bool tz, std::vector<double>* x,
                                      std::vector<double>* y) = 0;
  /// Where the arithmetic ran, for the log and the solution's algorithm string.
  [[nodiscard]] virtual const char* where() const = 0;
  /// Seed z = (x, y) directly (#981, QpFirstOrderWarmStart), when the caller has a point
  /// from elsewhere worth starting from rather than the projection of zero - the interior
  /// point's iterate when it stalls (qp_ipm_stall_handoff). False declines (sizes that do
  /// not match this operator's model, or no device support yet) and the caller keeps the
  /// cold start, which never costs an answer. The default implementation always declines.
  [[nodiscard]] virtual bool upload(const std::vector<double>& x, const std::vector<double>& y) {
    (void)x;
    (void)y;
    return false;
  }
};

/// The reference arithmetic on the host. x starts at the projection of 0 onto the box, y
/// at 0, as the engine always has.
[[nodiscard]] std::unique_ptr<QpOperator> make_host_operator(const Model& model);

}  // namespace sankhya::qp
