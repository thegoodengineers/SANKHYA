// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish after the QP interior point (#980). See qp_ipm_finish.cpp.
#pragma once

#include <functional>
#include <vector>

#include "qp_ipm_system.hpp"

namespace sankhya::qp::ipm_detail {

/// The largest complementarity product of (v, y, zl, zu): every finite bound's distance times
/// its multiplier, every column's derived reduced cost H v + g - M^T y times its distance to
/// the nearer finite bound (the product the in-process gate prices), and every slack row's
/// multiplier times its activity's distance to the nearer side (activity = v_slack - rp, rp
/// the primal residual b - M v).
[[nodiscard]] double largest_complementarity_product(
    const Standard& s, const std::vector<bool>& has_lower, const std::vector<bool>& has_upper,
    const std::vector<double>& v, const std::vector<double>& y, const std::vector<double>& zl,
    const std::vector<double>& zu, const std::vector<double>& rp);

/// Pin every bound the point sits within tol::kQpIpmFinishActiveDistance (scaled) of, solve
/// the reduced equality-constrained KKT system on the rest, and replace (v, y, zl, zu) with
/// the result only when it is primal feasible to `primal_tolerance`, meets `tolerance` on the
/// relative primal and dual residuals, has every multiplier of the right sign and every
/// product under the in-process gate. `should_stop` is the solve's own clock: a round is not
/// started once it says so, and the finish declines. Returns whether the point was replaced;
/// on false the arguments are untouched. `postpone_dense` orders the reduced system's dense
/// nodes last, as qp_ipm_postpone_dense does for the interior point's own factorization.
[[nodiscard]] bool active_set_finish(const Standard& s, const std::vector<bool>& has_lower,
                                     const std::vector<bool>& has_upper, double tolerance,
                                     double primal_tolerance, double regularization_floor,
                                     const std::function<bool()>& should_stop,
                                     std::vector<double>* v, std::vector<double>* y,
                                     std::vector<double>* zl, std::vector<double>* zu,
                                     bool postpone_dense = false);

}  // namespace sankhya::qp::ipm_detail
