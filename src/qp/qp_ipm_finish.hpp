// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish after the QP interior point (#980). See qp_ipm_finish.cpp.
#pragma once

#include <vector>

#include "qp_ipm_system.hpp"

namespace sankhya::qp::ipm_detail {

/// The largest complementarity product of (v, y, zl, zu): every finite bound's distance times
/// its multiplier, and every slack row's multiplier times its activity's distance to the
/// nearer side (activity = v_slack - rp, rp the primal residual b - M v).
[[nodiscard]] double largest_complementarity_product(
    const Standard& s, const std::vector<bool>& has_lower, const std::vector<bool>& has_upper,
    const std::vector<double>& v, const std::vector<double>& y, const std::vector<double>& zl,
    const std::vector<double>& zu, const std::vector<double>& rp);

/// Pin every bound the point sits within tol::kQpIpmFinishActiveDistance (scaled) of, solve
/// the reduced equality-constrained KKT system on the rest, and replace (v, y, zl, zu) with
/// the result only when it is primal feasible to `primal_tolerance`, meets `tolerance` on the
/// relative primal and dual residuals, has every multiplier of the right sign and every
/// product under the in-process gate. Returns whether the point was replaced; on false the
/// arguments are untouched.
[[nodiscard]] bool active_set_finish(const Standard& s, const std::vector<bool>& has_lower,
                                     const std::vector<bool>& has_upper, double tolerance,
                                     double primal_tolerance, double regularization_floor,
                                     std::vector<double>* v, std::vector<double>* y,
                                     std::vector<double>* zl, std::vector<double>* zu);

}  // namespace sankhya::qp::ipm_detail
