// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a primal-dual interior point for second-order cone programs (#755): the
// homogeneous self-dual embedding with Nesterov-Todd scaling. See conic_ipm.cpp for the method.
#pragma once

#include <string>
#include <vector>

#include "conic/conic_form.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya::conic {

struct HsdResult {
  SolveStatus status = SolveStatus::kNotSolved;
  LimitReason stopped_by = LimitReason::kNone;
  std::string message;
  Count iterations = 0;
  /// kOptimal: the point x/tau, y/tau, s/tau of the form. kInfeasible: `y`, `s` is the dual
  /// ray with A'y + s = 0, s in the cone, b'y = 1. kUnbounded: `x` is the primal ray with
  /// A x = 0, x in the cone, c'x = -1. Empty otherwise.
  std::vector<double> x, y, s;
  /// The relative measures at exit, on the normalized point.
  double primal_residual = 0.0;
  double dual_residual = 0.0;
  double gap = 0.0;
};

/// Run the embedding on `form`; the time and iteration limits in `options` are measured on
/// `clock` from its start.
[[nodiscard]] HsdResult solve_hsd(const ConicForm& form, const Options& options, Logger& logger,
                                  SolveControl* control, const Timer& clock);

}  // namespace sankhya::conic
