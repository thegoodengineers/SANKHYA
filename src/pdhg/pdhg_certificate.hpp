// SPDX-License-Identifier: Apache-2.0
// SANKHYA - infeasibility and unboundedness from PDHG restart differences (#484).
//
// Applegate, Lubin and Hinder, "Infeasibility detection with primal-dual hybrid gradient for
// large-scale linear programming", SIAM J. Optimization 34 (2024), arXiv:2102.04592: when the
// LP is infeasible (or unbounded) the difference of successive restart iterates converges to
// a certifying ray. Nothing is claimed on the paper's thresholds: the candidate goes to the
// project's own certificate checkers (src/core/certificate.cpp), the same ones every other
// engine's certificate goes through, and only what they accept is reported.
#pragma once

#include <string>
#include <vector>

#include "../la/scaling.hpp"
#include "sankhya/model.hpp"

namespace sankhya::pdhg {

/// Test one restart's iterate difference. The Farkas certificate is tried first. A primal ray
/// alone proves only that the DUAL is infeasible - the primal may be infeasible too - so a ray
/// is reported as kInfeasibleOrUnbounded, never kUnbounded: this engine does not also carry a
/// feasible point to make the second half of an unboundedness claim. Returns false, and
/// changes nothing, when neither checker accepts the candidate.
bool detect_certificate_from_restart(const Model& model, const Scaling& scaling,
                                     const std::vector<double>& dx, double dx_norm,
                                     const std::vector<double>& dy, double dy_norm,
                                     SolveStatus* status, std::vector<double>* certificate,
                                     std::string* message);

}  // namespace sankhya::pdhg
