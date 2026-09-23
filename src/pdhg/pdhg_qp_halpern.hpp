// SPDX-License-Identifier: Apache-2.0
// Halpern restarts and PID primal weight for the first-order QP engine (#493).
//
// References:
//   Lu & Yang, *A practical and optimal first-order method for large-scale
//     convex quadratic programming* (PDQP), arXiv:2311.07710
//   HPR-QP, arXiv:2507.02470
//   PDHCG-II, arXiv:2602.23967
#pragma once

#include "sankhya/options.hpp"

namespace sankhya::pdhg {

/// Apply one Halpern-restarted step for the QP primal-dual iterate.
/// The Qx term is computed once per outer iteration and passed in as qx_term.
/// Returns true when the option is active; false (no-op) otherwise.
///
/// CURRENTLY A STUB: returns false immediately when qp_halpern=false (the
/// default). The caller in pdhg.cpp is unchanged until the option is on.
bool qp_halpern_step(const double* qx_term, int n, const Options& options);

}  // namespace sankhya::pdhg
