// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish at the conic interior point's optimum (#755); see
// conic_duals.cpp.
#pragma once

#include <cstdint>
#include <functional>

#include "sankhya/model.hpp"
#include "sankhya/qcqp.hpp"

namespace sankhya::conic {

/// What finish_on_active_set() did to the solution.
enum class Finish : std::uint8_t {
  kNone,      ///< nothing measured better; the interior point's answer stands
  kDuals,     ///< the multipliers were re-derived by least squares at the given point
  kPolished,  ///< point and multipliers were moved by Newton's method on the active set
};

/// Finish an optimal `solution` of `model` (quadratic rows, no explicit cones) on the active
/// set read off its point: Newton's method on the KKT system restricted to it, else
/// least-squares multipliers at the given point. A result replaces the interior point's only
/// when it measures better: model KKT error smaller and primal feasibility no worse.
/// `should_stop` is the caller's deadline and interrupt (#1086), asked before every Newton
/// step and inside each analysis and factorization; when it fires the best answer measured
/// so far stands, which may be the interior point's own.
[[nodiscard]] Finish finish_on_active_set(const QcqpModel& model, Solution* solution,
                                          const std::function<bool()>& should_stop = {});

/// The cone part of solve_conic()'s KKT gate (conic_solve.cpp): the worst relative violation
/// of the cones' dual conditions at x with reduced costs d (minimization space), split as
/// tools/verify_solution_conic.py splits them. Above tol::kDualFeasibility is not optimal.
[[nodiscard]] double cone_dual_violation(const QcqpModel& model, const std::vector<double>& x,
                                         const std::vector<double>& d);

}  // namespace sankhya::conic
