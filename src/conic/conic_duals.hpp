// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish at the conic interior point's optimum (#755); see
// conic_duals.cpp.
#pragma once

#include <cstdint>

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
[[nodiscard]] Finish finish_on_active_set(const QcqpModel& model, Solution* solution);

}  // namespace sankhya::conic
