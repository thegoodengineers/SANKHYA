// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dual recovery at the conic interior point's optimum (#755); see conic_duals.cpp.
#pragma once

#include "sankhya/model.hpp"
#include "sankhya/qcqp.hpp"

namespace sankhya::conic {

/// Re-derive the row multipliers and reduced costs of an optimal `solution` of `model` (a
/// model with quadratic rows and no explicit cones) by least squares on the active set at its
/// point, and keep them only when their model KKT error is smaller than the given ones'.
/// Returns true when the recovered multipliers replaced the interior point's.
bool recover_duals(const QcqpModel& model, Solution* solution);

}  // namespace sankhya::conic
