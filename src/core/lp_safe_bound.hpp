// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the safe bound and certified gap of an optimal LP (#763).
//
// Every LP solve() reports optimal gets the Neumaier-Shcherbina bound (safe_bound.hpp) of
// its own row duals, and the gap from the objective to it. Reference: A. Neumaier and
// O. Shcherbina, "Safe bounds in linear and mixed-integer linear programming", Mathematical
// Programming 99 (2004) 283-296.
#pragma once

#include "sankhya/model.hpp"

namespace sankhya {

/// Fill solution->safe_lower_bound, certified_gap, certified_relative_gap and
/// safe_multipliers (model.hpp) when `model` is an LP and the solution is kOptimal with one
/// dual per row; leave them untouched otherwise.
void attach_safe_lower_bound(const Model& model, Solution* solution);

}  // namespace sankhya
