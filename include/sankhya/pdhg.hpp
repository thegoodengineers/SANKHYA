// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted PDHG, the first-order LP engine.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::pdhg {

/// A starting point for solve_pdhg (#913 part 2): a previous solve's primal-dual pair and
/// primal weight, in the MODEL's own (unscaled) columns and rows - typically the point PDHG
/// last converged to, offered again after the model was re-solved (bounds, costs, or the
/// right-hand side moved, but the rows and columns are the ones `x` and `y` are over). The
/// engine divides by its OWN scaling before use, since the model may have changed since the
/// point was found, and projects x into its own bounds, so a point that is now outside them
/// is pulled back rather than handed to the iteration as given. `omega` seeds the primal
/// weight the first step uses; every restart after that re-derives it from the iterates
/// regardless, so an unset or non-finite `omega` only costs that first step, and `x`/`y`
/// alone are enough to ask for a warm start at all.
struct PdhgWarmStart {
  std::vector<double> x;
  std::vector<double> y;
  double omega = 1.0;

  [[nodiscard]] bool empty() const noexcept { return x.empty(); }
};

/// Solve an LP with restarted primal-dual hybrid gradient.
///
/// Requires a model with no integrality and no quadratic objective; the solve() dispatcher
/// checks that. Never throws: every failure comes back as a status.
[[nodiscard]] Solution solve_pdhg(const Model& model, const Options& options, Logger& logger,
                                  sankhya::SolveControl* control = nullptr,
                                  const PdhgWarmStart* warm_start = nullptr);

}  // namespace sankhya::pdhg
