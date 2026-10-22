// SPDX-License-Identifier: Apache-2.0
// SANKHYA - from a first-order point to a vertex by fixing the active bounds first (#888).
//
// THE PROBLEM. The crossover (simplex/crossover.cpp) guesses a basis from a point and repairs
// its rank by refactorizing it, round after round; on the 779,640-row refinery year the
// guess from a CUDA PDHG point at relative 1e-8 had made no pivot after 409 s. A first-order
// point says more than a basis guess uses: the columns it holds at a bound, with the reduced
// cost agreeing, are the ones an optimal vertex holds there too, up to the few the point got
// wrong. Fixing them leaves an LP over the columns the point holds strictly inside their
// bounds, which is a fraction of the model, and the dual simplex solves that with presolve
// in a fraction of the time. Its vertex, with the fixed columns released at the bound they
// were fixed to, is an exact primal-feasible vertex of the original model near the optimal
// face; the existing crossover then starts from a point that IS a vertex - the basis guess
// is that vertex's basis, there is nothing to push, and the primal loop only has to pivot
// in whichever fixed columns carry a wrong-signed reduced cost.
//
// Bixby & Saltzman, "Recovering an optimal LP basis from an interior point solution",
// Operations Research Letters 15 (1994) - identify the optimal face from the point, then
// pivot; Megiddo, "On finding primal- and dual-optimal bases", ORSA J. Computing 3 (1991);
// Applegate, Hinder, Lu & Lubin, arXiv:2501.07018 (2025), for the reading of a first-order
// point as a classification of which bounds are active (their feasibility polishing is the
// same reading in the other direction). Nothing here trusts the classification: the
// restricted LP can be infeasible when a column was fixed wrongly, and then the plain
// crossover runs from the first-order point as before.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// `first` is PDHG's answer to `model` (its point, duals and reduced costs). Returns the
/// crossover's vertex when the route reaches one, otherwise `first` with a note; the caller
/// then falls back to the plain crossover. `timer` is the solve's clock and the time limit
/// bounds the restricted solve and the crossover alike.
[[nodiscard]] Solution fix_and_solve_then_crossover(const Model& model, Solution first,
                                                    const Options& options, Logger& logger,
                                                    SolveControl* control, const Timer& timer);

/// How many columns the route would fix at a bound from this point (the classification of
/// fix_and_solve_then_crossover, exposed for the test): at a finite bound within 1e-6 of
/// its own size, with the reduced cost in the model's sense not pushing away from it.
[[nodiscard]] Index count_fixable_columns(const Model& model, const Solution& first);

}  // namespace sankhya
