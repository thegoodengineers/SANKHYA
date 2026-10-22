// SPDX-License-Identifier: Apache-2.0
// SANKHYA - from a first-order point to a vertex by fixing the active bounds first (#888).
//
// THE PROBLEM. The crossover (simplex/crossover.cpp) guesses a basis from a point and repairs
// its rank by refactorizing it, round after round; on the 779,640-row refinery year the
// guess from a CUDA PDHG point at relative 1e-8 had made no pivot after 409 s, and on the
// 32,485-row year at T = 365 the guess was 61% interior, 9,457 of its 29,565 columns were
// evicted for rank, and the basis that came out reproduced the point to 1.9e+17: a
// first-order point is not centred the way an interior point is, so the guess that works
// from a barrier method does not work from it. A first-order point says more than a basis
// guess uses: the columns it holds at a bound, with the reduced cost agreeing, and the rows
// whose activity sits on a bound, are the ones an optimal vertex holds there too, up to the
// few the point got wrong. Fixing them leaves an LP over the entries the point holds strictly
// inside their bounds, which presolve shrinks further, and the dual simplex solves that from
// its slack basis. Its vertex, with the fixed columns and rows released at the bound they
// were fixed to, is an exact primal-feasible vertex of the original model near the optimal
// face, with a basis in hand; the crossover's push phase starts from that basis - nothing to
// push, since every nonbasic entry is on a bound - and its primal loop only has to pivot in
// whichever released entry carries a wrong-signed reduced cost.
//
// Bixby & Saltzman, "Recovering an optimal LP basis from an interior point solution",
// Operations Research Letters 15 (1994) - identify the optimal face from the point, then
// pivot; Megiddo, "On finding primal- and dual-optimal bases", ORSA J. Computing 3 (1991);
// Andersen & Ye, "Combining interior point and pivoting algorithms for constrained linear
// programs", Management Science 42 (1996); Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue &
// Schudy, "Practical large-scale linear programming using primal-dual hybrid gradient",
// NeurIPS 2021 (PDLP), for the first-order point itself; Ge, Wang, Xiong & Ye, "From an
// interior point to a crossover" (2021), for the reading of a first-order point as a
// classification of which bounds are active. Nothing here trusts the classification: the
// restricted LP can be infeasible when an entry was fixed wrongly, and then the plain
// crossover runs from the first-order point as before.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// What the route fixes from a point, exposed for the test: `col_fix[j]` is kAtLower or
/// kAtUpper for a column fixed at that bound and kUnknown for one left free; `row_fix`
/// likewise for a row snapped to an equality at that bound.
struct FixAndSolveClassification {
  std::vector<BasisStatus> col_fix;
  std::vector<BasisStatus> row_fix;
  Index cols_fixed = 0;
  Index rows_snapped = 0;
};

/// Classify `first`'s point: a column within `fraction` (relative to its own size) of a
/// finite bound, with the reduced cost in the model's sense not pushing away from it, is
/// fixed there; a row whose activity is within `fraction` of a bound is snapped to it.
/// Empty vectors when `first` holds no point of the model's size.
[[nodiscard]] FixAndSolveClassification classify_for_fix_and_solve(const Model& model,
                                                                   const Solution& first,
                                                                   double fraction);

/// `first` is PDHG's answer to `model` (its point, duals and reduced costs). Returns the
/// crossover's vertex when the route reaches one, otherwise `first` with a note; the caller
/// then falls back to the plain crossover. `timer` is the solve's clock and the time limit
/// bounds the restricted solve and the crossover alike.
[[nodiscard]] Solution fix_and_solve_then_crossover(const Model& model, Solution first,
                                                    const Options& options, Logger& logger,
                                                    SolveControl* control, const Timer& timer);

}  // namespace sankhya
