// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Feasibility Jump, an LP-free primal heuristic for MILP (#506).
//
// Luteberget and Sartor, "Feasibility Jump: an LP-free Lagrangian MIP heuristic",
// Mathematical Programming Computation 15 (2023); the preprint is on luteberget.github.io.
// docs/PROVENANCE.md carries the row. Written from the paper; no solver source was read.
//
// THE IDEA. Minimise the weighted sum of the constraint violations, sum_i w_i viol_i(x), by
// moving one column at a time. For one column j with the others held, that sum is a convex
// piecewise-linear function of x_j (each row contributes a "valley" with a flat bottom where
// the row is satisfied), so the value of x_j that minimises it - the JUMP value - is found
// exactly by sorting the breakpoints and walking the slope from negative to non-negative.
// Each step evaluates the jump of a sample of candidate columns and takes the best. When no
// sampled jump decreases the weighted violation, the point is taken to be a local minimum
// and every violated row's weight gains 1: the Lagrangian step that makes the rows the search
// keeps failing to satisfy weigh more until they are satisfied.
//
// WHAT THIS IMPLEMENTATION DOES, AND WHERE IT DIFFERS FROM THE PAPER.
//   - Candidates are sampled (kFeasibilityJumpSample of them) from the columns of randomly
//     chosen violated rows, and a sample in which no jump improves is treated as the local
//     minimum. The paper maintains the full set of improving columns incrementally and
//     samples from that; the sample here is cheaper per step and can bump the weights early,
//     which only makes the weights grow sooner.
//   - The objective enters only after a first feasible point: its weight starts at 0 (pure
//     feasibility, what #506 asks for first) and gains 1 at every local minimum where no row
//     is violated, which pushes the search towards a better objective; every feasible point
//     better than the last one found is returned. For a quadratic objective the objective
//     term is left out altogether.
//   - The work is counted in nonzero visits, so a run is reproducible for a given seed and
//     budget, and is independent of the clock.
//
// A point this returns is feasible by FJ's own exact recomputation of every row activity,
// but it is a PROPOSAL: the caller sends it through BranchAndBound::offer_incumbent(),
// which checks integrality, the bounds and every row against the original model.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

struct FeasibilityJumpSettings {
  Count work_limit = tol::kFeasibilityJumpWork;  ///< nonzero visits
  std::uint64_t seed = 0;
  double feasibility_tolerance = tol::kPrimalFeasibility;  ///< absolute, on a row activity
  double integrality_tolerance = tol::kIntegrality;
  bool use_objective = true;  ///< after the first feasible point; off for a quadratic model
  /// Called with each point the moment it is found, so the caller can offer it at once
  /// rather than after the whole budget is spent; the point is kept in `points` as well.
  std::function<void(const std::vector<double>&)> on_point;
  /// Polled every kFeasibilityJumpPollWork work units; true stops the search with what it
  /// has found (an interrupt, or the time limit when the schedule is clock-based).
  std::function<bool()> should_stop;
};

struct FeasibilityJumpResult {
  /// Feasible points in the order found, each with a strictly better linear objective than
  /// the one before it.
  std::vector<std::vector<double>> points;
  Count work = 0;            ///< nonzero visits spent
  Count moves = 0;           ///< jumps taken
  Count weight_updates = 0;  ///< local minima met
};

/// Run Feasibility Jump on `model` from `start` (one value per column; it is rounded on the
/// integer columns and clamped into the bounds first).
[[nodiscard]] FeasibilityJumpResult feasibility_jump(const Model& model,
                                                     const std::vector<double>& start,
                                                     const FeasibilityJumpSettings& settings);

/// The start used before any LP: every column at the value of its box closest to zero.
[[nodiscard]] std::vector<double> feasibility_jump_zero_start(const Model& model);

}  // namespace sankhya::mip
