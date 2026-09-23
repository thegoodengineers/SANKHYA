// SPDX-License-Identifier: Apache-2.0
// SANKHYA - crossover: from the interior point's answer to a vertex (#219).
//
// The interior point stops strictly inside the feasible region, accurate to its tolerance,
// with no basis: fine for an objective value, useless for everything downstream - a warm
// start, sensitivity ranging, or a planner who wants to read "these units run at capacity,
// these are off" rather than a point where every variable is slightly on. Crossover is the
// standard bridge (Bixby, "Solving real-world linear programs: a decade and more of
// progress", Operations Research 50 (2002), sec. 4; Andersen & Ye, "Combining interior
// point and pivoting algorithms for constrained linear programs", Management Science 42
// (1996)): identify from the interior point which variables sit at bounds, guess a basis
// from the rest, and let the simplex pivot from that basis to an optimal vertex. Because the
// starting point is already optimal to 1e-8, the pivots are few.
//
// WHAT IS GUESSED AND WHAT IS PROVED. The classification here is a heuristic: a column is
// called nonbasic-at-lower when it is within a tolerance of its lower bound and its reduced
// cost points there, basic otherwise, and the m entries with the largest slack from their
// bounds become the basis guess. Nothing about the answer rests on that guess being right:
// the simplex installs it, repairs it if it is singular, pivots to optimality and reports its
// own vertex, which the status guard and the independent verifier then judge exactly as they
// judge any simplex answer. When the simplex does not reach an optimum inside what is left
// of the time limit, the interior point's answer stands and the message says so.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// The basis guess crossover builds from an interior point, exposed for the test that checks
/// the guess describes exactly m basic entries with every nonbasic one on a bound it has.
struct CrossoverGuess {
  std::vector<BasisStatus> col_status;
  std::vector<BasisStatus> row_status;
  Index basic = 0;     ///< entries marked basic (m for a well-formed model with a point)
  Index interior = 0;  ///< entries the interior point held strictly inside their bounds
  Index repaired = 0;  ///< structurals evicted for row logicals to make the guess full rank
};

/// Classify an interior point (`col_value`, `row_activity`, `col_dual` in the model's sense)
/// into a basis guess with exactly m basic entries.
[[nodiscard]] CrossoverGuess crossover_guess(const Model& model, const Solution& interior);

/// Push `interior` (an interior-point answer to `model`) to an optimal vertex with the
/// simplex warm-started from crossover_guess(). Returns the simplex's vertex when it reaches
/// `optimal`, with the pivot count in the message and `iterations` summed; otherwise returns
/// `interior` unchanged apart from a note saying why the vertex was not reached. `timer` is
/// the solve's clock: the pivots get what the time limit has left.
///
/// An optimal `interior` is always a start. One that is not optimal is a start only when
/// crossover_from_nonoptimal is on and crossover_start_is_usable() says so (#474); anything
/// else comes back unchanged, and a numerical error comes back without the best iterate the
/// interior point attached for this purpose (withdraw_attached_point()).
[[nodiscard]] Solution crossover_to_vertex(const Model& model, Solution interior,
                                           const Options& options, Logger& logger,
                                           SolveControl* control, const Timer& timer);

/// Whether a NON-optimal interior-point answer is a usable crossover start (#474): its status
/// is feasible, time_limit, iteration_limit or numerical_error (the last with the best
/// iterate attached), its point is finite and of the model's size, and its scaled primal and
/// dual infeasibility are both at most tol::kCrossoverStartInfeasibility. An interrupted
/// solve is never one: the caller asked it to stop.
[[nodiscard]] bool crossover_start_is_usable(const Model& model, const Solution& interior);

/// The interior point's answer followed by crossover when the options ask for it: always
/// from an optimal answer under `crossover`, and also from a non-optimal one under
/// `crossover_from_nonoptimal` (#474). The one place solve() and the registry's ipm engine
/// decide this, so the two cannot drift.
[[nodiscard]] Solution crossover_when_wanted(const Model& model, Solution interior,
                                             const Options& options, Logger& logger,
                                             SolveControl* control, const Timer& timer);

/// The options the interior point runs under when a crossover from a non-optimal answer may
/// follow (#474): with crossover_from_nonoptimal on and a finite time limit, the interior
/// point stops at (1 - crossover_time_reserve) of it, so that a time-limited answer still
/// leaves the pivots some time. Otherwise `options` unchanged.
[[nodiscard]] Options interior_point_options_before_crossover(const Options& options);

/// A numerical error carries no point (#200), but with crossover_from_nonoptimal the interior
/// point attaches its best iterate to one so the crossover can start from it. Whatever does
/// not use it withdraws it here: the vectors go back to zero, as a numerical error's are, and
/// the measured quality is recomputed. Any other status is left alone.
void withdraw_attached_point(const Model& model, Solution* solution);

}  // namespace sankhya
