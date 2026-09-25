// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MIP primal heuristics on first-order LP relaxations (#509): a feasibility pump
// whose projection step is a PDHG solve, and fix-and-propagate. Written from the papers
// cited below; no solver source was read.
//
//   feasibility pump   Fischetti, Glover and Lodi, "The feasibility pump", Math. Programming
//                      104 (2005): round the LP point, project the rounding back onto the LP
//                      polytope in the L1 distance, repeat until the two meet; a rounding seen
//                      the round before is perturbed by flipping the T/2..3T/2 columns furthest
//                      from their rounding, one seen within the last R rounds by the random
//                      restart of their section 3. The general-integer distance takes one
//                      auxiliary column per column (Bertacco, Fischetti and Lodi, "A
//                      feasibility pump heuristic for general mixed-integer problems", Discrete
//                      Optimization 4, 2007). The projection is solved by restarted PDHG, on
//                      the device when there is one - the first-order projection of Mexi et
//                      al., arXiv:2307.03466, and of the GPU MIP heuristics of Corduk et al.,
//                      arXiv:2510.20499.
//   fix-and-propagate  Gamrath, Berthold, Heinz and Winkler, "Structure-based primal
//                      heuristics for mixed integer programming", Optimization in the Real
//                      World, Springer 2016, and the fix-propagate-and-repair scheme of
//                      Corduk et al.:
//                      fix the integer columns in order of the relaxation's fractionality
//                      (least fractional first) to the integer nearest the relaxation inside
//                      the column's CURRENT domain, propagate the rows after each batch of
//                      fixes (the synchronous propagator of domain_propagation.hpp, on the
//                      device through src/gpu/domain_prop.cu when there is one), back up a
//                      bounded number of times on a proved-empty box, and repair what is left
//                      with Feasibility Jump (#506).
//
// NOTHING HERE DECIDES AN ANSWER. A PDHG point is a guide, accurate to its relative tolerance
// and no further; the only points these functions return are integral on the integer columns,
// inside the ORIGINAL bounds and satisfy every ORIGINAL row to tol::kPrimalFeasibility, by
// point_is_feasible() below - and the search re-checks each one in offer_incumbent() before it
// can become the incumbent. The continuous columns of a mixed model are never taken from a
// PDHG point: once the integer columns are fixed they are completed by an LP solved with
// `completion_options` (the search passes its node simplex), whose point is then checked.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

struct PdhgHeuristicSettings {
  /// Run the PDHG solves and the propagation on the CUDA device when the build has the
  /// backend and a card answers (gpu_heur_backend=auto); false forces the CPU (=cpu).
  bool use_device = true;
  int pump_rounds = 50;  ///< projections at most (gpu_pump_max_iter)
  int backtracks = 5;    ///< fix-and-propagate back-ups at most (gpu_fix_backtrack)
  Count pdhg_iterations = tol::kPdhgHeuristicIterations;  ///< per PDHG solve
  /// Seconds one run may spend in all, negative for none: every PDHG and completion solve is
  /// limited to what is left of it, and the run stops between rounds once it is spent.
  double seconds = -1.0;
  double integrality_tolerance = tol::kIntegrality;
  std::uint64_t seed = 0;
  Count repair_work = tol::kFixPropRepairWork;  ///< Feasibility Jump's budget; 0: no repair
  /// What an LP with every integer column fixed is completed with. The caller sets its engine
  /// and limits; it must be an engine whose optimal point meets tol::kPrimalFeasibility.
  Options completion_options;
  /// Polled between rounds; true stops the heuristic with what it has.
  std::function<bool()> should_stop;
};

struct PdhgHeuristicResult {
  /// A point that passed point_is_feasible() against the model, or empty.
  std::vector<double> x;
  int rounds = 0;          ///< pump rounds, or fix-and-propagate batches attempted
  Count pdhg_solves = 0;   ///< PDHG solves: projections, and the relaxation when none given
  Count completions = 0;   ///< LPs solved to complete the continuous columns
  Count propagations = 0;  ///< propagation calls
  int backtracks = 0;      ///< fix-and-propagate back-ups taken
  int perturbations = 0;   ///< pump flips (one-cycles) and restarts (longer cycles)
  bool on_device = false;  ///< some of the work ran on the CUDA device
  bool repaired = false;   ///< the point came from the Feasibility Jump repair
  std::string stopped;     ///< why it stopped, for the log
};

/// The independent check every returned point passes: integral to `integrality` on the integer
/// columns, inside the column bounds and every row to tol::kPrimalFeasibility (absolute), the
/// same test BranchAndBound::offer_incumbent() applies.
[[nodiscard]] bool point_is_feasible(const Model& model,
                                     const std::vector<Index>& integer_columns,
                                     const std::vector<double>& x, double integrality);

/// The feasibility pump from `start` (the root relaxation; empty: solved here with PDHG).
[[nodiscard]] PdhgHeuristicResult pdhg_feasibility_pump(
    const Model& model, const std::vector<Index>& integer_columns,
    const std::vector<double>& start, const PdhgHeuristicSettings& settings);

/// Fix-and-propagate from `start` (the root relaxation; empty: solved here with PDHG).
[[nodiscard]] PdhgHeuristicResult fix_and_propagate(const Model& model,
                                                    const std::vector<Index>& integer_columns,
                                                    const std::vector<double>& start,
                                                    const PdhgHeuristicSettings& settings);

}  // namespace sankhya::mip
