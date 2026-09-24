// SPDX-License-Identifier: Apache-2.0
// SANKHYA - binary probing and the clique table (#512).
//
// References, written from the literature:
//   Savelsbergh, "Preprocessing and probing techniques for mixed integer programming
//     problems", ORSA J. Computing 6(4) (1994), sec. 4 - probing: tentatively fix a binary to
//     0 and to 1, propagate, and learn from what the two outcomes say
//   Atamturk, Nemhauser & Savelsbergh, "Conflict graphs in solving integer programming
//     problems", European J. Operational Research 121 (2000) - the conflict graph over the
//     literals x_j and 1 - x_j, and cliques in it as the inequalities it yields
//   Achterberg, Bixby, Gu, Rothberg & Weninger, "Presolve reductions in mixed integer
//     programming", INFORMS J. Computing 32(2) (2020) - probing and clique merging, with a
//     work limit because probing is the most expensive reduction
//
// WHAT A PROBE SAYS. For a binary x_j, fix x_j = 0 and propagate the row activities to a
// fixed point, then the same for x_j = 1, each from the current global bounds:
//   both infeasible   the model has no point;
//   one infeasible    x_j takes the other value in every feasible point: a FIXING;
//   both feasible     a column whose bounds came out tighter than the global ones under BOTH
//                     values is tighter in every feasible point: the union of the two boxes
//                     is a TIGHTENING; and every bound a probe moved is an IMPLICATION
//                     (x_j = v implies x_k <= u), and when it fixed another binary x_k = w,
//                     the literals x_j = v and x_k = 1 - w cannot both hold: a CONFLICT.
// Every deduction is a consequence of the rows and the box, so the feasible set is unchanged
// and no postsolve record is needed. Propagation is the activity-bound argument (Brearley,
// Mitra & Williams 1975) with integer bounds rounded inward and continuous ones loosened by a
// small margin, so a deduction is never stronger than the arithmetic supports; a probe is
// declared infeasible only when a row misses its bound by more than a scaled tolerance.
// A probe that runs out of work stops propagating: what it deduced so far is still implied,
// it has only deduced less, and it is never taken for an infeasible one.
//
// LITERALS. Literal L < n is "x_L = 1" (its value is x_L); literal n + j is "x_j = 0" (its
// value is 1 - x_j). A conflict edge (A, B) says A and B are never both true; a clique of
// the table says at most one of its literals is true: sum of their values <= 1.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::presolve {

struct ProbingResult {
  Count probed = 0;         ///< binaries probed (both values) within the work limit
  Count fixings = 0;        ///< binaries fixed because one value was infeasible
  Count tightenings = 0;    ///< bounds tightened: agreed by both probes, or propagated a fixing
  Count implications = 0;   ///< (x_j = v) implies a bound on another column, per probe
  bool infeasible = false;  ///< both values of some binary are infeasible
  bool work_limit_reached = false;  ///< candidates were left unprobed

  /// Pairwise conflicts between literals, each pair once with first < second. Capped (a
  /// missing edge only means fewer cliques, never an invalid one).
  std::vector<std::pair<Index, Index>> conflicts;
  bool conflicts_capped = false;
  /// Maximal cliques merged greedily from the conflicts; every pair in a clique is an edge.
  std::vector<std::vector<Index>> cliques;
  Count largest_clique = 0;

  /// Global bounds after every fixing and tightening, one per column. Equal to the model's
  /// own where nothing was learned. Not meaningful when `infeasible`.
  std::vector<double> col_lower;
  std::vector<double> col_upper;
};

/// Probe the binaries of `model` (integer columns with bounds [0, 1]) under the bounds
/// `col_lower` / `col_upper`, most constrained (most nonzeros) first, until every one is
/// probed or `work_limit` matrix entries have been visited. `model->matrix` must be frozen.
[[nodiscard]] ProbingResult probe_binaries(const Model& model,
                                           const std::vector<double>& col_lower,
                                           const std::vector<double>& col_upper,
                                           std::int64_t work_limit);

}  // namespace sankhya::presolve
