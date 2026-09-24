// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cut selection (#415).
//
// Generating a cut is one question; adding it is another. Every cut is a row, and every row
// is paid on every node LP for the rest of the search, so a round that adds everything
// violated makes the search slower even when each cut is valid and each is violated - the
// three-way A/B at 078cb24 measured the root round at 1.049x the nodes of no cuts at all.
// Wesselmann and Suhl, "Implementing cutting plane management and selection techniques"
// (2012), and Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 8: score the
// cuts, take the best few, and skip any that is nearly parallel to one already taken.
//
//   efficacy               (a x* - b) / ||a||, the Euclidean distance from the LP point to
//                          the cut's hyperplane - what "violated" should measure, since a
//                          violation of 1 on a cut with coefficients of 1000 is nothing;
//   objective parallelism  |a . c| / (||a|| ||c||), a cut aligned with the objective moves
//                          the bound the search cares about;
//   integer support        the share of the cut's nonzeros on integer columns, a cut on the
//                          integer part of the model is the one branching cannot find;
//   orthogonality          |a . s| / (||a|| ||s||) against every cut s already selected in
//                          this round, above kCutMaxParallelism the cut waits.
//
// The weights sit in tolerances.hpp in one place. What is not selected is not thrown away:
// it comes back as `deferred` for the caller to keep for a later round, where the LP point
// has moved and the parallelism test is against a different set.
#pragma once

#include <cstddef>
#include <vector>

#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// The score a cut was selected or deferred on, with its three terms.
struct CutScore {
  double efficacy = 0.0;
  double objective_parallelism = 0.0;
  double integer_support = 0.0;
  double score = 0.0;
};

struct CutSelection {
  std::vector<Cut> selected;  ///< in the order taken: best score first
  std::vector<Cut> deferred;  ///< violated and valid, not taken this round, best first
  std::vector<CutScore> selected_scores;
};

/// The score of one cut at `point`, over `model`'s columns and objective.
[[nodiscard]] CutScore score_cut(const Model& model, const std::vector<double>& point,
                                 const Cut& cut);

/// Take at most `max_per_round` of `candidates` by score, greedily, skipping any cut whose
/// normal has a cosine above `max_parallelism` with a cut already taken. Every candidate is
/// assumed to have passed filter_and_deduplicate_cuts(); a cut that is not violated at
/// `point` scores at most zero efficacy and is taken last. `deferred` keeps at most
/// kCutWaitingLimit of the rest, best first.
[[nodiscard]] CutSelection select_cuts(const Model& model, const std::vector<double>& point,
                                       std::vector<Cut> candidates, Index max_per_round,
                                       double max_parallelism);

/// The root separation loop's row budget (#495): max(kRootCutRowFloor, kRootCutRowShare *
/// `model_rows`) cut rows in all, `model_rows` the rows before the first cut.
[[nodiscard]] Index root_cut_row_budget(Index model_rows);

/// A selection cut short to `room` rows: the first `room` of `selected` (best first, as
/// select_cuts returns them) stay; the rest go to the FRONT of `waiting`, since they scored
/// above everything selection deferred, and `waiting` is held to kCutWaitingLimit.
void take_within_budget(std::vector<Cut>* selected, std::vector<Cut>* waiting,
                        std::size_t room);

}  // namespace sankhya::mip
