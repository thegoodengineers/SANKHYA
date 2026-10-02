// SPDX-License-Identifier: Apache-2.0
// SANKHYA - what the branch and bound enforces beyond integrality (#754): semi-continuous
// columns and special ordered sets, taken off the model before the search sees it.
//
// THE SEARCH RUNS ON A RELAXATION. take_from() lowers every semi-continuous column's lower
// bound to 0 and clears both fields, so the model the tree, the cuts, the propagation and
// every heuristic read is a plain MILP (or LP) whose feasible set CONTAINS the true one. All
// of them are then valid as they stand: a cut, a propagated bound or a pruned node that is
// valid for the relaxation is valid for any subset of it. What is left to enforce is held
// here, and enforced in exactly two places (branch_and_bound_sos.cpp): an incumbent is
// accepted only when it satisfies every condition, and a node whose relaxation breaks one is
// branched on it before any integer column.
#pragma once

#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::mip {

struct ScSosSpec {
  std::vector<Index> sc_columns;  ///< semi-continuous columns
  std::vector<double> sc_lower;   ///< and the lower end l of each one's run range
  std::vector<SosSet> sets;

  [[nodiscard]] bool empty() const noexcept { return sc_columns.empty() && sets.empty(); }
  [[nodiscard]] Index num_items() const noexcept {
    return static_cast<Index>(sc_columns.size() + sets.size());
  }

  /// Move the two fields out of `model` and relax it: each semi-continuous column's lower
  /// bound becomes 0 (validate() guarantees l >= 0, so 0 is the smaller of the two).
  [[nodiscard]] static ScSosSpec take_from(Model* model);

  /// The largest violation over every condition at `x`, as core/sc_sos.hpp measures it.
  [[nodiscard]] double violation(const std::vector<double>& x) const;
};

}  // namespace sankhya::mip
