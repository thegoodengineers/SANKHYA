// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: reduced-cost fixing at the root, and restarts on what it fixed
// (#418).
//
// REDUCED-COST FIXING. Nemhauser and Wolsey, "Integer and Combinatorial Optimization"
// (1988), and Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 7: at the root
// relaxation's optimum, a column nonbasic at its lower bound with reduced cost d > 0 costs at
// least d per unit it moves up - LP duality says every point of the root polytope with
// x_j = l_j + k has objective at least z_root + d k, and every integer solution is such a
// point. Once an incumbent z* exists, any k with z_root + d k >= z* cannot belong to a better
// solution, so the column's upper bound can be brought down to l_j + k_max for the whole tree;
// symmetrically at the upper bound with d < 0. It is a global tightening, it costs O(n), and
// it gets stronger every time the incumbent improves, which is when it is re-run.
//
// RESTARTS. Achterberg 2007, ch. 10, and Achterberg and Wunderling, "Mixed integer
// programming: analyzing 12 years of progress" (2013): after the root has been processed the
// search learns things the root never saw - here, columns fixed by the reduced costs once
// the heuristics found an incumbent. Rather than carry a tree built on the looser bounds, the
// search throws it away and re-solves the root on the tightened ones: the root bound itself
// moves, and the tree that follows is built on a better branching order. Pseudocosts, cut
// rows, learned conflicts and the incumbent are kept; only the node list goes. The trigger
// is the standard one: enough of the integer columns fixed since the root was last
// processed, at most mip_restarts times, and never after mip_restart_node_limit nodes.
//
// Both are between-node operations: no node is entered, so working_'s bounds ARE the global
// bounds and a write to them is a write to the whole tree, which is the one place the rule
// "every bound change goes through tighten_*()" does not apply and says so.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

void BranchAndBound::remember_root_relaxation(const Solution& relaxation) {
  if (!reduced_cost_fixing_ || quadratic_) return;
  const auto n = static_cast<std::size_t>(original_.num_cols());
  if (relaxation.col_dual.size() != n || relaxation.col_status.size() != n) return;
  // The reduced costs in minimise space, whatever the model's sense: col_dual carries the
  // model's own sign (see ranging.cpp), and the search reasons in minimise space throughout.
  root_reduced_.resize(n);
  for (std::size_t u = 0; u < n; ++u) root_reduced_[u] = sense_ * relaxation.col_dual[u];
  root_status_ = relaxation.col_status;
  fixing_incumbent_ = std::numeric_limits<double>::infinity();  // this root: nothing used yet
}

Count BranchAndBound::fix_by_reduced_cost() {
  if (!reduced_cost_fixing_ || !have_incumbent_ || root_reduced_.empty()) return 0;
  if (!saved_.empty()) return 0;  // a node is entered: not the global bounds
  // The pool wants the alternatives fixing would remove (#225): an incumbent-relative pool
  // gap or a complete pool asks for points no better than the incumbent, and reduced-cost
  // fixing exists to discard exactly those.
  if (pool_complete_ || pool_gap_ < kNoPoolGap) return 0;
  if (incumbent_internal_ >= fixing_incumbent_) return 0;  // nothing new since the last pass
  fixing_incumbent_ = incumbent_internal_;
  if (std::isnan(root_bound_after_cuts_internal_)) return 0;
  // What a better solution must beat, measured the way can_prune() measures it: the
  // incumbent less the absolute gap target. A column may move k units only while
  // z_root + d k stays below that; the first k at or past it is what the tree would prune
  // anyway, so fixing here removes nothing pruning would keep.
  const double room =
      incumbent_internal_ - absolute_gap_target_ - root_bound_after_cuts_internal_;
  if (!std::isfinite(room)) return 0;
  Count moved = 0;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    const double d = root_reduced_[u];
    if (root_status_[u] == BasisStatus::kAtLower && d > tol::kDualFeasibility &&
        is_finite_bound(global_lower_[u])) {
      // Units the column may still move up: the last k with z_root + d k < room's line,
      // computed conservatively (the 1e-9 keeps a k that sits on the line by rounding).
      const double k_max = std::max(0.0, std::ceil(room / d - 1e-9) - 1.0);
      const double new_upper = global_lower_[u] + k_max;
      if (new_upper < global_upper_[u] - 0.5) {  // integral bounds: a move is a whole unit
        global_upper_[u] = new_upper;
        working_.col_upper[u] = new_upper;
        ++moved;
        if (new_upper == global_lower_[u]) ++fixed_since_root_;
      }
    } else if (root_status_[u] == BasisStatus::kAtUpper && d < -tol::kDualFeasibility &&
               is_finite_bound(global_upper_[u])) {
      const double k_max = std::max(0.0, std::ceil(room / -d - 1e-9) - 1.0);
      const double new_lower = global_upper_[u] - k_max;
      if (new_lower > global_lower_[u] + 0.5) {
        global_lower_[u] = new_lower;
        working_.col_lower[u] = new_lower;
        ++moved;
        if (new_lower == global_upper_[u]) ++fixed_since_root_;
      }
    }
  }
  reduced_cost_fixings_ += moved;
  if (moved > 0) {
    logger_.verbose(
        "reduced-cost fixing: {} bound(s) tightened against the incumbent {:.10g}, "
        "{} integer column(s) fixed since the root",
        moved, reported(incumbent_internal_), fixed_since_root_);
  }
  return moved;
}

bool BranchAndBound::restart_due() const {
  // One search only: a parallel worker's tree belongs to the shared search (#222), and a
  // subtree planted by another worker has no root of its own to re-solve.
  if (restarts_ >= restarts_allowed_ || shared_ != nullptr || seed_ != nullptr) return false;
  if (nodes_explored_ > restart_node_limit_ || fixed_since_root_ == 0) return false;
  return static_cast<double>(fixed_since_root_) >=
         restart_fraction_ * static_cast<double>(integer_columns_.size());
}

void BranchAndBound::restart_search() {
  ++restarts_;
  logger_.info(
      "Restart {}: {} of {} integer columns fixed by reduced cost since the root; the {} open "
      "node(s) are discarded and the root re-solved on the tightened bounds",
      restarts_, fixed_since_root_, integer_columns_.size(), open_.size());
  // The tree goes; the search's learning stays: pseudocosts, cut rows, conflicts, the
  // incumbent and the heuristics' statistics are all untouched here.
  nodes_.clear();
  open_.clear();
  saved_.clear();
  // Built in place rather than copied in: GCC 13 at -O2 reports a null dereference inside
  // the copy of a node whose warm-start vectors are empty, a false positive -Werror turns
  // into a failed build.
  nodes_.emplace_back();
  nodes_.back().bound = -std::numeric_limits<double>::infinity();
  open_.push_back(0);
  current_warm_ = WarmStart{};  // the slack basis: the new root is solved from scratch
  deepest_node_ = 0;
  fixed_since_root_ = 0;
  // The new root's relaxation is the material for the next fixing pass; the old one's
  // reduced costs were valid for looser bounds and are not reused.
  root_reduced_.clear();
  root_status_.clear();
}

}  // namespace sankhya::mip
