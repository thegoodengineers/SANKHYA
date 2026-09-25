// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound (#502): strong-branch fixing, incremental propagation to a
// fixpoint, and the heap open list. Each is behind its own option and OFF by default until
// an A/B on main says otherwise; with all three off the search is exactly the one in
// branch_and_bound.cpp and branch_and_bound_node.cpp.
//
// References, written from the literature:
//   Achterberg, Koch & Martin, "Branching rules revisited", Operations Research Letters 33
//     (2005), 42-54 - strong branching, and what an infeasible strong-branching child proves
//   Achterberg, "Constraint Integer Programming" (thesis, TU Berlin, 2007), ch. 7 (domain
//     propagation to a fixpoint driven by the bounds that changed) and ch. 6 (node
//     selection over a priority queue)
//   Savelsbergh, "Preprocessing and probing for mixed integer programming problems",
//     ORSA J. Computing 6(4), 1994 - bound propagation from row activities

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

#include "../util/profiler.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

// ---- The heap open list -----------------------------------------------------------------
//
// Best-bound and best-estimate take the open node with the smallest key, ties to the
// smallest index. The linear scan in take_next_open_node() finds it in O(n) per node, which
// on a tree of 10^5 open nodes is most of the node's cost. A binary heap on the same key and
// the same tie-break yields the same node in O(log n): the order of the search, and with it
// every result, is unchanged. That equivalence is what the tests pin.

void BranchAndBound::push_open(Index node_index) {
  open_.push_back(node_index);
  if (open_is_heap()) {
    std::push_heap(open_.begin(), open_.end(),
                   [this](Index a, Index b) { return open_after(a, b); });
  }
}

void BranchAndBound::rebuild_open_heap() {
  if (open_is_heap()) {
    std::make_heap(open_.begin(), open_.end(),
                   [this](Index a, Index b) { return open_after(a, b); });
  }
}

// ---- Incremental propagation to a fixpoint ---------------------------------------------
//
// The default propagate() sweeps every row up to three times. A bound that moves in the
// last sweep is not followed, and a row nothing touched is read again anyway. Here the rows
// are a worklist: every row once, and after that only the rows of a column whose bound just
// moved - which saved_ records, since every tightening goes through tighten_lower/upper -
// until nothing moves or the work budget is spent. Stopping at the budget is sound:
// propagation only ever tightens to implied bounds, so less of it is weaker, never wrong.

bool BranchAndBound::propagate_to_fixpoint(const CsrView& by_row) {
  const Index rows = working_.num_rows();
  const Index cols = working_.num_cols();
  const std::int64_t budget =
      std::max<std::int64_t>(tol::kPropagationWorkFloor,
                             static_cast<std::int64_t>(tol::kPropagationWorkPerRow) * rows);

  std::vector<char> queued(static_cast<std::size_t>(rows), 1);
  std::deque<Index> queue;
  for (Index i = 0; i < rows; ++i) queue.push_back(i);

  // Queue the rows of every column tightened since `seen`.
  std::size_t seen = saved_.size();
  const auto requeue = [&] {
    for (; seen < saved_.size(); ++seen) {
      const Index column = saved_[seen].column;
      if (column < 0 || column >= cols) continue;
      const ColumnView entries = working_.matrix.column(column);
      for (Index p = 0; p < entries.size; ++p) {
        const auto r = static_cast<std::size_t>(entries.rows[p]);
        if (queued[r] == 0) {
          queued[r] = 1;
          queue.push_back(entries.rows[p]);
        }
      }
    }
  };

  std::int64_t work = 0;
  bool conflicts_due = !conflicts_.entries().empty();
  for (;;) {
    // Learned conflicts (#292) first, as in the sweeps, and again after the rows moved
    // anything: a bound a conflict implies feeds the rows, and the other way round.
    if (conflicts_due) {
      bool changed = false;
      if (!propagate_conflicts(&changed)) return false;
      requeue();
      conflicts_due = false;
    }
    if (queue.empty() || work >= budget) break;
    while (!queue.empty() && work < budget) {
      const Index i = queue.front();
      queue.pop_front();
      queued[static_cast<std::size_t>(i)] = 0;
      ++work;
      ++propagated_rows_;
      const std::size_t before = saved_.size();
      bool changed = false;
      if (!propagate_row(i, by_row, &changed)) return false;
      if (saved_.size() != before) {
        requeue();
        conflicts_due = !conflicts_.entries().empty();
      }
    }
  }
  return true;
}

// ---- Strong-branch fixing ---------------------------------------------------------------
//
// A strong-branching probe that comes back infeasible (Achterberg, Koch & Martin 2005) proves
// that side of the column holds no feasible point of the node - and so of any node below it.
// The plain rule only scores that column highest and branches on it, which spends a child on
// the closed side and leaves every other decision taken on an LP point that still sits in it.
//
// Here the column is FIXED to its open side, the node LP is re-solved, and the decision is
// taken again on the new point, for up to kStrongBranchFixRounds rounds. The fixes must
// then hold for the whole subtree, not only for this decision: a node stores one bound
// change, and enter() rebuilds a node's domain from its chain to the root, so a fix left
// only in working_ would be undone by leave() and the children would lose it - their LPs
// would sit in the closed side again and the pseudocost gains measured against the
// re-solved bound would be wrong. link_strong_fixes() therefore hangs the fixes as a chain
// of never-opened links, the device split_integral_node() already uses, and the children
// are created under its tail.

namespace {
// A certificate (#518) proves a leaf from its branching disjunctions; a bound derived from an
// infeasible probe is not one of them, so the tree it would describe is not the one searched.
constexpr const char* kFixNotCertified =
    "strong-branch fixing (mip_strong_branch_fix) fixed a column from an infeasible probe, "
    "which the certificate cannot express; run with mip_strong_branch_fix=false";
}  // namespace

Index BranchAndBound::select_branching_column(Solution& relaxation, double& node_bound,
                                              double& prune_bound) {
  strong_fixes_.clear();
  for (int round = 0;; ++round) {
    const bool may_fix = strong_branch_fix_ && round < tol::kStrongBranchFixRounds;
    // #520: the batched children start from this relaxation's duals, in minimise space.
    if (batch_strong_) {
      batch_branch_duals_.resize(relaxation.row_dual.size());
      for (std::size_t i = 0; i < batch_branch_duals_.size(); ++i) {
        batch_branch_duals_[i] = sense_ * relaxation.row_dual[i];
      }
    }
    std::vector<DomainChange> fixes;
    const Index column =
        choose_branching_column(relaxation.col_value, node_bound, may_fix ? &fixes : nullptr);
    if (column == kBranchPruned) {
      ++strong_branch_fix_prunes_;
      certificate_refuse(kFixNotCertified);
      return kBranchPruned;
    }
    if (fixes.empty() || column < 0) return column;
    certificate_refuse(kFixNotCertified);

    // choose_branching_column() left current_warm_ at the node's basis. The fixes move
    // bounds only, so it stays dual feasible for the re-solve.
    const WarmStart node_basis = current_warm_;
    const std::size_t mark = saved_.size();
    for (const DomainChange& fix : fixes) {
      const auto u = static_cast<std::size_t>(fix.column);
      if (fix.is_upper) {
        tighten_upper(u, fix.value);
      } else {
        tighten_lower(u, fix.value);
      }
    }
    // The full node options, not the probes' iteration cap: this is the node's LP now.
    Solution fixed = solve_node();
    ++strong_branch_solves_;
    strong_branch_iterations_ += fixed.iterations;
    if (fixed.status == SolveStatus::kInfeasible) {
      // The fixes are implied by the node, so the node itself is infeasible.
      ++strong_branch_fix_prunes_;
      return kBranchPruned;
    }
    if (fixed.status != SolveStatus::kOptimal) {
      // No point to branch on. Undo this round and branch as the option-off rule does, on
      // the column already scored: its closed side is then a child that prunes at once.
      unwind_to(mark);
      current_warm_ = node_basis;
      return column;
    }

    strong_fixes_.insert(strong_fixes_.end(), fixes.begin(), fixes.end());
    strong_branch_fixes_ += static_cast<Count>(fixes.size());
    relaxation = std::move(fixed);
    node_bound = internal_objective(relaxation.col_value);
    prune_bound = prune_bound_of(relaxation, node_bound);
    current_warm_ = basis_of(relaxation);
    if (can_prune(prune_bound)) {
      ++strong_branch_fix_prunes_;
      return kBranchPruned;
    }
    if (most_fractional(relaxation.col_value) < 0) return kBranchIntegral;
  }
}

Index BranchAndBound::link_strong_fixes(Index node_index, double bound) {
  Index tail = node_index;
  const Index depth = nodes_[static_cast<std::size_t>(node_index)].depth;
  for (const DomainChange& fix : strong_fixes_) {
    TreeNode link;  // never opened, never solved: it carries a bound change and nothing else
    link.parent = tail;
    link.has_change = true;
    link.change = fix;
    link.bound = bound;
    link.depth = depth;
    nodes_.push_back(std::move(link));
    tail = static_cast<Index>(nodes_.size() - 1);
  }
  return tail;
}

void BranchAndBound::report_branching_fixpoint() const {
  if (strong_branch_fix_) {
    logger_.info("Strong-branch fixing (#502): {} column(s) fixed, {} node(s) fathomed by it",
                 strong_branch_fixes_, strong_branch_fix_prunes_);
  }
  Profiler* profiler = logger_.profiler();
  if (profiler == nullptr) return;
  // So an A/B can tell "no effect" from "never ran".
  if (strong_branch_fix_) {
    profiler->count("strong-branch columns fixed",
                    static_cast<std::int64_t>(strong_branch_fixes_));
    profiler->count("strong-branch fix prunes",
                    static_cast<std::int64_t>(strong_branch_fix_prunes_));
  }
  if (incremental_propagation_) {
    profiler->count("propagation rows processed", static_cast<std::int64_t>(propagated_rows_));
  }
  if (heap_open_list_) {
    profiler->count("open-list heap selections", static_cast<std::int64_t>(heap_selections_));
  }
}

}  // namespace sankhya::mip
