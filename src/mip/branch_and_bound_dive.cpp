// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: the propagating dive (mip_dive_propagate, off by default).
//
// Achterberg, "Constraint Integer Programming", thesis, TU Berlin 2007, sec. 9.2, and
// Berthold, "Primal heuristics for mixed integer programs", thesis, TU Berlin 2006, ch. 3:
// the dive of dive() (branch_and_bound_heuristics.cpp) with the three things it lacks.
//
//   propagation   every fix is followed by the node's bound propagation, propagate(), so a
//                 fix that empties a row's activity range is caught before an LP is spent on
//                 it, and the bounds it implies reach the next LP.
//   batches       after each LP the rule's ranking (rank_dive_columns) is walked and up to
//                 kDivePropagateFixShare of the fractional columns are fixed, each only while
//                 its value is still inside the box the earlier fixes propagated to, before
//                 the LP is re-solved. On a model whose LP takes seconds the dive otherwise
//                 spends its budget one column at a time (neos-950242: 7 re-solves, 48 s).
//   backtracking  an infeasible LP after a batch of several fixes undoes the batch and
//                 retries its first fix alone; an infeasible single fix, by propagation or
//                 by LP, is flipped to the other side once; kDivePropagateBacktracks such
//                 retries per dive.
//
// The dives share one budget in LP iterations, a share of the search's node LP iterations
// plus an offset (Achterberg 2007, sec. 9.2, holds diving to a quotient of the node LP
// iterations), and every re-solve's iteration limit is what is left, so a single slow LP
// cannot overrun it; their propagation passes share kPropagationDiveWork nonzero visits.
// The rules take turns: the root runs every enabled one, a node below it the next. Every bound
// a dive fixes, its own or propagated, is undone before it returns.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

Count BranchAndBound::dive_lp_budget_left() const {
  const Count node = warm_node_iterations_ + cold_node_iterations_ - dive_lp_iterations_;
  return static_cast<Count>(tol::kDivePropagateLpShare * static_cast<double>(node)) +
         tol::kDivePropagateLpOffset - dive_lp_iterations_;
}

void BranchAndBound::run_propagating_dives(Index node_index, const std::vector<double>& x,
                                           std::size_t first_slot) {
  const std::size_t runs = node_index == 0 ? kDiveRules : 1;
  std::size_t done = 0;
  for (std::size_t k = 0; k < kDiveRules && done < runs; ++k) {
    const std::size_t r = next_dive_rule_;
    next_dive_rule_ = (next_dive_rule_ + 1) % kDiveRules;
    if (!schedule_.dive[r]) continue;
    const auto rule = static_cast<DiveRule>(r);
    if (rule == DiveRule::kGuided && !have_incumbent_) continue;  // nothing to be guided by
    if (dive_lp_budget_left() <= 0) return;
    propagating_dive(first_slot + r, rule, x);
    ++done;
  }
}

void BranchAndBound::propagating_dive(std::size_t slot, DiveRule rule,
                                      const std::vector<double>& start_x) {
  HeuristicStats& s = heuristic_stats_[slot];
  const Timer clock;
  ++s.calls;
  const std::size_t mark = saved_.size();
  const std::vector<double> no_incumbent;
  const std::vector<double>& guide = have_incumbent_ ? incumbent_x_ : no_incumbent;
  Options lp_options = node_options_;
  const Count own_iteration_limit = node_options_.get_int("iteration_limit");
  std::vector<double> x = start_x;

  const auto inside = [this](std::size_t u, double v) {
    return (!is_finite_bound(working_.col_lower[u]) || v >= working_.col_lower[u] - 1e-9) &&
           (!is_finite_bound(working_.col_upper[u]) || v <= working_.col_upper[u] + 1e-9);
  };
  // Fix column u at v and propagate; false when propagation proves the box empty. All the
  // dives' propagation passes together are held to kPropagationDiveWork nonzero visits over
  // the search; past it a fix is taken unpropagated, as dive() takes every fix. Per dive it
  // was not enough: on neos-950242 (104,160 nonzeros) four root dives of 1,923 passes each
  // ran a 60 s solve to the runner's kill at 120 s (bench run 43673306061).
  const Count pass_work = std::max<Count>(1, working_.matrix.num_nonzeros());
  const auto fix = [this, pass_work](std::size_t u, double v) {
    tighten_lower(u, v);
    tighten_upper(u, v);
    if (dive_propagation_work_ >= tol::kPropagationDiveWork) return true;
    dive_propagation_work_ += pass_work;
    return propagate();
  };
  const auto stopped = [this]() {
    if (control_ != nullptr && control_->interruption_requested()) return true;
    return schedule_.seconds_budgets && limits_.time_exhausted(timer_.elapsed_seconds());
  };

  int lp_resolves = 0;
  int backtracks = 0;
  bool single = false;   // the next batch is one fix: the last batch of several was infeasible
  bool rebuild = true;   // false: re-solve the batch as it stands (it was just flipped)
  bool flipped = false;  // the batch is a flipped fix, which has no other side left
  std::size_t batch_mark = saved_.size();
  std::size_t batch_size = 0;
  DiveChoice first;  // the batch's first fix, and the LP value it was taken at
  double first_x = 0.0;

  while (lp_resolves < schedule_.dive_lp_resolves && !stopped()) {
    if (rebuild) {
      const std::vector<DiveChoice> ranked = rank_dive_columns(
          original_, locks_, integer_columns_, x, rule, guide, integrality_tolerance_);
      if (ranked.empty()) {
        // Integral on every integer column. A candidate, not a claim: offer_incumbent()
        // re-checks it against the original model.
        if (offer_from(slot, x)) {
          logger_.verbose("{} (propagating) found an incumbent at {:.10g} after {} LP(s)",
                          to_string(rule), reported(incumbent_internal_), lp_resolves);
        }
        break;
      }
      const std::size_t cap =
          single ? 1
                 : std::max<std::size_t>(
                       1, static_cast<std::size_t>(tol::kDivePropagateFixShare *
                                                   static_cast<double>(ranked.size())));
      single = false;
      flipped = false;
      batch_mark = saved_.size();
      batch_size = 0;
      bool dead = false;
      for (const DiveChoice& c : ranked) {
        if (batch_size == cap || stopped()) break;
        const auto u = static_cast<std::size_t>(c.column);
        // Propagation from this batch has fixed the column, or ruled its value out: it waits
        // for the next LP point.
        if (working_.col_lower[u] == working_.col_upper[u] || !inside(u, c.value)) continue;
        const std::size_t before = saved_.size();
        if (fix(u, c.value)) {
          if (batch_size == 0) {
            first = c;
            first_x = x[u];
          }
          ++batch_size;
          continue;
        }
        unwind_to(before);
        if (batch_size > 0) break;  // keep the batch so far; this column waits too
        // The batch's first fix is infeasible by propagation alone: the other side, once.
        const double other = c.value > x[u] ? std::floor(x[u]) : std::ceil(x[u]);
        if (backtracks >= tol::kDivePropagateBacktracks || !inside(u, other)) {
          dead = true;
          break;
        }
        ++backtracks;
        if (!fix(u, other)) {
          dead = true;
          break;
        }
        first = DiveChoice{c.column, other};
        first_x = x[u];
        batch_size = 1;
        flipped = true;
        break;
      }
      if (dead || batch_size == 0) break;
    }
    rebuild = true;

    const Count left = dive_lp_budget_left();
    if (left <= 0) break;
    lp_options.set_int("iteration_limit", std::min(own_iteration_limit, left));
    const Solution probe = solve_node_with(lp_options);
    ++lp_resolves;
    dive_lp_iterations_ += probe.iterations;
    if (probe.status == SolveStatus::kOptimal) {
      x = probe.col_value;
      // The next batch fixes more columns of THIS point, so this basis is its warm start.
      current_warm_ = basis_of(probe);
      continue;
    }
    // An iteration or time limit, or no verdict: the budget is spent or the LP is not one
    // to dive on. Only a proved-infeasible LP is worth a retry.
    if (probe.status != SolveStatus::kInfeasible || flipped) break;
    if (backtracks >= tol::kDivePropagateBacktracks) break;
    ++backtracks;
    unwind_to(batch_mark);
    if (batch_size > 1) {
      single = true;  // the same LP point, so the same ranking: its first fix alone
      continue;
    }
    const auto u = static_cast<std::size_t>(first.column);
    const double other = first.value > first_x ? std::floor(first_x) : std::ceil(first_x);
    if (!inside(u, other) || !fix(u, other)) break;
    first.value = other;
    flipped = true;
    rebuild = false;
  }
  s.work += lp_resolves;
  unwind_to(mark);
  s.seconds += clock.elapsed_seconds();
}

}  // namespace sankhya::mip
