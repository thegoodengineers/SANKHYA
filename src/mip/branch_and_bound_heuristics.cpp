// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: scheduling the primal heuristics and accounting for them
// (#290, #414). The heuristics themselves are pure functions in heuristics.cpp; this file
// decides when each runs, with what budget, and offers what they propose to
// offer_incumbent(), which is the only way anything becomes the incumbent.
//
// EVERY HEURISTIC HAS ITS OWN SWITCH (#414). #378 put four heuristics behind one flag, and
// the measurement that followed could say only that the four together did not pay - not
// which one cost the nodes. Each now resolves its own auto/on/off against mip_heuristics
// (HeuristicSchedule), so a measurement can run exactly one of them, and each spends a
// budget the search counts - LP re-solves, sub-MIP nodes, repair moves - so deterministic
// mode reproduces it. The sub-MIP heuristics keep a budget in seconds as well, except under
// deterministic=true, where only the counted budgets apply: the rule #288 set for the
// interior-point polish.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

namespace {
/// Where each heuristic's row lives in heuristic_stats_. The dive rules are contiguous, in
/// DiveRule order, so a rule's slot is kDiveFractional plus its index.
enum Slot : std::size_t {
  kRounding,
  kLockRounding,
  kRepair,
  kDiveFractional,
  kDiveCoefficient,
  kDiveVectorLength,
  kDiveGuided,
  kPump,
  kRins,
  kRens,
  kSlots
};
constexpr const char* kNames[kSlots] = {"rounding",
                                        "lock rounding",
                                        "repair",
                                        "fractional diving",
                                        "coefficient diving",
                                        "vector length diving",
                                        "guided diving",
                                        "feasibility pump",
                                        "RINS",
                                        "RENS"};
static_assert(kDiveGuided - kDiveFractional + 1 == kDiveRules);

/// The options a sub-MIP (RINS, RENS) is solved with: the search's own, quiet, capped at
/// `node_limit` nodes, no pool, none of the search's files, one thread, and no sub-MIP
/// heuristics of its own - a user's explicit `on` must not recurse. A `time_limit` below
/// zero sets none.
Options sub_mip_options(const Options& base, Count node_limit, double time_limit) {
  Options sub = base;
  sub.set_bool("log_to_console", false);
  sub.set_bool("mip_heuristics", false);
  for (const char* name : {"mip_heur_lock_rounding", "mip_heur_repair", "mip_heur_pump",
                           "mip_heur_rins", "mip_heur_rens", "mip_heur_dive_coefficient",
                           "mip_heur_dive_vector_length", "mip_heur_dive_guided"}) {
    sub.set_string(name, "off");
  }
  sub.set_int("mip_dive_frequency", 0);
  sub.set_int("node_limit", node_limit);
  sub.set_bool("pool_complete", false);
  sub.set_int("pool_size", 1);
  // The search's checkpoint is the search's (#287): a sub-MIP stopped at its node cap would
  // otherwise write its own tree over the file, and after a resume every sub-MIP would try
  // to load the search's file and be refused as a different model. The same for the other
  // file-valued options (#368, #223, #292): a sub-MIP's profile, progress stream and
  // conflicts are not the search's.
  sub.set_string("checkpoint", "");
  sub.set_string("resume", "");
  sub.set_string("profile_out", "");
  sub.set_string("progress_out", "");
  sub.set_string("conflict_out", "");
  // A sub-MIP is one worker's business (#222): it must not start a parallel search of its
  // own inside a tree worker.
  sub.set_int("mip_threads", 1);
  if (time_limit >= 0.0) sub.set_double("time_limit", time_limit);
  return sub;
}

/// The seconds one sub-MIP may take: a share of what remains of the search's time limit, or
/// none (negative) when there is no limit or the solve is deterministic.
double sub_mip_seconds(const ResourceLimits& limits, double elapsed, bool seconds_budgets) {
  if (!seconds_budgets || !limits.has_time_limit()) return -1.0;
  return std::max(0.0, tol::kSubMipBudgetShare * limits.remaining_seconds(elapsed));
}
}  // namespace

void BranchAndBound::init_heuristics() {
  heuristic_stats_.assign(kSlots, HeuristicStats{});
  for (std::size_t k = 0; k < kSlots; ++k) heuristic_stats_[k].name = kNames[k];
  schedule_ = HeuristicSchedule::from(options_);
  if (schedule_.lock_rounding || schedule_.repair ||
      schedule_.dive[static_cast<std::size_t>(DiveRule::kCoefficient)]) {
    locks_ = compute_locks(original_);
  }
}

bool BranchAndBound::offer_from(std::size_t slot, const std::vector<double>& x) {
  // offer_incumbent() accepts only a point that is integral, feasible against the ORIGINAL
  // model AND better than the incumbent, so "proposed" and "improved" are the two numbers
  // that can differ: what the heuristic produced, and what survived the check.
  HeuristicStats& s = heuristic_stats_[slot];
  ++s.found;
  if (!offer_incumbent(x)) return false;
  ++s.improved;
  return true;
}

void BranchAndBound::run_node_heuristics(Index node_index, const Solution& relaxation) {
  const std::vector<double>& x = relaxation.col_value;
  {
    // The rounding every node always had, now counted.
    HeuristicStats& s = heuristic_stats_[kRounding];
    const Timer clock;
    ++s.calls;
    const double before = incumbent_internal_;
    const bool had = have_incumbent_;
    try_rounding(x);
    ++s.found;  // try_rounding always proposes its rounding
    if (have_incumbent_ && (!had || incumbent_internal_ < before)) ++s.improved;
    s.seconds += clock.elapsed_seconds();
  }

  // LOCK ROUNDING, every node: O(nonzeros), no LP, and it succeeds exactly where nearest
  // rounding fails for the reason locks see - a covering row rounded the wrong way.
  if (schedule_.lock_rounding) {
    HeuristicStats& s = heuristic_stats_[kLockRounding];
    const Timer clock;
    ++s.calls;
    (void)offer_from(kLockRounding, lock_round(original_, locks_, integer_columns_, x));
    s.seconds += clock.elapsed_seconds();
  }

  // REPAIR, at the root only: it is the most expensive of the no-LP heuristics, and its
  // value is an early incumbent, which only matters before the tree has found one.
  if (schedule_.repair && node_index == 0 && !have_incumbent_) {
    HeuristicStats& s = heuristic_stats_[kRepair];
    const Timer clock;
    ++s.calls;
    std::vector<double> candidate = lock_round(original_, locks_, integer_columns_, x);
    Count moves = 0;
    const int budget = static_cast<int>(std::min<std::size_t>(4 * integer_columns_.size() + 10,
                                                              static_cast<std::size_t>(10000)));
    if (repair(original_, integer_columns_, &candidate, budget, tol::kPrimalFeasibility,
               &moves)) {
      (void)offer_from(kRepair, candidate);
    }
    s.work += moves;
    s.seconds += clock.elapsed_seconds();
  }

  // RENS (Berthold 2014), once, at the root: the columns the relaxation already has integral
  // are what every rounding keeps anyway; the sub-MIP searches the box around the rest,
  // which is exactly where a rounding goes wrong. Skipped when fewer than half the integer
  // columns are integral - that box is the model itself.
  if (schedule_.rens && node_index == 0) {
    HeuristicStats& s = heuristic_stats_[kRens];
    const Timer clock;
    ++s.calls;
    Model box;
    Count fixed = 0;
    if (rens_submodel(original_, integer_columns_, x, tol::kSubMipMinFixedFraction,
                      integrality_tolerance_, &box, &fixed)) {
      Logger quiet(nullptr);
      const Solution found = solve_branch_and_bound(
          box,
          sub_mip_options(
              options_, schedule_.rens_nodes,
              sub_mip_seconds(limits_, timer_.elapsed_seconds(), schedule_.seconds_budgets)),
          quiet, control_);
      s.work += found.nodes;
      if (claims_a_point(found.status) && !found.col_value.empty()) {
        (void)offer_from(kRens, found.col_value);
      }
    }
    s.seconds += clock.elapsed_seconds();
  }

  // RINS, every mip_rins_frequency nodes once there is an incumbent to compare against, with
  // TWO budgets - a heuristic that out-spends the search it serves is not one. Its sub-MIP
  // nodes are capped in total at a tenth of the main search's (with a floor), and, unless
  // the solve is deterministic, its TIME at a tenth of the solve's. The node cap alone was
  // not enough: on pk1 a sub-MIP node cost seven main nodes, and 601 of them - inside the
  // node cap - took 8.8 s of a 20 s solve.
  if (schedule_.rins && schedule_.rins_frequency > 0 && have_incumbent_ && node_index > 0 &&
      nodes_explored_ % schedule_.rins_frequency == 0) {
    HeuristicStats& s = heuristic_stats_[kRins];
    const Count budget = std::max<Count>(
        schedule_.rins_nodes,
        static_cast<Count>(tol::kSubMipBudgetShare * static_cast<double>(nodes_explored_)));
    const double time_budget =
        tol::kSubMipBudgetShare * std::max(1.0, timer_.elapsed_seconds());
    if (s.work < budget && (!schedule_.seconds_budgets || s.seconds < time_budget)) {
      const Timer clock;
      ++s.calls;
      Model neighbourhood;
      Count fixed = 0;
      if (rins_submodel(original_, integer_columns_, x, incumbent_x_,
                        tol::kSubMipMinFixedFraction, integrality_tolerance_, &neighbourhood,
                        &fixed)) {
        Logger quiet(nullptr);
        const Solution found = solve_branch_and_bound(
            neighbourhood,
            sub_mip_options(
                options_, std::min<Count>(schedule_.rins_nodes, budget - s.work),
                sub_mip_seconds(limits_, timer_.elapsed_seconds(), schedule_.seconds_budgets)),
            quiet, control_);
        s.work += found.nodes;
        if (claims_a_point(found.status) && !found.col_value.empty()) {
          (void)offer_from(kRins, found.col_value);
        }
      }
      s.seconds += clock.elapsed_seconds();
    }
  }
}

void BranchAndBound::unwind_to(std::size_t mark) {
  // In reverse, so a column tightened twice returns to what it was before the first.
  while (saved_.size() > mark) {
    const DomainChange& change = saved_.back();
    const auto u = static_cast<std::size_t>(change.column);
    if (change.is_upper) {
      working_.col_upper[u] = change.value;
    } else {
      working_.col_lower[u] = change.value;
    }
    saved_.pop_back();
  }
}

void BranchAndBound::run_dives(Index node_index, const std::vector<double>& x) {
  if (integer_columns_.empty()) return;
  const bool due = node_index == 0 || (schedule_.dive_frequency > 0 &&
                                       nodes_explored_ % schedule_.dive_frequency == 0);
  if (!due) return;
  for (std::size_t r = 0; r < kDiveRules; ++r) {
    if (!schedule_.dive[r]) continue;
    const auto rule = static_cast<DiveRule>(r);
    if (rule == DiveRule::kGuided && !have_incumbent_) continue;  // nothing to be guided by
    dive(kDiveFractional + r, rule, x);
  }
}

// Achterberg, "Constraint Integer Programming" (thesis, 2007), sec. 9.2: a diving heuristic
// commits to a fractional column's rounded value, re-solves the LP relaxation with that bound
// fixed, and repeats - a single, greedy descent toward an integral point rather than a
// search. Its value is concentrated at the root: an early incumbent is what lets can_prune()
// start fathoming nodes from the very first branch, instead of only after the tree has found
// one on its own. The depth cap in tolerances.hpp and mip_dive_lp_resolves bound what one
// dive may spend, so it cannot itself dominate the cost of the node it runs at.
//
// The fractional rule fixes the LEAST fractional column, not the most: most_fractional()
// (used for branching) picks the column the relaxation is least sure of, because that is
// where a split actually separates the search space. Diving wants the opposite bias - lock
// in what the relaxation already agrees on, disturb the point as little as possible, and let
// the next re-solve reveal whether that choice was consistent with everything else. The
// other three rules (#414) are in choose_dive_column().
void BranchAndBound::dive(std::size_t slot, DiveRule rule, const std::vector<double>& start_x) {
  HeuristicStats& s = heuristic_stats_[slot];
  const Timer clock;
  ++s.calls;
  // Every bound this dive fixes is undone before it returns, so the next dive - and the
  // branching decision after it - start from the node's own domain.
  const std::size_t mark = saved_.size();
  const std::vector<double> no_incumbent;
  const std::vector<double>& guide = have_incumbent_ ? incumbent_x_ : no_incumbent;
  std::vector<double> x = start_x;
  int depth = 0;
  int lp_resolves = 0;
  bool backtracked = false;

  while (depth < tol::kDivingMaxDepth && lp_resolves < schedule_.dive_lp_resolves) {
    const DiveChoice choice = choose_dive_column(original_, locks_, integer_columns_, x, rule,
                                                 guide, integrality_tolerance_);
    if (choice.column < 0) {
      // Every integer column is within tolerance: an integral point. Diving finds a
      // candidate, it does not get to assert it is one - offer_incumbent() re-checks
      // feasibility and integrality against the ORIGINAL model exactly as it does for
      // try_rounding() or a node whose relaxation happened to be integral.
      if (offer_from(slot, x)) {
        logger_.verbose("{} found an incumbent at {:.10g} after {} LP re-solve(s)",
                        to_string(rule), reported(incumbent_internal_), lp_resolves);
      }
      break;
    }

    const auto u = static_cast<std::size_t>(choice.column);
    const std::size_t before_fix = saved_.size();
    tighten_lower(u, choice.value);
    tighten_upper(u, choice.value);
    ++depth;
    Solution probe = solve_node();
    ++lp_resolves;
    if (probe.status != SolveStatus::kOptimal && schedule_.dive_backtrack && !backtracked) {
      // ONE backtrack (Achterberg 2007, sec. 9.2): undo the fix and try the other side of
      // the column once; a second dead end ends the dive.
      backtracked = true;
      unwind_to(before_fix);
      const double other = choice.value > x[u] ? std::floor(x[u]) : std::ceil(x[u]);
      const bool inside =
          (!is_finite_bound(working_.col_lower[u]) || other >= working_.col_lower[u] - 1e-9) &&
          (!is_finite_bound(working_.col_upper[u]) || other <= working_.col_upper[u] + 1e-9);
      if (inside) {
        tighten_lower(u, other);
        tighten_upper(u, other);
        probe = solve_node();
        ++lp_resolves;
      }
    }
    if (probe.status != SolveStatus::kOptimal) break;  // dead end: infeasible or worse
    x = probe.col_value;
    // The next probe fixes one more column of THIS point, so this basis is its warm start.
    current_warm_ = basis_of(probe);
  }
  s.work += lp_resolves;
  unwind_to(mark);
  s.seconds += clock.elapsed_seconds();
}

void BranchAndBound::run_root_pump(const Solution& relaxation) {
  if (!schedule_.pump || schedule_.pump_rounds <= 0 || have_incumbent_ || quadratic_) return;
  HeuristicStats& s = heuristic_stats_[kPump];
  const Timer clock;
  ++s.calls;
  Options lp = node_options_;
  lp.set_bool("log_to_console", false);
  lp.set_bool("presolve", false);
  if (schedule_.seconds_budgets && limits_.has_time_limit()) {
    lp.set_double("time_limit",
                  std::max(0.0, 0.2 * limits_.remaining_seconds(timer_.elapsed_seconds())));
  }
  Count solves = 0;
  const std::vector<double> x =
      feasibility_pump(original_, integer_columns_, relaxation.col_value, lp,
                       schedule_.pump_rounds, integrality_tolerance_, &solves);
  s.work += solves;
  if (!x.empty()) (void)offer_from(kPump, x);
  s.seconds += clock.elapsed_seconds();
}

void BranchAndBound::report_heuristics() {
  if (heuristic_stats_.empty()) return;
  for (const HeuristicStats& s : heuristic_stats_) {
    if (s.calls == 0) continue;
    logger_.info(
        "Heuristic {:<21} calls {:>7}  proposed {:>6}  improved {:>4}  work {:>7}  {:.3f}s",
        s.name, s.calls, s.found, s.improved, s.work, s.seconds);
  }
}

}  // namespace sankhya::mip
