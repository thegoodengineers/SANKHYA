// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the root separation loop (#495): rounds 2 onward of the root cut round.
//
// WHY A LOOP. One round separates at the first LP vertex, adds what it chose and re-solves.
// The new vertex is a different point: the Gomory cuts of its basis are new cuts, the MIR
// and combinatorial separators see a different x*, and the cuts the first round left
// waiting may now be the most violated. Every published branch-and-cut code loops at the
// root for that reason, and stops when the bound stops moving (Achterberg, "Constraint
// Integer Programming", thesis, TU Berlin 2007, ch. 8; Achterberg & Wunderling, "Mixed
// integer programming: analyzing 12 years of progress", in Facets of Combinatorial
// Optimization, Springer 2013, for what the root cuts are worth; Wesselmann & Suhl,
// "Implementing cutting plane management and selection techniques", Paderborn 2012, for
// the selection each round repeats).
//
// VALIDITY. Every round separates on `working_` at the root, whose rows are the model's and
// the cut rows earlier rounds appended: valid inequalities all, so what is derived from them
// is valid (a Gomory cut of a tableau row containing a cut row's logical treats that
// logical as continuous, as it does every logical). Column bounds are the root's, which
// hold for the whole tree. Nothing here is specific to a node.
//
// WARM. Appending rows to an optimal basis with their logicals basic leaves it dual
// feasible, so each round re-solves with the dual simplex from the last round's basis, as
// the tree rounds do (branch_and_bound_cuts.cpp).
//
// STOPPING, whichever comes first: a round that gains at most a share of round 1's gain
// (tailing off, kRootCutTailingOffFraction; Padberg and Rinaldi 1991), the bound stalls
// (tolerances.hpp, kRootCutStall*), the round cap, the time share, the row budget, a point
// already integral, a round whose filter and selection take nothing, or a re-solve that does
// not come back optimal (that round is rolled back).
//
// THE ROW BUDGET. A cut row is paid for at every node LP of the tree, not at the root, so
// neither the stall rule nor the time share sees its cost: rounds that are each fast and
// each move the bound can still leave the tree an LP several times the model's size. The
// loop therefore adds at most max(kRootCutRowFloor, kRootCutRowShare * m) rows in all,
// round 1 included, m the rows before the first cut. A round the budget cannot hold whole
// takes its best cuts up to the budget (selection returns them best first) and the rest
// wait for the tree rounds with the other deferred cuts. Achterberg's thesis (ch. 8) limits
// the root's separation effort the same way, in rounds and in cuts; the numbers are ours.

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "branch_and_bound_internal.hpp"
#include "mir_cmir.hpp"

namespace sankhya::mip {

namespace {

/// "gomory 3, mir 2" for a set of cuts, in family order; "none" when empty.
std::string by_family(const std::vector<Cut>& cuts) {
  std::map<int, int> counts;
  for (const Cut& cut : cuts) ++counts[static_cast<int>(cut.family)];
  std::string out;
  for (const auto& [family, count] : counts) {
    if (!out.empty()) out += ", ";
    out += fmt::format("{} {}", cut_family_name(static_cast<CutFamily>(family)), count);
  }
  return out.empty() ? "none" : out;
}

}  // namespace

bool BranchAndBound::root_cut_loop_wanted(const std::vector<Cut>& round1) {
  if (root_loop_wanted_ < 0) {
    bool wanted = options_.get_bool("root_cut_loop");
    const std::string when = options_.get_string("root_cut_loop_when");
    if (wanted && when == "flow-covers") {
      // Round 1 took a flow cover: the separator found a violated cover on a switched flow,
      // which is the reading that separated the loop's wins from its losses in #1158's
      // A/B (b) - bell5 has the rows but no cover was ever found on it.
      wanted = std::any_of(round1.begin(), round1.end(),
                           [](const Cut& cut) { return cut.family == CutFamily::kFlowCover; });
      if (!wanted) {
        logger_.info(
            "Root cut loop: not entered, round 1 took no flow cover "
            "(root_cut_loop_when=flow-covers)");
      }
    } else if (wanted && when == "auto") {
      // A flow cover in round 1 enters as under flow-covers. Without one, a model with a
      // variable upper bound enters ON PROBATION (tolerances.hpp, kRootCutProbationGain):
      // gesa2 and mod011 have the structure and take no cover in round 1, and so does
      // bell5, which the rounds only cost; what tells them apart is what the rounds gain,
      // and root_cut_loop() rolls the root back to round 1 when it is too little.
      const bool cover = std::any_of(round1.begin(), round1.end(), [](const Cut& cut) {
        return cut.family == CutFamily::kFlowCover;
      });
      wanted = cover || find_variable_bounds(working_).rows > 0;
      root_loop_probation_ = wanted && !cover;
      if (!wanted) {
        logger_.info(
            "Root cut loop: not entered, round 1 took no flow cover and the model has no "
            "variable upper bound (root_cut_loop_when=auto)");
      }
    } else if (wanted && when == "switched-flows") {
      // The rounds pay on fixed-charge and lot-sizing models, whose switched flows x <= u y
      // are what the aggregated flow covers and the c-MIR path cuts work on (#1158's A/B:
      // every instance the loop proved had such a row, every tree it lost had none). The
      // reading is the c-MIR's, once, on the root model.
      wanted = find_variable_bounds(working_).rows > 0;
      if (!wanted) {
        logger_.info(
            "Root cut loop: not entered, the model has no variable upper bound "
            "(root_cut_loop_when=switched-flows)");
      }
    }
    // The pool follows the gate (#1193's tier-1 gate): when root_cut_loop is on but its gate
    // keeps the loop out, the search runs as it would without the loop, pool included. On ej
    // and enlight8 the pool's row removal was the only difference the recipe made (same cuts,
    // loop not entered), and it cost ej its optimum and enlight8 its proof. Decided in round
    // 1, before any row has aged, so no row has been removed yet.
    if (!wanted && options_.get_bool("root_cut_loop") && cut_pooling_) {
      cut_pooling_ = false;
      cut_removal_ = false;
      logger_.info("Cut pool: off with the root cut loop (mip_cut_pooling follows its gate)");
    }
    root_loop_wanted_ = wanted ? 1 : 0;
  }
  return root_loop_wanted_ == 1;
}

void BranchAndBound::root_cut_loop(Solution* relaxation, Index model_rows,
                                   const std::vector<Cut>& first_taken,
                                   std::int64_t first_iterations) {
  // bounds[k] is the root bound after round k; bounds[0] before any cut.
  std::vector<double> bounds{root_bound_internal_, root_bound_after_cuts_internal_};
  root_cut_rounds_ = 1;
  logger_.info("Root cut round 1: bound {:.10g}, took {} ({}), {} LP iteration(s)",
               reported(bounds.back()), first_taken.size(), by_family(first_taken),
               first_iterations);
  const bool timed = time_limit_ > 0.0 && time_limit_ < 1e300;
  // With the pool removing the slack rows at the end (below), the budget is the pooled one.
  const auto row_budget = static_cast<Count>(
      root_cut_row_budget(model_rows, cut_removal_ ? cut_max_per_round_ : 0));
  // Probation (root_cut_loop_wanted, `auto`): the root as round 1 left it, to go back to.
  struct AfterRoundOne {
    Model model;
    decltype(scaling_) scaling;
    Solution relaxation;
    std::vector<Cut> waiting;
    std::size_t pool = 0;
    Count applied = 0;
  };
  std::optional<AfterRoundOne> round_one;
  if (root_loop_probation_) {
    round_one.emplace();
    round_one->model = working_;
    round_one->scaling = scaling_;
    round_one->relaxation = *relaxation;
    round_one->waiting = waiting_cuts_;
    round_one->pool = pool_cuts_.size();
    round_one->applied = root_cuts_applied_;
  }
  const char* why = "the round cap";
  for (int round = 2; round <= tol::kRootCutMaxRounds; ++round) {
    if (root_cuts_applied_ >= row_budget) {
      why = "the row budget";
      break;
    }
    const auto rounds_done = static_cast<int>(bounds.size()) - 1;
    // Tailing off (tolerances.hpp): the last round gained at most a fixed share of what
    // round 1 gained, so the rounds after it would add rows for little bound.
    if (rounds_done >= 2) {
      const double first_gain = bounds[1] - bounds[0];
      const double last_gain = bounds.back() - bounds[bounds.size() - 2];
      if (last_gain <= tol::kRootCutTailingOffFraction * std::max(first_gain, 0.0)) {
        why = "tailing off";
        break;
      }
    }
    if (rounds_done >= tol::kRootCutStallRounds) {
      const double now = bounds.back();
      const double moved = now - bounds[bounds.size() - 1 - tol::kRootCutStallRounds];
      const double reference = have_incumbent_ ? std::max(incumbent_internal_ - now, 0.0)
                                               : std::max(1.0, std::fabs(now));
      if (moved <= tol::kRootCutStallFraction * reference) {
        why = "the bound stalled";
        break;
      }
    }
    if (timed && timer_.elapsed_seconds() > tol::kRootCutTimeShare * time_limit_) {
      why = "the time share";
      break;
    }
    if (most_fractional(relaxation->col_value) < 0) {
      why = "an integral point";
      break;
    }

    std::vector<Cut> candidates = separate_root_candidates(*relaxation, model_rows);
    if (debug_.has_value()) {
      debug_round_ = fmt::format("root round {}", round);
      debug_check_cuts(candidates, -1);  // #500: every new candidate, before the filter
    }
    candidates.insert(candidates.end(), waiting_cuts_.begin(), waiting_cuts_.end());
    waiting_cuts_.clear();
    const std::size_t found = candidates.size();
    auto filtered =
        filter_and_deduplicate_cuts(working_, *relaxation, candidates, cut_filter_policy(true));
    admit_dense_root_cuts(&filtered, *relaxation);  // #496: the same budget as round 1
    std::vector<Cut> passing;
    for (auto& fc : filtered) {
      if (fc.reason != CutFilterReason::kAccepted) continue;
      if (is_pooled_duplicate(fc.cut)) continue;  // a row an earlier round already added
      passing.push_back(std::move(fc.cut));
    }
    CutSelection chosen = select_cuts(working_, relaxation->col_value, std::move(passing),
                                      cut_max_per_round_, cut_max_parallelism_);
    std::vector<Cut> accepted = std::move(chosen.selected);
    waiting_cuts_ = std::move(chosen.deferred);
    // A round the budget cannot hold whole takes its best cuts up to it (#495).
    take_within_budget(&accepted, &waiting_cuts_,
                       static_cast<std::size_t>(row_budget - root_cuts_applied_));
    certify_round_cuts(&accepted);  // #518: certificate mode only
    if (accepted.empty()) {
      logger_.info("Root cut round {}: {} candidate(s), none taken ({})", round, found,
                   describe_cut_filter(filtered));
      why = "a round that took nothing";
      break;
    }

    Model pre_round_model = working_;
    auto pre_round_scaling = scaling_;
    const std::size_t pool_before = pool_cuts_.size();
    append_cut_rows(accepted);
    resize_warm_starts(working_.num_rows());
    current_warm_ = basis_of(*relaxation);
    current_warm_.row_status.resize(static_cast<std::size_t>(working_.num_rows()),
                                    BasisStatus::kBasic);
    Solution next = solve_node();
    if (next.status != SolveStatus::kOptimal) {
      // The rows go back out and the loop ends on the last round that solved.
      logger_.info("Root cut round {} induced {}; rolled back to round {}", round,
                   to_string(next.status), round - 1);
      working_ = std::move(pre_round_model);
      scaling_ = std::move(pre_round_scaling);
      drop_cuts_from(pool_before);
      resize_warm_starts(working_.num_rows());
      why = "a failed re-solve";
      break;
    }
    *relaxation = std::move(next);
    root_cuts_applied_ += static_cast<Count>(accepted.size());
    root_bound_after_cuts_internal_ = internal_objective(relaxation->col_value);
    bounds.push_back(root_bound_after_cuts_internal_);
    root_cut_rounds_ = round;
    logger_.info(
        "Root cut round {}: bound {:.10g}, {} candidate(s) ({}), took {} ({}), {} LP "
        "iteration(s)",
        round, reported(bounds.back()), found, describe_cut_filter(filtered), accepted.size(),
        by_family(accepted), relaxation->iterations);
  }
  // PROBATION ENDS. The rounds after round 1 stay only if they gained at least
  // kRootCutProbationGain times round 1's own gain; otherwise their rows go back out and the
  // search runs as it does when the gate keeps the loop out: round 1's rows, no pool. The
  // rows of rounds 2.. are the last ones appended (nothing is removed before this point), so
  // dropping them from the pool and restoring the saved model is the whole of it.
  if (round_one.has_value()) {
    const double first_gain = std::max(bounds[1] - bounds[0], 0.0);
    const double later_gain = bounds.back() - bounds[1];
    if (later_gain <= tol::kRootCutProbationGain * first_gain) {
      logger_.info(
          "Root cut loop: probation failed after {} round(s), bound {:.10g} to {:.10g} against "
          "{:.10g} after round 1; back to round 1 (root_cut_loop_when=auto), stopped on {}",
          root_cut_rounds_, reported(bounds.front()), reported(bounds.back()),
          reported(bounds[1]), why);
      drop_cuts_from(round_one->pool);
      working_ = std::move(round_one->model);
      scaling_ = std::move(round_one->scaling);
      *relaxation = std::move(round_one->relaxation);
      waiting_cuts_ = std::move(round_one->waiting);
      root_cuts_applied_ = round_one->applied;
      root_bound_after_cuts_internal_ = bounds[1];
      root_cut_rounds_ = 1;
      root_loop_wanted_ = 0;
      cut_pooling_ = false;
      cut_removal_ = false;
      resize_warm_starts(working_.num_rows());
      current_warm_ = basis_of(*relaxation);
      return;
    }
  }
  // ONLY THE TIGHT ROWS STAY (#956's follow-up): a cut row whose logical is basic at the
  // final root LP is slack there, and every node LP would carry it for nothing until it
  // ages out. With the cut pool removing rows, it goes to the pool now; the point and the
  // duals do not change (remove_cut_rows), and a node whose point violates it gets it back.
  // Without removal it stays: a row freed in place still costs its place in the LP.
  Count pooled = 0;
  if (cut_removal_ && first_cut_row_ >= 0 &&
      relaxation->row_status.size() == static_cast<std::size_t>(working_.num_rows())) {
    std::vector<std::size_t> slack;
    for (std::size_t p = 0; p < cut_lp_rows_.size(); ++p) {
      const std::size_t k = cut_lp_rows_[p];
      const auto row = static_cast<std::size_t>(first_cut_row_) + p;
      if (cut_row_free_[k] || relaxation->row_status[row] != BasisStatus::kBasic) continue;
      cut_row_free_[k] = true;
      slack.push_back(p);
    }
    if (!slack.empty()) remove_cut_rows(slack, relaxation);
    pooled = static_cast<Count>(slack.size());
  }
  current_warm_ = basis_of(*relaxation);
  if (pooled > 0) {
    logger_.info("Root cut loop: {} slack cut row(s) moved to the pool, {} kept in the LP",
                 pooled, cut_lp_rows_.size());
  }
  if (root_cuts_applied_ >= row_budget)
    why = "the row budget";  // e.g. the last round filled it
  logger_.info(
      "Root cut loop: {} round(s), {} row(s) of a budget of {}, bound {:.10g} to {:.10g}; "
      "stopped on {}",
      root_cut_rounds_, root_cuts_applied_, row_budget, reported(bounds.front()),
      reported(bounds.back()), why);
}

}  // namespace sankhya::mip
