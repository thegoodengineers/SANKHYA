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
// STOPPING, whichever comes first: the bound stalls (tolerances.hpp, kRootCutStall*), the
// round cap, the time share, a point already integral, a round whose filter and selection
// take nothing, or a re-solve that does not come back optimal (that round is rolled back).

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "branch_and_bound_internal.hpp"

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
  const char* why = "the round cap";
  for (int round = 2; round <= tol::kRootCutMaxRounds; ++round) {
    const auto rounds_done = static_cast<int>(bounds.size()) - 1;
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
        filter_and_deduplicate_cuts(working_, *relaxation, candidates, cut_filter_policy());
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
      pool_cuts_.resize(pool_before);
      cut_row_slack_.resize(pool_before);
      cut_row_free_.resize(pool_before);
      if (pool_cuts_.empty()) first_cut_row_ = -1;
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
  current_warm_ = basis_of(*relaxation);
  logger_.info("Root cut loop: {} round(s), {} row(s), bound {:.10g} to {:.10g}; stopped on {}",
               root_cut_rounds_, root_cuts_applied_, reported(bounds.front()),
               reported(bounds.back()), why);
}

}  // namespace sankhya::mip
