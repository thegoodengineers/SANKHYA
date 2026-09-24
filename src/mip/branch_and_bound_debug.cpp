// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the branch and bound's side of the debug-solution check (#500). What is checked
// and why is in debug_solution.hpp; this file holds the hooks the search calls, so the
// search itself carries one line per check and no logic.
//
// THE NODE BOX. A known point can only be demanded of a node whose domain contains it: the
// branching bounds of every other node exclude it legitimately. So the node checks ask first
// whether the point lies in the node's column box (and, under objective branching, inside
// the objective row's branched bounds); every row of the working model already contains it,
// because the rows are checked once at the start and every cut row as it is appended.
#include <cmath>
#include <string>

#include <fmt/format.h>

#include "branch_and_bound_internal.hpp"

namespace sankhya::mip {

namespace {

std::string column_label(const Model& model, Index j) {
  const auto u = static_cast<std::size_t>(j);
  if (u < model.col_names.size() && !model.col_names[u].empty()) return model.col_names[u];
  return fmt::format("C{}", j);
}

}  // namespace

void BranchAndBound::debug_start() {
  debug_ = load_debug_solution(options_, original_, logger_);
  if (!debug_.has_value()) return;
  // The rows the search added itself before the first node: symmetry ordering rows (#413)
  // and the free objective row (#418). The model's own rows were checked by
  // solve_branch_and_bound before the search was built.
  if (const std::string bad = debug_->model_violation(working_); !bad.empty()) {
    debug_solution_abort(logger_,
                         fmt::format("a row the search appended before the root ({} symmetry "
                                     "ordering row(s), optimality-based) excludes it: {}",
                                     symmetry_rows_, bad));
  }
  debug_objective_ = internal_objective(debug_->x());
  logger_.info(
      "debug_solution (#500): checking every cut and reduction against a point of "
      "objective {:.10g}",
      reported(debug_objective_));
}

bool BranchAndBound::debug_node_contains() const {
  if (!debug_.has_value()) return false;
  if (debug_->first_column_outside(working_.col_lower, working_.col_upper) >= 0) return false;
  if (objective_row_ >= 0) {
    const auto r = static_cast<std::size_t>(objective_row_);
    const double value = objective_row_value(debug_->x());
    const double slack = tol::kDebugSolutionTolerance * std::max(1.0, std::fabs(value));
    if (value < working_.row_lower[r] - slack || value > working_.row_upper[r] + slack) {
      return false;
    }
  }
  return true;
}

void BranchAndBound::debug_check_cuts(const std::vector<Cut>& cuts, Index first_row) {
  if (!debug_.has_value()) return;
  // Tree cuts are built on the global bounds, and reduced-cost fixing tightens those against
  // the incumbent. Once that has legitimately excluded the point (an incumbent at least as
  // good exists; debug_after_global_tightening aborts otherwise), a cut need only be valid
  // for what is left, which no longer contains it.
  if (debug_->first_column_outside(global_lower_, global_upper_) >= 0) return;
  for (std::size_t k = 0; k < cuts.size(); ++k) {
    const std::string bad = debug_->cut_violation(cuts[k]);
    if (bad.empty()) continue;
    // A candidate is checked before the filter too: an invalid cut is a bug in its
    // separator whether or not the filter happened to refuse it this time.
    const std::string where = first_row < 0
                                  ? fmt::format("candidate {} before the filter", k)
                                  : fmt::format("row {}", first_row + static_cast<Index>(k));
    debug_solution_abort(
        logger_, fmt::format("{} cut, {}, node {}, {}: {}", cut_family_name(cuts[k].family),
                             debug_round_, debug_node_, where, bad));
  }
}

void BranchAndBound::debug_after_propagation(bool feasible) {
  if (!feasible) {
    debug_solution_abort(
        logger_, fmt::format("node propagation at node {} declared the node infeasible, and "
                             "the node's box contains the debug solution",
                             debug_node_));
  }
  const Index outside = debug_->first_column_outside(working_.col_lower, working_.col_upper);
  if (outside < 0) return;
  const auto u = static_cast<std::size_t>(outside);
  debug_solution_abort(
      logger_, fmt::format("node propagation at node {} tightened column {} to "
                           "[{:.17g}, {:.17g}]; the debug solution has {:.17g}",
                           debug_node_, column_label(original_, outside), working_.col_lower[u],
                           working_.col_upper[u], debug_->x()[u]));
}

void BranchAndBound::debug_after_node_lp(const Solution& relaxation) {
  if (relaxation.status == SolveStatus::kInfeasible) {
    debug_solution_abort(
        logger_, fmt::format("the node LP at node {} is infeasible, and the node's box and "
                             "every row contain the debug solution",
                             debug_node_));
  }
  if (relaxation.status != SolveStatus::kOptimal) return;
  // A relaxation bound is a LOWER bound on every point of the node, so it cannot exceed the
  // objective of a point the node contains; the rounding to the objective step (#221) is
  // part of what is checked, because can_prune() prunes on the rounded value.
  const double bound = integral_bound(internal_objective(relaxation.col_value));
  const double slack =
      tol::kDebugSolutionTolerance * std::max(1.0, std::fabs(debug_objective_)) +
      tol::kPrimalFeasibility;
  if (bound <= debug_objective_ + slack) return;
  debug_solution_abort(
      logger_, fmt::format("the bound at node {} is {:.17g} (after rounding to the objective "
                           "step), above the debug solution's objective {:.17g}, which the "
                           "node contains",
                           debug_node_, reported(bound), reported(debug_objective_)));
}

void BranchAndBound::debug_after_global_tightening() {
  if (!debug_.has_value()) return;
  // Reduced-cost fixing removes points no better than the incumbent less the gap target; a
  // point that good is not what it promises to keep (debug_solution.hpp).
  if (have_incumbent_ && incumbent_internal_ - absolute_gap_target_ <= debug_objective_) return;
  const Index outside = debug_->first_column_outside(global_lower_, global_upper_);
  if (outside < 0) return;
  const auto u = static_cast<std::size_t>(outside);
  debug_solution_abort(
      logger_,
      fmt::format("reduced-cost fixing (optimality-based) moved column {} to "
                  "[{:.17g}, {:.17g}] against the incumbent {:.17g}; the debug "
                  "solution has {:.17g} at objective {:.17g}",
                  column_label(original_, outside), global_lower_[u], global_upper_[u],
                  reported(incumbent_internal_), debug_->x()[u], reported(debug_objective_)));
}

void check_search_input_against_debug_solution(const Model& received, const Model* tightened,
                                               const DebugSolution& point, Logger& logger) {
  if (const std::string bad = point.model_violation(received); !bad.empty()) {
    debug_solution_abort(logger,
                         "the model the branch and bound received (presolve's output "
                         "when presolve ran) excludes it: " +
                             bad);
  }
  if (tightened == nullptr) return;
  if (const std::string bad = point.model_violation(*tightened); !bad.empty()) {
    debug_solution_abort(logger,
                         "integral row rounding (tighten_integral_rows) excludes it: " + bad);
  }
}

void check_search_result_against_debug_solution(const Model& model, const Solution& result,
                                                const Options& options,
                                                const DebugSolution& point, Logger& logger) {
  const double sense = model.sense_multiplier();
  const double known = model.evaluate_objective(point.x().data());
  if (result.status == SolveStatus::kInfeasible) {
    debug_solution_abort(logger, fmt::format("the search reported the model infeasible; the "
                                             "debug solution is feasible at objective {:.17g}",
                                             known));
  }
  if (result.status != SolveStatus::kOptimal) return;
  // Optimal within the gap targets: the answer may be worse than the known point by that
  // much and no more.
  const double allowed =
      std::max(options.get_double("mip_absolute_gap"),
               options.get_double("mip_relative_gap") * std::max(1.0, std::fabs(known))) +
      tol::kDebugSolutionTolerance * std::max(1.0, std::fabs(known));
  if (sense * (result.objective - known) <= allowed) return;
  debug_solution_abort(logger,
                       fmt::format("the search reported {:.17g} optimal; the debug solution is "
                                   "feasible at {:.17g}, better by more than the gap targets",
                                   result.objective, known));
}

}  // namespace sankhya::mip
