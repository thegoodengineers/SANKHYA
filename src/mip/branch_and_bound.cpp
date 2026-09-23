// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound over the revised primal simplex.
//
// References, written from the literature:
//   Land & Doig, "An automatic method of solving discrete programming problems",
//     Econometrica 28(3), 1960 - the method itself
//   Wolsey, "Integer Programming" (1998), ch. 7 - bounding, fathoming, node selection
//   Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 5-6 - the practical
//     shape of a modern search: propagation at nodes, and the incumbent as a cutoff
//   Savelsbergh, "Preprocessing and probing for mixed integer programming problems",
//     ORSA J. Computing 6(4), 1994 - bound propagation from row activities
//
// THE TREE DOES NOT COPY THE MODEL. One working Model is built once, and a node is entered
// by applying the chain of bound changes from the root and left by undoing them. A node
// therefore costs O(depth) to enter, not O(nonzeros), and the constraint matrix exists once
// no matter how large the tree grows.
//
// WHAT THIS FILE DOES BY DEFAULT, AND WHAT IT DOES NOT. Reliability branching (#69) is the
// default rule. The root cutting planes in cuts.hpp (#159) are ON by default since the
// three-way at 0018254, once cut selection (#415) made the root round prove 10 of 30 where
// no cuts prove 9 at 1.015x the nodes; before selection the same round cost nodes and a
// match, and was off. Rounds below the root (tree_cut_depth) lose that proof again and
// stay opt-in. There is no node presolve beyond simple propagation, and no parallelism
// unless mip_threads asks for it. The
// order those arrived in was deliberate: a cut that is very slightly invalid removes the
// optimum and the search then proves the wrong answer, confidently, so a plain, correct
// search came first and is what makes each addition checkable.

#include "sankhya/mip.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

#include "cuts.hpp"
#include "mir_cuts.hpp"
#include "solution_pool.hpp"
#include "symmetry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/stop_controller.hpp"
#include "../util/profiler.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

#include "branch_and_bound_internal.hpp"
#include "parallel_search.hpp"

namespace sankhya::mip {

// THE OBJECTIVE IS INTEGRAL MORE OFTEN THAN IT LOOKS (#221). On the 30-instance MIPLIB set,
// four of the six instances that hold the published optimum without proving it have an
// objective that every integer solution evaluates to an integer: noswot's costs are
// integers on integer columns; b-ball, opt1217 and rlp1 minimise one continuous column that
// a single row defines from integer columns with integer coefficients. Their bounds sat at
// 14 against an incumbent of 15 (rlp1), -43 against -41 (noswot): a bound that no integer
// solution can attain is one the search may round, and nothing here knew the objective was
// integral. Both patterns are detected once, and every relaxation bound is rounded up to
// the next multiple of the step before it is compared to the incumbent, so nodes in
// (14, 15) are fathomed and a bound above 14 proves 15. Wolsey, "Integer Programming"
// (1998), sec. 7.3; the defining-row case is what presolve's free-column-singleton
// substitution would produce if it ran on these models.
void BranchAndBound::detect_objective_integrality() {
  objective_step_ = 0.0;
  if (quadratic_ || !options_.get_bool("mip_objective_integrality")) return;
  const Index n = original_.num_cols();
  const auto is_integer_column = [&](Index j) {
    return original_.col_type[static_cast<std::size_t>(j)] == VarType::kInteger;
  };
  const auto integral = [](double v) { return std::fabs(v - std::round(v)) <= 1e-9; };
  const auto gcd = [](double a, double b) {
    auto x = static_cast<std::int64_t>(std::llround(std::fabs(a)));
    auto y = static_cast<std::int64_t>(std::llround(std::fabs(b)));
    while (y != 0) {
      const std::int64_t r = x % y;
      x = y;
      y = r;
    }
    return static_cast<double>(x);
  };

  // Direct rule: every costed column is integer with an integer cost; the step is their gcd.
  Index continuous_costed = -1;
  double step = 0.0;
  for (Index j = 0; j < n; ++j) {
    const double cost = original_.col_cost[static_cast<std::size_t>(j)];
    if (cost == 0.0) continue;
    if (!is_integer_column(j)) {
      if (continuous_costed >= 0) return;  // two continuous costed columns: nothing known
      continuous_costed = j;
      continue;
    }
    if (!integral(cost) || std::fabs(cost) > 1e12) return;
    step = gcd(step, cost);
  }
  if (continuous_costed < 0) {
    if (step >= 1.0) {
      objective_step_ = step;
      logger_.verbose("objective integrality: every costed column is integer, step {:g}",
                      objective_step_);
    }
    return;
  }
  // Defining-rows rule: the one continuous costed column carries the whole objective and
  // every row that bounds it from the side the objective pushes it to (from below when
  // minimising it, from above when maximising) defines it from integer columns with integer
  // coefficients and an integer right-hand side, after dividing by its own coefficient in
  // that row. That is the min-max shape of rlp1 (Z >= each resource's load), opt1217 and
  // b-ball: the node optimum sets the column to the largest of integer-valued expressions,
  // or to its own bound, which must be integral or absent too. Rows that bound it from the
  // other side, or not at all, only restrict the integer columns and do not matter. Then the
  // node optimum's value of the column is an integer and the objective a multiple of its
  // cost.
  if (step != 0.0) return;  // integer columns carry cost as well: mixed, not handled
  const auto uo = static_cast<std::size_t>(continuous_costed);
  const double cost = original_.col_cost[uo];
  for (const double bound : {original_.col_lower[uo], original_.col_upper[uo]}) {
    if (is_finite_bound(bound) && !integral(bound)) return;
  }
  const bool push_down = sense_ * cost > 0.0;
  const ColumnView column = original_.matrix.column(continuous_costed);
  if (column.size == 0) return;  // the column is free of every row: its bound is the answer
  const CsrView by_row(original_.matrix);
  Index defining_rows = 0;
  for (Index p = 0; p < column.size; ++p) {
    const Index row = column.rows[p];
    const double a_o = column.values[p];
    if (a_o == 0.0) continue;
    const auto ur = static_cast<std::size_t>(row);
    // In terms of x_o alone: a_o x_o + rest is within [row_lower, row_upper]. With a_o > 0
    // the row's lower bound bounds x_o from below; with a_o < 0 its upper bound does.
    const double lower_side = (a_o > 0.0) ? original_.row_lower[ur] : original_.row_upper[ur];
    const double upper_side = (a_o > 0.0) ? original_.row_upper[ur] : original_.row_lower[ur];
    const double side = push_down ? lower_side : upper_side;
    if (!is_finite_bound(side)) continue;  // bounds x_o from the other side only
    if (!integral(side / a_o)) return;
    const ColumnView entries = by_row.row(row);
    for (Index q = 0; q < entries.size; ++q) {
      const Index j = entries.rows[q];
      if (j == continuous_costed) continue;
      if (!is_integer_column(j) || !integral(entries.values[q] / a_o)) return;
    }
    ++defining_rows;
  }
  if (defining_rows == 0 &&
      !is_finite_bound(push_down ? original_.col_lower[uo] : original_.col_upper[uo])) {
    return;  // nothing bounds the column from the objective's side: unbounded or unknown
  }
  objective_step_ = std::fabs(cost);
  logger_.verbose(
      "objective integrality: column {} is bounded by {} row(s) defined from integer "
      "columns, step {:g}",
      continuous_costed, defining_rows, objective_step_);
}

// OBJECTIVE BRANCHING (#418) needs the objective as a row a node can bound: appended once at
// the root, free on both sides, before any cut row so the cut pool's row arithmetic is
// untouched. The coefficients are the model's own costs, so the row measures c x in the
// model's units and the step #398 detected applies to it directly.
void BranchAndBound::append_objective_row() {
  const Index old_rows = working_.num_rows();
  const Index cols = working_.num_cols();
  SparseMatrix matrix(old_rows + 1, cols);
  for (Index j = 0; j < cols; ++j) {
    const ColumnView view = working_.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) matrix.add_entry(view.rows[k], j, view.values[k]);
    const double cost = original_.col_cost[static_cast<std::size_t>(j)];
    if (std::fabs(cost) > tol::kZeroDrop) matrix.add_entry(old_rows, j, cost);
  }
  matrix.finalize();
  working_.matrix = std::move(matrix);
  working_.resize_rows(old_rows + 1);
  working_.row_lower[static_cast<std::size_t>(old_rows)] = -kInfinity;
  working_.row_upper[static_cast<std::size_t>(old_rows)] = kInfinity;
  objective_row_ = old_rows;
}

double BranchAndBound::objective_row_value(const std::vector<double>& x) const {
  double value = 0.0;
  for (Index j = 0; j < original_.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    value += original_.col_cost[u] * x[u];
  }
  return value;
}

// FORMULATION SYMMETRY (#413). A permutation of columns and rows that leaves the model
// unchanged maps every feasible point to one of the same objective, so a search that
// explores every point of an orbit does the same work as many times as the orbit is large -
// enlight8's 99,188 nodes at 60 s are that signature. The first slice is detection and one
// row per generator: for a generator whose first moved column is i, mapping it to k, the
// row x_i <= x_k holds for the lexicographically smallest point of every orbit (Liberti
// 2012), so at least one optimum survives; appended to working_ at the root, before any
// cut row, the way the cut rows are. Orbital fixing in the tree is the second slice. Every
// generator was verified entry by entry against original_ before it reached here, which is
// what makes the rows safe whatever the search found.
void BranchAndBound::append_symmetry_rows() {
  const SymmetryGroup group = detect_symmetry(
      original_, static_cast<Count>(options_.get_int("mip_symmetry_search_limit")));
  symmetry_generators_ = static_cast<Count>(group.generators.size());
  logger_.info(
      "Symmetry (#413): {} generator(s), {} orbit(s) of more than one column, the largest "
      "{} column(s); {} search node(s), {} refinement round(s){}",
      group.generators.size(), group.nontrivial_orbits, group.largest_orbit, group.search_nodes,
      group.refinements,
      group.budget_exhausted ? " (stopped at mip_symmetry_search_limit)" : "");
  const std::vector<std::pair<Index, Index>> pairs = ordering_rows(group);
  if (pairs.empty()) return;
  const Index old_rows = working_.num_rows();
  const Index cols = working_.num_cols();
  const auto added = static_cast<Index>(pairs.size());
  SparseMatrix matrix(old_rows + added, cols);
  for (Index j = 0; j < cols; ++j) {
    const ColumnView view = working_.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) matrix.add_entry(view.rows[k], j, view.values[k]);
  }
  for (Index k = 0; k < added; ++k) {
    matrix.add_entry(old_rows + k, pairs[static_cast<std::size_t>(k)].first, 1.0);
    matrix.add_entry(old_rows + k, pairs[static_cast<std::size_t>(k)].second, -1.0);
  }
  matrix.finalize();
  working_.matrix = std::move(matrix);
  working_.resize_rows(old_rows + added);
  for (Index k = 0; k < added; ++k) {
    working_.row_lower[static_cast<std::size_t>(old_rows + k)] = -kInfinity;
    working_.row_upper[static_cast<std::size_t>(old_rows + k)] = 0.0;
  }
  symmetry_rows_ = added;
  logger_.info("Symmetry (#413): {} ordering row(s) x_i <= x_k appended for the generators",
               added);
}

Solution BranchAndBound::run() {
  init_heuristics();
  detect_objective_integrality();
  // Reduced-cost fixing and restarts (#418), both off unless asked for.
  reduced_cost_fixing_ = options_.get_bool("mip_reduced_cost_fixing");
  restarts_allowed_ = reduced_cost_fixing_ ? options_.get_int("mip_restarts") : 0;
  restart_fraction_ = options_.get_double("mip_restart_fraction");
  restart_node_limit_ = options_.get_int("mip_restart_node_limit");
  // Objective branching (#418): only with a known step, and only in a search that owns its
  // working model - a parallel worker shares the driver's scaling, whose row count the
  // appended row would not match.
  if (options_.get_bool("mip_objective_branching") && objective_step_ > 0.0 && !quadratic_ &&
      shared_ == nullptr && seed_ == nullptr) {
    append_objective_row();
  }
  global_lower_ = working_.col_lower;
  global_upper_ = working_.col_upper;
  if (options_.get_bool("enable_root_cuts")) {
    tree_cut_depth_ = static_cast<Index>(options_.get_int("tree_cut_depth"));
    tree_cut_rows_per_round_ = static_cast<Index>(options_.get_int("tree_cut_rows_per_round"));
    cut_max_per_round_ = static_cast<Index>(options_.get_int("cut_max_per_round"));
    cut_max_parallelism_ = options_.get_double("cut_max_parallelism");
  }
  Solution solution;
  solution.allocate_for(original_);
  solution.algorithm = "branch-and-bound";

  const std::string problem = original_.validate();
  if (!problem.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = problem;
    return solution;
  }
  // Formulation symmetry (#413): only in a search that owns its working model - a parallel
  // worker shares the driver's scaling, whose row count the appended rows would not match -
  // and not for a quadratic objective, whose Hessian the detection does not read.
  if (options_.get_bool("mip_symmetry") && !quadratic_ && shared_ == nullptr &&
      seed_ == nullptr) {
    append_symmetry_rows();
  }

  // Once, here, and not once per node (#76). Built from working_ before any branching has
  // touched its bounds, though it would not matter if it had: only the matrix, the cost and
  // the row bounds feed the multipliers, and branching changes none of them.
  // A parallel worker (#222) takes the scaling the driver computed once: the matrix is the
  // same in every subtree, so it is not rebuilt for each one.
  scaling_ = shared_ != nullptr && shared_->scaling() != nullptr
                 ? *shared_->scaling()
                 : build_node_scaling(working_, node_options_);
  probe_options_ = node_options_;
  probe_options_.set_int("iteration_limit", tol::kStrongBranchingIterations);

  logger_.info("Branch and bound: {} rows, {} columns, {} integer columns",
               original_.num_rows(), original_.num_cols(), integer_columns_.size());

  // THE ROOT STARTS WITH NO BOUND PROVED (#289). TreeNode::bound is 0.0 by default, which
  // is a placeholder for "inherited from the parent" and the root has no parent. A search
  // stopped before it evaluated the root then reported a dual bound of 0: true by luck on a
  // model whose optimum is positive, and a false claim on one whose optimum is negative.
  // Minus infinity in minimise space is what "nothing is proved yet" actually is, and it
  // prunes nothing, which is also what an unevaluated root should do.
  TreeNode root;
  root.bound = -std::numeric_limits<double>::infinity();
  nodes_.push_back(root);
  if (seed_ != nullptr) {
    plant_seed();  // a subtree given away by another worker (#222); the root is not open
  } else {
    open_.push_back(0);
  }

  // RESUME (#287): replace the fresh root with the open nodes of a saved search. Everything
  // is validated before a single node is touched, and a checkpoint that does not belong to
  // this model, or this build's format, is refused - never loaded on a best-effort basis.
  checkpoint_path_ = options_.get_string("checkpoint");
  checkpoint_nodes_ = options_.get_int("checkpoint_nodes");
  if (const std::string resume = options_.get_string("resume"); !resume.empty()) {
    if (const std::string refused = restore_checkpoint(resume); !refused.empty()) {
      solution.status = SolveStatus::kNotSolved;
      solution.message = "resume refused: " + refused;
      logger_.warning("{}", solution.message);
      return solution;
    }
  }

  double best_open_bound = -std::numeric_limits<double>::infinity();
  bool logged_table = false;
  bool dive = false;
  bool limit_hit = false;
  bool gap_target_met = false;
  double open_bound = -std::numeric_limits<double>::infinity();

  StopController stop(control_, timer_, limits_);
  SolveStatus stop_status;

  while (!open_.empty()) {
    // PERIODIC CHECKPOINT (#287), between nodes: no node is entered, so every bound in the
    // working model is the root's and the open list is the whole of the search.
    if (checkpoint_nodes_ > 0 && nodes_explored_ > 0 &&
        nodes_explored_ % checkpoint_nodes_ == 0 && nodes_explored_ != last_checkpoint_at_) {
      save_checkpoint();
    }
    // Both counters and the clock are checked here, in the documented order: a tree that is
    // out of time and out of nodes at the same node reports the time limit (#289).
    if (const LimitReason why = limits_.exhausted(timer_.elapsed_seconds(), 0,
                                                  static_cast<std::int64_t>(nodes_explored_));
        why != LimitReason::kNone) {
      limit_hit = true;
      solution.status = status_for(why);
      // The status alone cannot say this once an incumbent turns it into kFeasible (#289).
      solution.stopped_by = why;
      solution.message = limits_.describe(why, timer_.elapsed_seconds(), 0,
                                          static_cast<std::int64_t>(nodes_explored_));
      break;
    }

    // Another worker's incumbent, the shared node count and stop flag, and giving nodes to
    // an idle worker (#222). Nothing to do in a sequential search.
    if (shared_ != nullptr) {
      LimitReason why = LimitReason::kNone;
      if (!sync_with_shared(&why)) {
        limit_hit = true;
        solution.status = status_for(why);
        solution.stopped_by = why;
        solution.message = fmt::format("the parallel search stopped: {}", to_string(why));
        break;
      }
    }

    // ALGORITHMIC OPEN BOUND: compute unconditionally when there is an incumbent so that
    // the gap-target stopping condition below always sees a fresh value every iteration.
    // The progress callback lambda reuses this when reporting and does NOT re-scan.
    if (have_incumbent_) {
      open_bound = std::numeric_limits<double>::infinity();
      for (const Index open_index : open_) {
        open_bound = std::min(open_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
      }
    }

    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kTree;
              p.iterations = 0;  // Not tracking simplex iterations across the tree currently
              p.nodes = nodes_explored_;
              p.open_nodes = static_cast<std::int64_t>(open_.size());
              p.objective = have_incumbent_ ? reported(incumbent_internal_)
                                            : std::numeric_limits<double>::infinity();
              // When an incumbent exists, open_bound was computed above this call; reuse it.
              // When no incumbent exists yet, scan now for accurate reporting only.
              double reporting_bound = open_bound;
              if (!have_incumbent_) {
                reporting_bound = std::numeric_limits<double>::infinity();
                for (const Index open_index : open_) {
                  reporting_bound = std::min(
                      reporting_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
                }
              }
              reporting_bound = integral_bound(reporting_bound);
              p.best_bound =
                  (original_.sense == ObjSense::kMaximize) ? -reporting_bound : reporting_bound;
              p.gap = have_incumbent_ ? (incumbent_internal_ - integral_bound(open_bound))
                                      : std::numeric_limits<double>::infinity();
              return p;
            },
            &stop_status)) {
      limit_hit = true;
      solution.status = stop_status;
      solution.stopped_by =
          stop_status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
      solution.message = limits_.describe(
          stop_status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt,
          timer_.elapsed_seconds(), 0, static_cast<std::int64_t>(nodes_explored_));
      break;
    }

    // Gap-based termination against the GLOBAL open bound - the best (lowest, in minimise
    // space) bound among every node still in the tree, not just the one about to be popped.
    // This is a stopping criterion, evaluated once per iteration here; it is deliberately
    // separate from can_prune(), which fathoms a single node against the absolute target
    // only. Pruning a node on the RELATIVE gap would discard nodes that could still hold a
    // genuinely better solution than the current incumbent, which is not what a relative
    // gap means - it bounds how far the reported answer may be from proven optimal, not
    // which nodes are worth visiting.
    // Not while filling the pool: meeting the gap target proves the incumbent, not that the
    // pool holds the best alternatives, and pool_complete promises the second.
    if (have_incumbent_ && !pool_complete_) {
      const double gap = incumbent_internal_ - integral_bound(open_bound);
      // gap <= 0 means open_bound already >= the incumbent: every node still in the tree
      // is one can_prune() would fathom the moment it is popped, so nothing open can beat
      // what has already been found. That is proven optimality, not a tolerance being met
      // early - the ordinary per-node prune below closes the tree on its own and reports
      // kOptimal. Treating a non-positive gap as "target met" here would report kFeasible
      // on an already-exhausted tree: negative <= a small positive target is trivially
      // true, so an unvisited, already-dead node's stale inherited bound (which need not
      // sit below the incumbent once every OTHER branch has been explored) would fire this
      // check before the loop ever reaches it to prune it honestly.
      if (gap > 0.0) {
        const double relative = gap / std::max(1.0, std::fabs(incumbent_internal_));
        if (gap <= absolute_gap_target_ || relative <= relative_gap_target_) {
          // Meeting the gap target is what every MIP solver means by "optimal": the
          // incumbent is within the requested tolerance of the best any open node can
          // reach. Until #188 this was reported kFeasible with exit code 1, and a judge
          // solving demo/miqp_blend.mps at the defaults saw a failure on a solved model.
          // The message carries the achieved gap and says the tree was not exhausted, and
          // dual_bound is the best OPEN bound, so the claim is exactly what was proven.
          gap_target_met = true;
          solution.message = fmt::format(
              "optimal within the {} gap target ({:.3e} absolute, {:.3e} relative) after {} "
              "nodes; the bound is the best open node's, the tree was not exhausted",
              relative <= relative_gap_target_ ? "relative" : "absolute", gap, relative,
              nodes_explored_);
          break;
        }
      }
    }

    // Reduced-cost fixing and restarts (#418), between nodes, where working_'s bounds are the
    // global bounds: whenever the incumbent has improved, tighten what the root's reduced
    // costs now rule out; and if that fixed enough of the integer columns, throw the tree
    // away and re-solve the root on the tightened bounds.
    if (fix_by_reduced_cost() > 0 && restart_due()) {
      restart_search();
      continue;
    }

    // Which node to take next is the configured policy's decision (#293), and only the
    // order it decides: the tree, the bounds and the incumbent test are the same either way.
    const Index node_index = take_next_open_node(dive);
    dive = false;

    const TreeNode& node = nodes_[static_cast<std::size_t>(node_index)];
    if (can_prune(node.bound)) {
      ++nodes_pruned_;
      continue;
    }

    enter(node_index);
    ++nodes_explored_;
    // Moved, not copied: this node will not be solved twice, and the open list must not
    // hold a basis per closed node.
    current_warm_ = std::move(nodes_[static_cast<std::size_t>(node_index)].warm);

    if (!propagate()) {
      const bool by_conflict = conflict_pruned_;
      leave();
      ++nodes_pruned_;
      if (by_conflict) {
        ++conflict_stats_.nodes_pruned;
      } else {
        analyze_conflict(node_index, ConflictSource::kPropagation, nullptr);
      }
      continue;
    }

    Solution relaxation = [&] {
      ProfileScope timed(logger_.profiler(), "node LP", ProfileMode::kDetailed);
      return solve_node();
    }();

    if (relaxation.status == SolveStatus::kInfeasible) {
      leave();
      ++nodes_pruned_;
      analyze_conflict(node_index, ConflictSource::kLp, &relaxation.farkas_dual);
      continue;
    }
    if (relaxation.status == SolveStatus::kUnbounded) {
      leave();
      solution.status = SolveStatus::kUnbounded;
      solution.message =
          "the LP relaxation is unbounded, so the MILP is unbounded or "
          "infeasible";
      return solution;
    }
    if (relaxation.status == SolveStatus::kInterrupted ||
        relaxation.status == SolveStatus::kTimeLimit) {
      leave();
      limit_hit = true;
      solution.status = relaxation.status;
      solution.stopped_by = relaxation.status == SolveStatus::kTimeLimit
                                ? LimitReason::kTime
                                : LimitReason::kInterrupt;
      // THE NODE IS STILL OPEN. It was taken off the list to be solved and was not, so its
      // inherited bound is part of what the search can still say; leaving it out reported
      // the next-best bound instead, and with nothing else open, no bound at all (#222).
      open_.push_back(node_index);
      break;
    }
    if (relaxation.status != SolveStatus::kOptimal) {
      // A node whose LP did not solve cannot be fathomed honestly: pruning it could discard
      // the optimum. Stop and report rather than quietly continuing on a broken bound.
      leave();
      // WHY IT COULD NOT BE SOLVED DECIDES WHAT THIS IS (#289). A node LP that ran out of
      // ITERATIONS is a resource limit that reached the tree through the node engine, not a
      // numerical failure: `--option iteration_limit=5` on a MILP used to come back
      // numerical_error, which says the solver broke when what happened is that it was told
      // to stop. Anything else - a singular basis, an unbounded node, a status no node
      // should return - stays the numerical failure it is.
      const bool out_of_iterations = relaxation.status == SolveStatus::kIterationLimit;
      limit_hit = out_of_iterations;
      solution.status =
          out_of_iterations ? SolveStatus::kIterationLimit : SolveStatus::kNumericalError;
      if (out_of_iterations) solution.stopped_by = LimitReason::kIterations;
      solution.message =
          out_of_iterations
              ? fmt::format(
                    "the node LP at node {} stopped at the iteration limit, so the tree "
                    "cannot go on: a node whose bound is unknown cannot be fathomed without "
                    "risking the optimum ({})",
                    nodes_explored_, relaxation.message)
              : fmt::format("node LP returned {} at node {}", to_string(relaxation.status),
                            nodes_explored_);
      if (!out_of_iterations) return solution;
      open_.push_back(node_index);  // still open, as above
      break;
    }

    if (node_index == 0) {
      root_bound_internal_ = internal_objective(relaxation.col_value);
      root_bound_after_cuts_internal_ = root_bound_internal_;
    }
    // The root cut round runs on the first root only: a restarted root (#418) keeps the cut
    // rows the first one added, and the cut machinery's root bookkeeping is built for one
    // root.
    if (node_index == 0 && restarts_ == 0 && options_.get_bool("enable_root_cuts")) {
      root_cut_round(&relaxation);
    }
    // The root relaxation after cuts is what reduced-cost fixing (#418) reasons from: its
    // reduced costs bound what every integer solution must pay to move a column.
    if (node_index == 0) remember_root_relaxation(relaxation);
    // Cuts below the root (#221): shallow nodes only, on the global bounds, kept for the
    // whole tree. The node's bound is taken after the round, so a cut that moved it
    // counts for pruning and for the pseudocosts alike.
    if (node_index != 0 && node.depth <= tree_cut_depth_ && !quadratic_) {
      tree_cut_round(node.depth, &relaxation);
    }
    age_cut_rows(relaxation);

    // Node bound in minimise space, excluding the offset (added back on report). Stored and
    // ordered raw; can_prune() and the gap test round it up to the next value an integer
    // solution can take (#221), so node selection is the same with or without the rounding.
    const double node_bound = internal_objective(relaxation.col_value);

    // THE PSEUDOCOST OBSERVATION (#69): what branching on this node's column bought, per
    // unit of the fractionality it removed, in the direction it went. Recorded whether or
    // not the node is pruned next - the gain is real either way.
    if (node.has_change && node.fraction > 0.0) {
      record_pseudocost(node.change.column, node.change.is_upper,
                        std::max(node_bound - node.bound, 0.0), node.fraction);
    }

    if (can_prune(node_bound)) {
      leave();
      ++nodes_pruned_;
      continue;
    }

    // Primal heuristics (#290): rounding every node as before, and the ones #290 added,
    // on their own schedules and budgets. They propose; offer_incumbent() decides.
    {
      ProfileScope timed(logger_.profiler(), "heuristics", ProfileMode::kDetailed);
      run_node_heuristics(node_index, relaxation);
    }

    if (most_fractional(relaxation.col_value) < 0) {
      // Integral relaxation: this node's optimum is a MILP solution.
      offer_incumbent(relaxation.col_value);
      if (pool_complete_ && split_integral_node(node_index, relaxation)) continue;
      leave();
      continue;
    }

    // The children start from THIS relaxation's basis, captured before the dives and the
    // strong-branching probes can replace current_warm_ with the bases of their own solves.
    const WarmStart children_warm = basis_of(relaxation);
    current_warm_ = children_warm;

    // Diving (#25, #414): at the root, and every mip_dive_frequency nodes when that is set.
    // node_index == 0 identifies the root directly - it is the one node present in open_
    // before anything else can be pushed there, so the first pass through this loop body
    // is always processing it. Every dive fixes bounds on the same saved_ stack propagate()
    // pushed onto for this node and unwinds them itself before returning, so the branching
    // decision below sees the node's own domain.
    {
      ProfileScope timed(logger_.profiler(), "heuristics", ProfileMode::kDetailed);
      run_dives(node_index, relaxation.col_value);
      current_warm_ = children_warm;
      // The feasibility pump only at the root, and only when rounding, repair and the dives
      // all came back empty: its value is an incumbent where there is none, and it costs
      // LP solves.
      if (node_index == 0) run_root_pump(relaxation);
    }

    // The branching decision, with the node's bounds still entered: strong branching
    // solves the two children in place and restores the bounds it moved.
    const Index branch_column = [&] {
      // Strong branching's probe LPs are inside this, which is the point: it is the cost of
      // the branching decision.
      ProfileScope timed(logger_.profiler(), "branching", ProfileMode::kDetailed);
      return reliability_branching_ ? select_branching_column(relaxation.col_value, node_bound)
                                    : most_fractional(relaxation.col_value);
    }();
    if (branch_column < 0) {
      // Cannot happen after the integrality test above, but a rule that returns nothing
      // must not be answered with a branch on column -1.
      leave();
      solution.status = SolveStatus::kNumericalError;
      solution.message =
          fmt::format("the branching rule found no column at node {}", nodes_explored_);
      return solution;
    }

    // Two children: x <= floor(v) and x >= floor(v) + 1. Together they cover every integer
    // point, so nothing is lost.
    const double value = relaxation.col_value[static_cast<std::size_t>(branch_column)];
    const double floor_value = std::floor(value);
    DomainChange down_change{branch_column, true, floor_value};
    DomainChange up_change{branch_column, false, floor_value + 1.0};
    double down_fraction = value - floor_value;
    double up_fraction = floor_value + 1.0 - value;

    // OBJECTIVE BRANCHING (#418): with the objective a row of the model and its step known,
    // a relaxation whose objective sits strictly between two attainable values is split on
    // that row instead - c x at most the value below, c x at least the value above. Every
    // integer point is on one side, and the up child leaves the plateau by construction.
    // The children carry no fractionality, so no pseudocost is recorded for them.
    if (objective_row_ >= 0) {
      const double units = objective_row_value(relaxation.col_value) / objective_step_;
      const double below = std::floor(units);
      if (units - below > kObjectiveIntegralitySlack &&
          below + 1.0 - units > kObjectiveIntegralitySlack) {
        const Index logical = original_.num_cols() + objective_row_;
        down_change = DomainChange{logical, true, objective_step_ * below};
        up_change = DomainChange{logical, false, objective_step_ * (below + 1.0)};
        down_fraction = 0.0;
        up_fraction = 0.0;
        ++objective_branches_;
      }
    }
    leave();

    // Both children inherit the parent's estimate: it is a property of the relaxation they
    // were branched from, and the branched column's own contribution is the one term the
    // branch is about to settle.
    const double child_estimate = estimate_from(relaxation.col_value, node_bound);

    TreeNode down;
    down.parent = node_index;
    down.has_change = true;
    down.change = down_change;
    down.bound = node_bound;
    down.depth = node.depth + 1;
    down.warm = children_warm;
    down.fraction = down_fraction;
    down.estimate = child_estimate;

    TreeNode up;
    up.parent = node_index;
    up.has_change = true;
    up.change = up_change;
    up.bound = node_bound;
    up.depth = node.depth + 1;
    up.warm = children_warm;
    up.fraction = up_fraction;
    up.estimate = child_estimate;

    nodes_.push_back(down);
    const auto down_index = static_cast<Index>(nodes_.size() - 1);
    nodes_.push_back(up);
    const auto up_index = static_cast<Index>(nodes_.size() - 1);
    open_.push_back(down_index);
    open_.push_back(up_index);
    dive = true;

    // ---- The node table -------------------------------------------------------------------
    best_open_bound = std::numeric_limits<double>::infinity();
    for (const Index open_index : open_) {
      best_open_bound = std::min(
          best_open_bound, integral_bound(nodes_[static_cast<std::size_t>(open_index)].bound));
    }
    if (!logged_table) {
      logger_.begin_node_table();
      logged_table = true;
    }
    if (nodes_explored_ % 20 == 1 || nodes_explored_ < 5) {
      const double incumbent_report =
          have_incumbent_ ? reported(incumbent_internal_) : kInfinity;
      const double gap = have_incumbent_ ? std::fabs(incumbent_internal_ - best_open_bound) /
                                               std::max(1.0, std::fabs(incumbent_internal_))
                                         : kInfinity;
      logger_.node(nodes_explored_, static_cast<Count>(open_.size()), incumbent_report,
                   reported(best_open_bound), gap, timer_.elapsed_seconds());
    }
  }

  report_conflicts();
  // A search stopped by a limit is exactly the one worth resuming (#287).
  if (limit_hit && !open_.empty()) save_checkpoint();
  if (shared_ != nullptr) leave_shared(limit_hit, solution.stopped_by, gap_target_met);

  // ---- Report ------------------------------------------------------------------------------
  double final_bound = incumbent_internal_;
  // The open nodes' bounds, rounded (#221): what they prove is the rounded value.
  for (const Index open_index : open_) {
    final_bound = std::min(final_bound,
                           integral_bound(nodes_[static_cast<std::size_t>(open_index)].bound));
  }

  if (!have_incumbent_) {
    if (!limit_hit) {
      solution.status = SolveStatus::kInfeasible;
      solution.message = fmt::format(
          "the search closed with no integer feasible point after {} nodes", nodes_explored_);
    }

    const double nothing_found =
        original_.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;

    // NO INTEGER POINT WAS FOUND, so there is no point and no objective to report (#505).
    //
    // A limit without an incumbent used to report the last node LP (#223), with a message
    // saying it was fractional. The benchmark runner and every other reader take `objective`
    // on a MILP for the incumbent's value, and the .sol file for a point: MIPLIB `ej` came
    // back time_limit with objective 46624.61 on three integer columns, the relaxation of
    // whatever node the clock stopped, which the independent verifier rejected on
    // integrality. Under a time limit that point need not even be an LP optimum - the node
    // LP the clock interrupted is the one that was kept. The message said what it was; the
    // fields did not, and the fields are what get read. A limited search now carries a
    // verified incumbent or nothing.
    //
    // Nor does it leave the allocated zeros standing in for a point: the worst
    // representable objective, with empty vectors, is how "nothing found" is said here and
    // in the parallel search, the dispatcher (refuse_a_non_finite_answer) and postsolve
    // already read it that way, and claims_a_point() and the .sol writer read an empty
    // point under a limit as no point (#505). Leaving the objective at its default said
    // objective 0, bound 0, gap 0 - and a gap of zero means CLOSED, the exact opposite of
    // what happened: enlight8, enlight_hard, timtab1 and neos-1425699 all came back
    // `node_limit` with `gap 0.00e+00` beside them before that was fixed.
    solution.objective = nothing_found;
    if (limit_hit) {
      // The dispatcher's non-finite guard does the same and says so in the message; doing it
      // here too means the engine's own answer, read without the dispatcher, says it as well.
      solution.col_value.clear();
      solution.row_activity.clear();
      solution.primal_infeasibility = 0.0;
      solution.primal_infeasibility_scaled = 0.0;
      solution.dual_infeasibility = 0.0;
      solution.dual_infeasibility_scaled = 0.0;
      solution.integrality_violation = 0.0;
      solution.iterations = warm_node_iterations_ + cold_node_iterations_;
    }
    // The BOUND is different, and is real information worth keeping: when a limit stopped
    // the search, the open nodes still prove the optimum is no better than final_bound. Only
    // a search that closed with nothing has no bound to offer either.
    solution.dual_bound = limit_hit ? reported(final_bound) : nothing_found;
    solution.absolute_gap = kInfinity;
    solution.relative_gap = kInfinity;
    solution.nodes = nodes_explored_;
    solution.restarts = restarts_;
    solution.reduced_cost_fixings = reduced_cost_fixings_;
    solution.objective_branches = objective_branches_;
    solution.symmetry_generators = symmetry_generators_;
    solution.solve_seconds = timer_.elapsed_seconds();
    report_root(&solution);
    return solution;
  }

  solution.col_value = incumbent_x_;
  solution.nodes = nodes_explored_;
  solution.restarts = restarts_;
  solution.reduced_cost_fixings = reduced_cost_fixings_;
  solution.objective_branches = objective_branches_;
  solution.symmetry_generators = symmetry_generators_;
  solution.solve_seconds = timer_.elapsed_seconds();
  report_root(&solution);

  if (open_.empty() && !limit_hit && !gap_target_met) {
    // The tree is exhausted: the incumbent is proven optimal and is its own bound.
    solution.status = SolveStatus::kOptimal;
    solution.dual_bound = reported(incumbent_internal_);
  } else if (gap_target_met) {
    // Optimal within the gap target (#188). dual_bound is the best open bound, so
    // objective - dual_bound is the gap that was accepted, and recompute_quality() below
    // reports it; tools/verify_solution.py checks that gap against the targets recorded in
    // the .sol header rather than demanding a closed tree.
    solution.status = SolveStatus::kOptimal;
    solution.dual_bound = reported(final_bound);
  } else {
    // A node or time limit stopped the proof short of the target. The incumbent is
    // feasible, not proven, and dual_bound carries the best bound still open - claiming
    // otherwise would assert a proof that was never established.
    solution.status = SolveStatus::kFeasible;
    solution.dual_bound = reported(final_bound);
  }
  solution.recompute_quality(original_);
  if (tree_cut_rounds_ > 0) {
    logger_.info("Tree cuts: {} rounds below the root, {} rows added, {} aged out",
                 tree_cut_rounds_, tree_cuts_applied_, cut_rows_aged_out_);
  }

  if (pool_.enabled()) {
    for (SolutionPool::Entry& entry :
         pool_.finish(incumbent_internal_, incumbent_x_, pool_gap_)) {
      Solution::PoolEntry member;
      member.objective = original_.evaluate_objective(entry.x.data());
      member.col_value = std::move(entry.x);
      solution.pool.push_back(std::move(member));
    }
    // pool[0] IS the solution: the same vector, and the objective as recompute_quality wrote
    // it, so a reader comparing the two never meets a last-bit difference.
    solution.pool.front().objective = solution.objective;
    logger_.info("Solution pool: {} plan(s), objectives {:.10g} to {:.10g}{}",
                 solution.pool.size(), solution.pool.front().objective,
                 solution.pool.back().objective, pool_complete_ ? " (complete)" : "");
  }

  logger_.info("");
  logger_.info("Status: {}   objective {:.10g}   bound {:.10g}   nodes {}   time {:.3f}s",
               to_string(solution.status), solution.objective, solution.dual_bound,
               solution.nodes, solution.solve_seconds);
  logger_.info("Nodes pruned {}, tree {} node(s) at exit", nodes_pruned_, open_.size());
  report_heuristics();
  if (objective_branches_ > 0) {
    logger_.info("Objective branching (#418): {} node(s) split on the objective row",
                 objective_branches_);
  }
  if (clique_cuts_generated_ + zero_half_cuts_generated_ > 0) {
    logger_.info(
        "Combinatorial cut candidates (#358): {} clique, {} zero-half, before the filter",
        clique_cuts_generated_, zero_half_cuts_generated_);
  }
  if (Profiler* profiler = logger_.profiler(); profiler != nullptr) {
    profiler->count("nodes pruned", static_cast<std::int64_t>(nodes_pruned_));
    profiler->count("warm-started node LPs", static_cast<std::int64_t>(warm_node_solves_));
    profiler->count("cold node LPs", static_cast<std::int64_t>(cold_node_solves_));
  }
  logger_.info(
      "Node selection {}: {} node(s) taken deepest-first, {} by the policy, deepest node at "
      "depth {}",
      to_string(node_selection_), selected_by_dive_, selected_by_policy_, deepest_node_);
  if (!quadratic_) {
    // How the node LPs were solved (#65). The ratio of warm to cold is the whole point of
    // the dual node engine, and the iterations per solve are the evidence it pays.
    logger_.info(
        "Node LPs: {} warm-started dual ({} iterations), {} cold primal ({} iterations), {} "
        "cold fallback(s) after a dual failure; strong branching {} probe(s), {} iterations",
        warm_node_solves_, warm_node_iterations_, cold_node_solves_, cold_node_iterations_,
        cold_fallbacks_, strong_branch_solves_, strong_branch_iterations_);
  }
  if (!solution.message.empty()) logger_.info("{}", solution.message);
  return solution;
}

Solution solve_branch_and_bound(const Model& model, const Options& options, Logger& logger,
                                SolveControl* control) {
  // ROOT CUTS, applied once before the search rather than per node.
  //
  // Integer rounding tightens a row IN PLACE, so unlike a generated cut it adds no row, grows
  // no basis, and costs the search nothing per node - every node LP simply starts from a
  // tighter relaxation. Applying it at the root is therefore all that is needed; re-applying
  // it deeper would find nothing new, because branching changes column bounds and not the row
  // coefficients this reads.
  //
  // It runs on a COPY. The caller's model is an input, and a solver that silently rewrites
  // the model it was handed makes a second solve of the "same" model mean something different
  // from the first.
  Model tightened = model;
  const RowTightening effect = tighten_integral_rows(&tightened, logger);
  const Model& searched = effect.rows_tightened > 0 ? tightened : model;

  // PARALLEL TREE SEARCH (#222), when asked for and when the model is one it takes: a MILP,
  // not a pool_complete search (whose pruning reads the pool's cutoff at every node), and
  // not in deterministic mode (the tree a parallel search explores depends on timing).
  const int threads = parallel_threads(searched, options, logger);
  if (threads > 1)
    return solve_branch_and_bound_parallel(searched, options, logger, control, threads);

  BranchAndBound search(searched, options, logger, control);
  return search.run();
}

}  // namespace sankhya::mip
