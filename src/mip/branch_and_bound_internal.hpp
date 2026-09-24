// SPDX-License-Identifier: Apache-2.0
// SANKHYA - internal state shared by branch_and_bound.cpp and branch_and_bound_node.cpp
// (issue #262 split this out of one 1,464-line file so neither half stays over the
// ~600-line rule; see ENGINEERING_RULES.md). Not a public header - nothing outside
// src/mip/ includes this.
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
#pragma once

#include "sankhya/mip.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

#include "certificate_writer.hpp"
#include "checkpoint.hpp"
#include "conflict.hpp"
#include "cut_selection.hpp"
#include "cuts.hpp"
#include "debug_solution.hpp"
#include "flow_cover_cuts.hpp"
#include "heuristics.hpp"
#include "mir_cuts.hpp"
#include "solution_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/resource_limits.hpp"
#include "../core/stop_controller.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/factor_cache.hpp"
#include "simplex/primal_simplex.hpp"

namespace sankhya::mip {

class SharedSearch;
struct SubtreeSpec;

/// One tightened bound, recorded so entering a node can be undone rather than rebuilt.
struct DomainChange {
  Index column = -1;
  bool is_upper = false;
  double value = 0.0;
};

/// A node holds only its OWN bound change and a link to its parent. The full domain is
/// recovered by walking to the root, which is why the tree costs O(depth) per node instead
/// of O(columns).
/// Which open node the search takes next (#293).
///
/// Node selection changes the ORDER the tree is explored in and nothing else: the same nodes
/// exist, the same bounds prune them, and the optimum is the optimum under every policy. What
/// it does change is when the first incumbent arrives, how fast the global bound moves, and
/// how many nodes stay open at once - which is why one fixed policy suits no model.
///
/// Reference: Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 6, for the
/// best-estimate rule and the dive-then-best-bound hybrid.
enum class NodeSelection {
  /// Dive to a leaf, then best-bound. An early incumbent is what makes every later bound able
  /// to prune, and best-bound afterwards keeps the tree from growing where it cannot pay.
  kHybrid,
  /// Always the smallest bound. Proves optimality in the fewest nodes and keeps the widest
  /// tree open, so the memory is the price.
  kBestBound,
  /// Always the deepest node. Narrow tree, fast to a first incumbent, and the global bound
  /// barely moves until the search backtracks.
  kDepthFirst,
  /// Smallest pseudocost estimate of where the node's subtree will end up, which is a guess
  /// at where a GOOD incumbent is rather than at where the bound is.
  kBestEstimate,
};

[[nodiscard]] const char* to_string(NodeSelection selection) noexcept;

/// select_branching_column()'s answers other than a column (#502).
inline constexpr Index kBranchIntegral = -1;  ///< the (re-solved) relaxation is integral
inline constexpr Index kBranchPruned = -2;    ///< the node is fathomed: infeasible or bounded

struct TreeNode {
  Index parent = -1;
  DomainChange change;
  bool has_change = false;
  double bound = 0.0;  ///< the LP bound inherited from the parent, in minimise space
  /// The parent's LP objective as computed, in minimise space (#519). Equal to `bound`
  /// unless safe_bounds replaced `bound` with the proved bound, which may be -inf. The
  /// pseudocost observation reads this one: it measures what the branching bought in the
  /// LP, and a proved bound of -inf would record an infinite gain. NaN where a path does
  /// not set it (a restored checkpoint, a parallel seed); the observation then uses `bound`.
  double parent_lp_bound = std::numeric_limits<double>::quiet_NaN();
  Index depth = 0;
  /// The parent's optimal basis, as statuses (#65). One bound differs between parent and
  /// child, so this basis is dual feasible at the child and the dual simplex reaches the
  /// child's optimum in a few pivots. Moved out when the node is processed, so an open
  /// node costs n + m bytes and a closed one nothing.
  WarmStart warm;
  /// How far the branching moved the column from the parent's relaxation value: v - floor(v)
  /// for the down child, ceil(v) - v for the up child. The pseudocost observation (#69) is
  /// this node's bound gain divided by it.
  double fraction = 0.0;
  /// Where the pseudocosts expect this subtree's integer answer to land (#293), computed from
  /// the parent's relaxation when the node is created. Only kBestEstimate reads it; it equals
  /// `bound` until the pseudocosts have seen anything.
  double estimate = 0.0;
};

/// Convergence tolerance for a QP node relaxation in an MIQP search.
///
/// Deliberately far tighter than the gap targets the search compares bounds against. The
/// bound a first-order method reports is only accurate to its own tolerance, and branch and
/// bound FATHOMS on that bound - so the error has to be small enough that widening
/// can_prune()'s margin by it does not stop the search closing.
constexpr double kMiqpNodeTolerance = 1e-10;
/// How far below the next multiple of the objective step a relaxation bound may sit and
/// still be rounded up to it (#221), in units of the step: an LP bound carries the
/// simplex's tolerance, and 1e-6 of a step is well above it and well below one step.
constexpr double kObjectiveIntegralitySlack = 1e-6;

/// `bound` rounded up to the next multiple of `step` (#221), or unchanged when the step is
/// unknown (0). BranchAndBound::integral_bound below and the parallel driver (#222) both use
/// it, so a bound means the same thing whichever of them reports it.
[[nodiscard]] inline double round_up_to_step(double bound, double step) {
  if (step <= 0.0 || !std::isfinite(bound)) return bound;
  const double units = bound / step;
  const double slack = std::max(kObjectiveIntegralitySlack, 1e-9 * std::fabs(units));
  return step * std::ceil(units - slack);
}

/// Iteration cap for one node QP. Condat-Vu has no warm start, so every node pays a cold
/// solve; this keeps a single pathological node from consuming the whole time limit while
/// still being generous enough to reach kMiqpNodeTolerance on the node sizes this handles.
constexpr std::int64_t kMiqpNodeIterationLimit = 2000000;

/// Distance from the nearest integer.
inline double fractionality(double value) {
  return std::fabs(value - std::round(value));
}

class BranchAndBound {
 public:
  BranchAndBound(const Model& model, const Options& options, Logger& logger,
                 SolveControl* control)
      : original_(model),
        working_(model),
        options_(options),
        logger_(logger),
        control_(control) {
    integrality_tolerance_ = options.get_double("integrality_tolerance");
    relative_gap_target_ = options.get_double("mip_relative_gap");
    absolute_gap_target_ = options.get_double("mip_absolute_gap");
    // One interpretation of every limit, shared with every other engine (#289).
    limits_ = ResourceLimits(options, logger);
    time_limit_ = options.get_double("time_limit");
    sense_ = model.sense_multiplier();

    node_engine_dual_ = options.get_string("mip_node_engine") != "primal";
    const std::string selection = options.get_string("mip_node_selection");
    if (selection == "best-bound") {
      node_selection_ = NodeSelection::kBestBound;
    } else if (selection == "depth-first") {
      node_selection_ = NodeSelection::kDepthFirst;
    } else if (selection == "best-estimate") {
      node_selection_ = NodeSelection::kBestEstimate;
    } else {
      node_selection_ = NodeSelection::kHybrid;
    }
    reliability_branching_ = options.get_string("mip_branching") != "most-fractional";
    safe_bounds_ = options.get_bool("safe_bounds");
    certificate_path_ = options.get_string("write_certificate");
    const auto columns = static_cast<std::size_t>(model.num_cols());
    pseudo_down_sum_.assign(columns, 0.0);
    pseudo_up_sum_.assign(columns, 0.0);
    pseudo_down_count_.assign(columns, 0);
    pseudo_up_count_.assign(columns, 0);

    // Node LPs are solved silently; the node table is the log the user wants, not several
    // hundred simplex iteration tables.
    node_options_ = options;
    node_options_.set_bool("log_to_console", false);

    // MIQP: the node relaxation is a QP rather than an LP (#58 names MIQP as the class this
    // dispatcher refused). The Hessian is a property of the model, not of a node - branching
    // only moves bounds - so this is decided once here.
    quadratic_ = model.has_quadratic_objective();
    if (quadratic_) {
      // A NODE BOUND FROM A FIRST-ORDER METHOD IS NOT EXACT, and branch and bound prunes on
      // it. The simplex returns a vertex whose objective is exact to rounding; Condat-Vu
      // returns a point converged to a tolerance, so a node bound can be optimistic by about
      // that much - and an optimistic bound can fathom the subtree containing the true
      // optimum, which is the one error this search must never make.
      //
      // Two things follow. The node tolerance is tightened well below the gap targets the
      // search compares against, and can_prune() widens its margin by that tolerance so a
      // node is only fathomed when it loses by more than the bound could be wrong by.
      node_options_.set_double("qp_tolerance", kMiqpNodeTolerance);
      node_options_.set_int("iteration_limit", kMiqpNodeIterationLimit);
    }

    for (Index j = 0; j < model.num_cols(); ++j) {
      if (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) {
        integer_columns_.push_back(j);
      }
    }

    // The solution pool (#225). Built after integer_columns_, because an assignment is what
    // makes two points the same plan.
    pool_ =
        SolutionPool(integer_columns_, static_cast<std::size_t>(options.get_int("pool_size")),
                     options.get_bool("pool_diversity"));
    pool_gap_ = options.get_double("pool_gap");
    // A complete search with nowhere to keep what it finds would enumerate for nothing.
    pool_complete_ = options.get_bool("pool_complete") && pool_.enabled();
    init_conflicts();
  }

  Solution run();

  /// Run as one worker of the parallel search (#222): share the incumbent, the node count,
  /// the pool and the pseudocosts through `shared`, and start from `seed` - a subtree some
  /// other worker gave away - instead of the root when it is not null.
  void attach(SharedSearch* shared, const SubtreeSpec* seed) {
    shared_ = shared;
    seed_ = seed;
  }

 private:
  // ---- Parallel tree search (branch_and_bound_parallel.cpp, #222) ----------------------

  /// Put the seed's chain into nodes_ and its last node into open_.
  void plant_seed();
  /// Once per node, at the top of the loop: take a better incumbent from the other workers,
  /// count nodes, give nodes away to an idle worker. False when the search must stop, with
  /// the reason in `why`.
  bool sync_with_shared(LimitReason* why);
  void donate_open_nodes();
  /// Merge what this worker observed into the shared pseudocosts and take the merged ones.
  void sync_pseudocosts();
  /// At the end of run(): what this subtree leaves open, and the pseudocosts it learned.
  void leave_shared(bool limit_hit, LimitReason why, bool gap_target_met);

  /// Apply a node's whole domain, walking from the node to the root.
  void enter(Index node_index);
  /// Restore the domain saved by the last enter().
  void leave();

  /// Tighten bounds from row activities until nothing moves. Returns false when the node is
  /// proved infeasible in the process, which fathoms it without an LP solve at all.
  bool propagate();
  /// One row of propagate(): the activity test, then every column's implied bounds and
  /// integer rounding. Returns false when the row proves the node infeasible; sets *changed
  /// when an implied bound (not a rounding) moved, which is what the sweep loop counts.
  bool propagate_row(Index row, const CsrView& by_row, bool* changed);
  /// mip_incremental_propagation (#502): propagate() to a fixpoint through a worklist of the
  /// rows whose columns moved, instead of three full sweeps. Same return contract.
  bool propagate_to_fixpoint(const CsrView& by_row);

  /// Tighten a bound AND record the old value so leave() can undo it.
  ///
  /// Every write to working_.col_lower / col_upper outside enter() must go through these.
  /// Propagation that writes directly leaks its tightenings into sibling and later nodes,
  /// permanently shrinking the tree's domain and discarding feasible integer points - the
  /// search then proves that the second-best answer is optimal, which looks completely
  /// correct from outside.
  /// The working bound a change names: a column's, or, for an index at or past the column
  /// count, a row's in the [A | -I] convention - the objective row's, under objective
  /// branching (#418). enter(), leave() and the dives' unwind all go through this.
  double& bound_of(const DomainChange& change) {
    const Index n = original_.num_cols();
    if (change.column >= n) {
      const auto r = static_cast<std::size_t>(change.column - n);
      return change.is_upper ? working_.row_upper[r] : working_.row_lower[r];
    }
    const auto u = static_cast<std::size_t>(change.column);
    return change.is_upper ? working_.col_upper[u] : working_.col_lower[u];
  }

  void tighten_lower(std::size_t column, double value) {
    saved_.push_back(
        DomainChange{static_cast<Index>(column), false, working_.col_lower[column]});
    working_.col_lower[column] = value;
  }
  void tighten_upper(std::size_t column, double value) {
    saved_.push_back(
        DomainChange{static_cast<Index>(column), true, working_.col_upper[column]});
    working_.col_upper[column] = value;
  }

  /// The integer column furthest from integral, or -1 when the point is integral.
  [[nodiscard]] Index most_fractional(const std::vector<double>& x) const;

  /// Reliability branching (#69): the column with the best pseudocost product score, with
  /// strong branching on columns whose pseudocosts are not yet reliable. Requires the
  /// node's bounds to be entered and current_warm_ to hold its relaxation's basis. Returns
  /// -1 when the point is integral.
  ///
  /// With `fixes` non-null (mip_strong_branch_fix, #502), a candidate whose probe proved
  /// one side infeasible is not scored: the bound that closes that side is appended to
  /// *fixes instead, and a candidate with BOTH sides infeasible returns kBranchPruned.
  [[nodiscard]] Index choose_branching_column(const std::vector<double>& x, double node_bound,
                                              std::vector<DomainChange>* fixes);
  /// The branching decision at a node: choose_branching_column(), and under
  /// mip_strong_branch_fix the rounds of fixing and re-solving (#502). A round that fixes
  /// anything replaces `relaxation`, `node_bound` and `prune_bound` with the re-solved
  /// node's, leaves the fixes entered (on saved_, so leave() undoes them) and records them in
  /// strong_fixes_ for the caller to hang the children under. Returns a column, or
  /// kBranchIntegral / kBranchPruned when the re-solved node is integral / fathomed.
  [[nodiscard]] Index select_branching_column(Solution& relaxation, double& node_bound,
                                              double& prune_bound);
  /// Hang strong_fixes_ under `node_index` as a chain of never-opened link nodes, the way
  /// split_integral_node() chains its fixes, and return the tail: children created under
  /// it inherit every fix through enter()'s walk to the root. `node_index` when none.
  [[nodiscard]] Index link_strong_fixes(Index node_index, double bound);

  /// Fold one observed bound gain into a column's pseudocost.
  void record_pseudocost(Index column, bool downward, double gain, double fraction);

  /// Take the next open node under the configured policy (#293), removing it from `open_`.
  /// `diving` is the hybrid's signal that the previous node just produced children.
  [[nodiscard]] Index take_next_open_node(bool diving);
  /// Add a node to `open_`, keeping the heap order under mip_heap_open_list (#502).
  void push_open(Index node_index);
  /// Restore the heap order after `open_` was rebuilt wholesale (a checkpoint restore).
  void rebuild_open_heap();
  /// mip_heap_open_list (#502): `open_` is kept as a binary heap under best-bound or
  /// best-estimate, in one worker only (parallel donation erases from the middle of it).
  [[nodiscard]] bool open_is_heap() const {
    return heap_open_list_ && (node_selection_ == NodeSelection::kBestBound ||
                               node_selection_ == NodeSelection::kBestEstimate);
  }
  /// The heap's ordering: `a` comes out AFTER `b`. The key is the policy's (bound or
  /// estimate) and ties go to the smaller node index - exactly the linear scan's order.
  [[nodiscard]] bool open_after(Index a, Index b) const {
    const TreeNode& na = nodes_[static_cast<std::size_t>(a)];
    const TreeNode& nb = nodes_[static_cast<std::size_t>(b)];
    const bool by_bound = node_selection_ == NodeSelection::kBestBound;
    const double ka = by_bound ? na.bound : na.estimate;
    const double kb = by_bound ? nb.bound : nb.estimate;
    return ka > kb || (ka == kb && a > b);
  }

  /// Where the pseudocosts expect a node branched from this relaxation to end up: the node's
  /// own bound plus, for every column still fractional, the cheaper of the two directions
  /// (Achterberg 2007, sec. 6.1). Returns `bound` unchanged when nothing is fractional.
  [[nodiscard]] double estimate_from(const std::vector<double>& x, double bound) const;

  /// Round the relaxation to the nearest integers and test the result. Cheap, and on models
  /// with a lot of structure it finds the incumbent that makes every later bound useful.
  void try_rounding(const std::vector<double>& x);

  /// Accept a candidate if it is integral, feasible and better than the incumbent.
  bool offer_incumbent(const std::vector<double>& x);

  // ---- Primal heuristics (#290, #414), in branch_and_bound_heuristics.cpp ---------------
  void init_heuristics();
  /// Offer a heuristic's candidate and count it against that heuristic.
  bool offer_from(std::size_t slot, const std::vector<double>& x);
  /// Rounding (every node), lock rounding (every node), repair and RENS (root) and RINS
  /// (scheduled), each on its own switch in schedule_.
  void run_node_heuristics(Index node_index, const Solution& relaxation);
  /// The diving family: every enabled rule, at the root and every mip_dive_frequency nodes.
  /// Each dive's fixes are undone before the next starts; current_warm_ is the caller's to
  /// restore. Requires the node's bounds to be entered.
  void run_dives(Index node_index, const std::vector<double>& x);
  /// One dive under `rule`, counted against `slot`: fix the column the rule picks to the
  /// integer it picks, re-solve, repeat until the point is integral, an LP dead-ends (once
  /// backtracked if mip_dive_backtrack is set) or the budget runs out. Leaves working_'s
  /// bounds as it found them.
  void dive(std::size_t slot, DiveRule rule, const std::vector<double>& start_x);
  /// Restore the bounds saved_ holds beyond `mark` and drop those entries: leave() for the
  /// tail of the stack only.
  void unwind_to(std::size_t mark);
  /// The feasibility pump at the root, only when nothing else found an incumbent.
  void run_root_pump(const Solution& relaxation);
  /// Feasibility Jump (#506): before the root LP when `from` is null, else from `from`
  /// rounded; the root only, and only when mip_heur_fj resolves on.
  void run_feasibility_jump(const std::vector<double>* from);
  void report_heuristics();
  // ---- Reduced-cost fixing and restarts (#418), in branch_and_bound_restart.cpp ---------
  /// Keep the root relaxation's reduced costs and basis: the material fixing works from.
  void remember_root_relaxation(const Solution& relaxation);
  /// Tighten, for the whole tree, the integer bounds the root reduced costs and the
  /// incumbent rule out. Between nodes only (no node entered); a no-op until the incumbent
  /// has improved since the last pass. Returns the bounds moved.
  Count fix_by_reduced_cost();
  /// Enough fixed since the root was last processed, within the restart budget.
  [[nodiscard]] bool restart_due() const;
  /// Discard the tree and start again from the root on the tightened bounds.
  void restart_search();
  // ---- Objective branching (#418), in branch_and_bound.cpp -------------------------------
  /// Append the objective as a free row of working_, once, before any cut row, so a node
  /// can bound it; sets objective_row_.
  void append_objective_row();
  /// c x at `x`, in the model's own units: what the objective row measures.
  [[nodiscard]] double objective_row_value(const std::vector<double>& x) const;
  // ---- Formulation symmetry (#413), in branch_and_bound.cpp ------------------------------
  /// Detect the model's symmetry and append the generators' ordering rows to working_,
  /// once, at the root, before any cut row.
  void append_symmetry_rows();
  // ---- Checkpoint and resume (#287), in branch_and_bound_checkpoint.cpp -----------------
  /// The search as it stands between nodes.
  [[nodiscard]] TreeCheckpoint make_checkpoint() const;
  /// Write it to checkpoint_path_ when one is set; a failed write is a warning.
  void save_checkpoint();
  /// Load, validate and rebuild the open nodes and the incumbent. Empty on success, the
  /// reason it was refused otherwise.
  [[nodiscard]] std::string restore_checkpoint(const std::string& path);

  /// pool_complete (#225): an integral relaxation closes a node for the OPTIMUM, not for the
  /// pool - the node's region can still hold the second-best assignment. Partition the rest
  /// of the region around the relaxation's point, every unfixed integer column at once (see
  /// the definition). Returns false, with the node's bounds still entered, when every integer
  /// column is fixed and the node really is a single assignment.
  bool split_integral_node(Index node_index, const Solution& relaxation);

  /// Is a bound worth exploring given the incumbent?
  /// Solve the current node's relaxation with whichever engine the model calls for.
  ///
  /// Both engines take the same Model and return the same Solution, which is what makes this
  /// a one-line choice rather than a second search. The QP path carries no basis, so nothing
  /// downstream may assume one - the diving heuristic and the branching rule both read
  /// col_value only, which they already did.
  [[nodiscard]] Solution solve_node() { return solve_node_with(node_options_); }

  [[nodiscard]] Solution solve_node_with(const Options& options) {
    if (quadratic_) return qp::solve_convex_qp(working_, options, logger_, control_);
    // WARM-STARTED DUAL SIMPLEX BELOW THE ROOT (#65). The basis in current_warm_ was
    // optimal for a problem that differs from this one by a bound or two, so it is dual
    // feasible here, which is exactly the state the dual simplex starts from. Measured
    // before this: every node was a cold primal solve from the slack basis.
    //
    // The primal stays as the fallback, cold, for a node the dual could not finish: a
    // numerical answer at a node cannot be fathomed honestly, and the search below stops
    // on it, so it is worth one more solve to avoid.
    if (node_engine_dual_ && !current_warm_.empty()) {
      Solution warm = solve_dual_simplex(working_, options, logger_, scaling_, control_,
                                         &current_warm_, factor_cache_.get());
      if (warm.status == SolveStatus::kOptimal || warm.status == SolveStatus::kInfeasible ||
          warm.status == SolveStatus::kUnbounded ||
          warm.status == SolveStatus::kIterationLimit) {
        ++warm_node_solves_;
        warm_node_iterations_ += warm.iterations;
        return warm;
      }
      logger_.verbose(
          "node LP: the warm-started dual simplex returned {}; re-solving cold "
          "with the primal simplex",
          to_string(warm.status));
      ++cold_fallbacks_;
    }
    Solution cold = solve_primal_simplex(working_, options, logger_, scaling_, control_,
                                         nullptr, factor_cache_.get());
    ++cold_node_solves_;
    cold_node_iterations_ += cold.iterations;
    return cold;
  }

  /// The basis a solved relaxation reports, or an empty start when it reports none.
  [[nodiscard]] static WarmStart basis_of(const Solution& relaxation) {
    WarmStart warm;
    if (relaxation.status != SolveStatus::kOptimal) return warm;
    for (const BasisStatus status : relaxation.col_status) {
      if (status == BasisStatus::kUnknown) return warm;
    }
    for (const BasisStatus status : relaxation.row_status) {
      if (status == BasisStatus::kUnknown) return warm;
    }
    warm.col_status = relaxation.col_status;
    warm.row_status = relaxation.row_status;
    return warm;
  }

  [[nodiscard]] bool can_prune(double bound) const {
    // Minimise space throughout: a node whose bound is no better than the incumbent, to
    // within the absolute gap target, cannot contain an improving solution.
    // The margin is widened for a QP node by the tolerance its bound is only accurate to.
    // Pruning too little costs nodes; pruning too much loses the optimum silently.
    const double margin =
        quadratic_ ? std::max(absolute_gap_target_, kMiqpNodeTolerance) : absolute_gap_target_;
    // Rounded up to the next value an integer solution can take (#221) before the test.
    const double rounded = integral_bound(bound);
    if (pool_complete_) return rounded >= pool_cutoff() - margin;
    if (!have_incumbent_) return false;
    return rounded >= incumbent_internal_ - margin;
  }

  /// What a node must beat to matter when the search is filling the pool (#225): the worst
  /// member of a full pool, and never worse than pool_gap past the incumbent. +infinity while
  /// neither applies, which means nothing is pruned - the price of a complete pool. The
  /// incumbent-relative limit only tightens as the incumbent improves, so a node pruned on it
  /// stays prunable.
  [[nodiscard]] double pool_cutoff() const {
    double cutoff = pool_.cutoff();
    if (have_incumbent_ && pool_gap_ < kNoPoolGap) {
      cutoff = std::min(cutoff, incumbent_internal_ +
                                    pool_gap_ * std::max(1.0, std::fabs(incumbent_internal_)));
    }
    return cutoff;
  }

  /// Objective at `x` in minimise space, excluding the offset.
  ///
  /// THE NODE BOUND AND THE INCUMBENT MUST BE THE SAME QUANTITY. Both used to be computed
  /// from col_cost alone, which is the whole objective for a MILP and only part of it for an
  /// MIQP - so with a Hessian present the search compared a linear bound against a quadratic
  /// incumbent and pruned on the difference. Measured on min x^2 - 3x, x integer in [0, 10]:
  /// the root bound came out -6 (the linear term at x = 2) against a true relaxation value of
  /// -2.25, an "optimistic" bound that is not a bound at all.
  ///
  /// The quadratic term is delegated to Model::evaluate_objective rather than rewritten here,
  /// because the lower-triangular storage convention it implements - stored off-diagonals
  /// standing for two entries of the symmetric matrix, the diagonal for one - is exactly the
  /// kind of detail that drifts when it exists in two places.
  [[nodiscard]] double internal_objective(const std::vector<double>& x) const {
    // The LP path keeps its own exact loop. Routing it through evaluate_objective would add
    // the offset and subtract it again, which is not an identity in floating point, and this
    // value decides pruning across the whole MIPLIB set.
    if (!quadratic_) {
      double value = 0.0;
      for (Index j = 0; j < original_.num_cols(); ++j) {
        const auto u = static_cast<std::size_t>(j);
        value += sense_ * original_.col_cost[u] * x[u];
      }
      return value;
    }
    return sense_ * (original_.evaluate_objective(x.data()) - original_.objective_offset);
  }

  [[nodiscard]] double reported(double internal) const {
    return sense_ * internal + original_.objective_offset;
  }

  // ---- Objective integrality (#221) -------------------------------------------------------

  /// The step every node optimum's internal objective is a multiple of, or 0 when nothing
  /// is known. Set once by detect_objective_integrality(): from the costs directly when every
  /// column with a cost is integer with an integer cost, or through the one row that defines
  /// a single continuous objective column from integer columns. A relaxation bound may then
  /// be rounded up to the next multiple: the node's integer optimum cannot lie in between.
  double objective_step_ = 0.0;
  void detect_objective_integrality();

  /// A relaxation bound rounded up to the next value an integer solution can take, in
  /// minimise space excluding the offset. Wolsey (1998) sec. 7.3 calls the reason a node
  /// with a bound of 14.3 and an incumbent of 15 is finished "bounding with integrality":
  /// the true optimum of the node is an integer and no integer lies in (14.3, 15). The slack
  /// keeps a bound that is a multiple of the step to rounding error from being pushed a
  /// whole step up: an LP bound of 14 + 1e-9 stays 14.
  [[nodiscard]] double integral_bound(double bound) const {
    return round_up_to_step(bound, objective_step_);
  }

  // ---- Safe bounds (#519), in branch_and_bound_safe.cpp ----------------------------------
  /// The Neumaier-Shcherbina bound on the entered node's LP from `relaxation`'s duals, in
  /// minimise space without the offset, with its gap to `believed` recorded; -inf when none.
  [[nodiscard]] double safe_node_bound(const Solution& relaxation, double believed);
  void report_safe_bounds() const;
  /// #502: what each option did, in the log and the profiler's counters, on every exit
  /// path (an infeasible search returns before the other counters are written).
  void report_branching_fixpoint() const;
  bool safe_bounds_ = false;
  // ---- Certificates (#518), in branch_and_bound_certificate.cpp -------------------------
  /// Keep what a solved node's LP proves: its duals (kDual) or Farkas multipliers (kFarkas).
  void certificate_record(Index node, const Solution& relaxation, CertificateTree::Proof proof);
  void certificate_children(Index node, Index down, Index up);
  /// Give up on the certificate, keeping the first reason.
  void certificate_refuse(const std::string& why);
  /// Write it, or say why not. At the end of run().
  void finish_certificate();
  /// True when y or -y proves the entered node's LP infeasible, by the safe test (#519).
  [[nodiscard]] bool farkas_proves(const std::vector<double>& y) const;
  /// The rows a certificate states the node LPs had: the model's, then every cut row in
  /// order, each `<=` its right-hand side (certificate_writer.hpp's with_cut_rows).
  [[nodiscard]] Model certificate_rows() const;
  [[nodiscard]] std::vector<CertificateTree::CutRow> certificate_cut_rows() const;
  /// True when working_ is the model's rows plus certified cut rows; refuses otherwise.
  bool certificate_rows_match();
  /// Certificate mode: keep only the cuts of a round whose derivation certify_cuts() can
  /// prove (cut_derivation.hpp), and say what was dropped. No-op otherwise.
  void certify_round_cuts(std::vector<Cut>* accepted);
  /// Certificate mode: the cut families that state no derivation are not separated; the
  /// first root round logs which ones were on.
  [[nodiscard]] bool certificate_mode() const { return !certificate_path_.empty(); }
  Count certificate_cuts_derived_ = 0;
  Count certificate_cuts_dropped_ = 0;
  Count certificate_farkas_resolves_ = 0;
  std::string certificate_path_;  ///< write_certificate; empty when off
  std::string certificate_refusal_;
  std::vector<CertificateTree::Node> certificate_nodes_;  ///< parallel to nodes_, grown lazily
  Count safe_bound_nodes_ = 0;                            ///< node bounds computed
  Count safe_bound_infinite_ = 0;  ///< of which -inf (no finite bound from those duals)
  Count safe_bound_refusals_ = 0;  ///< the believed bound prunes and the safe one does not
  /// max over nodes of believed - safe (finite ones; negative when safe was always higher)
  double safe_bound_max_gap_ = -std::numeric_limits<double>::infinity();
  double safe_bound_max_rel_gap_ = -std::numeric_limits<double>::infinity();

  // ---- Conflict analysis (branch_and_bound_conflicts.cpp, #292) -------------------------

  void init_conflicts();
  /// Every stored conflict against the current box: false when one holds entirely, else
  /// the bound each conflict with one undecided literal implies. Called from propagate().
  bool propagate_conflicts(bool* changed);
  /// Learn from a node just proved infeasible; the caller has already left it, so the
  /// working bounds are the global ones. `farkas` is the node LP's certificate, if any.
  void analyze_conflict(Index node_index, ConflictSource source,
                        const std::vector<double>* farkas);
  void report_conflicts();

  // ---- Cut rounds (branch_and_bound_cuts.cpp, #221) ------------------------------------

  /// The root round: cover, Gomory and MIR candidates, filtered, appended, the root
  /// re-solved; rolled back if the re-solve fails. Replaces `relaxation` on success.
  void root_cut_round(Solution* relaxation);
  /// The candidates every root round separates at `relaxation`: covers on the first
  /// `model_rows` rows, Gomory, MIR and the combinatorial families, in that order.
  [[nodiscard]] std::vector<Cut> separate_root_candidates(const Solution& relaxation,
                                                          Index model_rows);
  /// Rounds 2 onward of the root loop (#495, branch_and_bound_root_loop.cpp): separate at
  /// the current LP point, append, re-solve warm, until the bound stalls, the round cap, the
  /// time share or a round that takes nothing. `first_taken` and `first_iterations` are
  /// round 1's, for its log line.
  void root_cut_loop(Solution* relaxation, Index model_rows,
                     const std::vector<Cut>& first_taken, std::int64_t first_iterations);
  /// A round at a node of depth <= tree_cut_depth_: MIR cuts on the global bounds,
  /// appended for the whole tree, the node re-solved from its own basis.
  void tree_cut_round(Index depth, Solution* relaxation);
  /// Append `accepted` as rows of working_ and register them in the pool.
  void append_cut_rows(const std::vector<Cut>& accepted);
  /// Bring every stored basis to `rows` row statuses: a new row's logical is basic.
  void resize_warm_starts(Index rows);
  [[nodiscard]] bool is_pooled_duplicate(const Cut& cut) const;
  /// Clique and {0,1/2} candidates (#358) at `relaxation`, appended to `candidates`, each
  /// family behind its own option.
  /// `root`: the root round, the only place probing may run (#623 review).
  void add_combinatorial_cuts(const Solution& relaxation, std::vector<Cut>* candidates,
                              bool root);
  /// Count node solves in which each cut row was slack; free a row slack for too long.
  void age_cut_rows(const Solution& relaxation);

  /// How MIR separates (#498): c-MIR when `mir_cmir` is set.
  [[nodiscard]] MirOptions mir_options() const {
    MirOptions mir;
    // c-MIR states no derivation (#518), so a certificate's search uses the plain MIR.
    mir.cmir = options_.get_bool("mir_cmir") && certificate_path_.empty();
    mir.derive = !certificate_path_.empty();
    return mir;
  }
  /// The filter's policy from the options (#496): both default to the filter's own
  /// behaviour until the A/B on main says otherwise.
  [[nodiscard]] CutFilterPolicy cut_filter_policy() const {
    CutFilterPolicy policy;
    policy.support_floor = static_cast<Index>(options_.get_int("cut_support_floor"));
    policy.efficacy = options_.get_bool("cut_efficacy_test");
    return policy;
  }

  // ---- The debug-solution check (#500), in branch_and_bound_debug.cpp ------------------
  void debug_start();                ///< load the point; check the rows appended at the root
  bool debug_node_contains() const;  ///< the entered node's domain contains the point
  /// Every cut against the point; `first_row` is the row the first becomes, or -1 for the
  /// candidates of a round before its filter.
  void debug_check_cuts(const std::vector<Cut>& cuts, Index first_row);
  void debug_after_propagation(bool feasible);   ///< for a node that contained the point
  void debug_after_node_lp(const Solution& lp);  ///< for a node that contains the point
  void debug_after_global_tightening();          ///< reduced-cost fixing
  std::optional<DebugSolution> debug_;
  double debug_objective_ = 0.0;
  Index debug_node_ = 0;
  std::string debug_round_;

  /// The root bounds and the cut counts onto the answer (#221).
  void report_root(Solution* solution) const {
    solution->cuts_applied = root_cuts_applied_ + tree_cuts_applied_;
    solution->cut_filter_report = cut_filter_report_;
    solution->incumbent_trace = incumbent_trace_;  // #504, not a root quantity but same exits
    if (std::isnan(root_bound_internal_)) return;
    solution->root_bound = reported(root_bound_internal_);
    solution->root_bound_after_cuts = reported(root_bound_after_cuts_internal_);
  }

  const Model& original_;
  Model working_;
  const Options& options_;
  Logger& logger_;
  SolveControl* control_;
  Options node_options_;

  double integrality_tolerance_ = tol::kIntegrality;
  double relative_gap_target_ = tol::kMipRelativeGap;
  double absolute_gap_target_ = tol::kMipAbsoluteGap;
  ResourceLimits limits_;
  double time_limit_ = 0.0;
  double sense_ = 1.0;

  std::vector<Index> integer_columns_;
  /// EQUILIBRATION, COMPUTED ONCE (#76). The tree does not copy the model - one working
  /// Model is built up front and nodes differ ONLY in variable bounds - so the constraint
  /// matrix, and therefore the row and column multipliers, are identical at every node.
  /// Rebuilding them per node was ten Ruiz passes plus a Pock-Chambolle pass over a full copy
  /// of the matrix, discarded and repeated at the next node. Measured on the case studies
  /// that was 5-10x of the whole solve; on a MILP with thousands of nodes it would dominate.
  ///
  /// The bounds are still scaled per node by solve_primal_simplex, because those are exactly
  /// what branching changes. Only the reusable part is cached.
  bool quadratic_ = false;  ///< the node relaxation is a QP, not an LP

  NodeScaling scaling_;

  /// mip_node_engine: warm-started dual (default) or cold primal for every node.
  bool node_engine_dual_ = true;
  /// mip_node_selection (#293). Order only: every policy explores the same tree.
  NodeSelection node_selection_ = NodeSelection::kHybrid;
  /// How many nodes each rule chose, for the report at the end of the search.
  Count selected_by_dive_ = 0;
  Count selected_by_policy_ = 0;
  Index deepest_node_ = 0;
  /// mip_branching: reliability (default) or the most-fractional rule it replaced.
  bool reliability_branching_ = true;
  /// Pseudocosts (#69): per integer column, the sum and count of observed bound gains per
  /// unit of fractionality, in each branching direction.
  std::vector<double> pseudo_down_sum_;
  std::vector<double> pseudo_up_sum_;
  std::vector<Count> pseudo_down_count_;
  std::vector<Count> pseudo_up_count_;
  Options probe_options_;  ///< node_options_ with the strong-branching iteration cap
  Count strong_branch_solves_ = 0;
  Count strong_branch_iterations_ = 0;
  /// #502, all three off by default until an A/B on main says otherwise.
  bool strong_branch_fix_ = false;        ///< mip_strong_branch_fix
  bool incremental_propagation_ = false;  ///< mip_incremental_propagation
  bool heap_open_list_ = false;           ///< mip_heap_open_list (single worker only)
  /// The current node's strong-branch fixes (#502), for link_strong_fixes().
  std::vector<DomainChange> strong_fixes_;
  Count strong_branch_fixes_ = 0;       ///< columns fixed that way over the search
  Count strong_branch_fix_prunes_ = 0;  ///< nodes a re-solve after fixing fathomed
  Count propagated_rows_ = 0;           ///< rows the fixpoint worklist processed
  Count heap_selections_ = 0;           ///< nodes taken from the heap open list
  /// The basis to start the NEXT node LP from; empty means the slack basis (the root).
  WarmStart current_warm_;
  /// mip_node_factor_cache (#501): first factorizations kept for the next node LP that
  /// starts from the same basis. Null when the option is 0.
  std::unique_ptr<NodeFactorCache> factor_cache_;
  Count warm_node_solves_ = 0;
  Count cold_node_solves_ = 0;
  Count cold_fallbacks_ = 0;
  Count warm_node_iterations_ = 0;
  Count cold_node_iterations_ = 0;

  std::vector<TreeNode> nodes_;
  std::vector<Index> open_;

  /// Bounds saved by the current enter(), restored by leave().
  std::vector<DomainChange> saved_;

  // Primal heuristics (#290, #414): which run and with what budgets, resolved once.
  std::vector<HeuristicStats> heuristic_stats_;
  /// The incumbent (internal objective) Local-MIP last started from (#507); NaN before any.
  double local_mip_from_ = std::numeric_limits<double>::quiet_NaN();
  Locks locks_;
  HeuristicSchedule schedule_;
  // Reduced-cost fixing and restarts (#418).
  bool reduced_cost_fixing_ = false;
  std::vector<double> root_reduced_;  ///< the root relaxation's reduced costs, minimise space
  std::vector<BasisStatus> root_status_;
  /// The bound each nonbasic column sat at in the root LP, which is the root's PROPAGATED
  /// bound, not global_lower_/global_upper_: the reduced cost prices moves from there.
  std::vector<double> root_at_bound_;
  double fixing_incumbent_ = std::numeric_limits<double>::infinity();  ///< last pass used
  Count reduced_cost_fixings_ = 0;  ///< bounds moved over the search, reported
  Count symmetry_generators_ = 0;   ///< #413: verified generators the detection found
  Count symmetry_rows_ = 0;         ///< #413: ordering rows appended for them
  Count fixed_since_root_ = 0;      ///< integer columns fixed since the root was processed
  Count restarts_ = 0;
  Count restarts_allowed_ = 0;
  double restart_fraction_ = 0.0;
  Count restart_node_limit_ = 0;
  /// Objective branching (#418): the objective's row of working_ once appended, or -1, and
  /// how many nodes branched on it.
  Index objective_row_ = -1;
  Count objective_branches_ = 0;
  Count clique_cuts_generated_ = 0;     ///< #358, before the filter
  Count zero_half_cuts_generated_ = 0;  ///< #358, before the filter
  /// #512: literal conflicts from probing the root model once, fed to the clique separator
  /// alongside the row-derived graph; probed_ says the probe has run (it may find none).
  std::vector<std::pair<Index, Index>> probed_conflicts_;
  bool probed_ = false;
  std::string checkpoint_path_;
  Count checkpoint_nodes_ = 0;
  Count last_checkpoint_at_ = -1;
  Count checkpoints_written_ = 0;

  bool have_incumbent_ = false;
  double incumbent_internal_ = std::numeric_limits<double>::infinity();
  std::vector<double> incumbent_x_;
  /// Each accepted improvement, on this search's clock (#504); the sequential search only.
  std::vector<Solution::IncumbentEvent> incumbent_trace_;

  /// Every integer-feasible point offer_incumbent() found feasible, not only the improving
  /// ones (#225).
  SolutionPool pool_;
  double pool_gap_ = std::numeric_limits<double>::max();
  bool pool_complete_ = false;
  /// pool_gap at or above this is the option's keep-everything default.
  static constexpr double kNoPoolGap = 1e300;

  Count nodes_explored_ = 0;
  /// Root relaxation objective (internal, minimise space) before and after the root cut
  /// round; NaN until the root LP solved. Reported through Solution::root_bound (#221).
  double root_bound_internal_ = std::numeric_limits<double>::quiet_NaN();
  double root_bound_after_cuts_internal_ = std::numeric_limits<double>::quiet_NaN();
  Count root_cuts_applied_ = 0;
  Count root_cut_rounds_ = 0;      ///< #495: rounds that appended rows at the root
  std::string cut_filter_report_;  ///< the root filter's verdicts per family and reason (#496)
  /// Cut rows below the root (#221): the option-driven depth cap and per-round row cap,
  /// the column bounds every tree cut is built on (valid everywhere), the pool of rows
  /// appended so far starting at working_ row first_cut_row_, and their ageing state.
  Index tree_cut_depth_ = 0;
  Index tree_cut_rows_per_round_ = 20;
  /// Cut selection (#415): the root round's cap, the parallelism above which a cut waits,
  /// and the cuts that passed a round's filter but were not taken, offered again at the
  /// next round where the LP point has moved.
  Index cut_max_per_round_ = 30;
  double cut_max_parallelism_ = tol::kCutMaxParallelism;
  std::vector<Cut> waiting_cuts_;
  std::vector<double> global_lower_;
  std::vector<double> global_upper_;
  Index first_cut_row_ = -1;
  std::vector<Cut> pool_cuts_;
  std::vector<Count> cut_row_slack_;
  std::vector<bool> cut_row_free_;
  Count tree_cuts_applied_ = 0;
  Count tree_cut_rounds_ = 0;
  Count cut_rows_aged_out_ = 0;
  /// Node solves a cut row may sit slack before it is freed (Achterberg 2007, sec. 8.10
  /// uses a comparable age).
  static constexpr Count kCutRowAgeLimit = 50;
  Count nodes_pruned_ = 0;

  /// Conflict analysis (#292): the learned conflicts, their statistics, and whether the
  /// search is inside an analysis (whose trial propagations must not count as uses).
  bool conflicts_enabled_ = false;
  bool conflict_minimize_ = true;
  /// conflict_use: what the search does with a learned conflict (#292's ablation).
  enum class ConflictUse { kNone, kPrune, kPropagate };
  ConflictUse conflict_use_ = ConflictUse::kPropagate;
  std::size_t conflict_max_size_ = 0;
  ConflictStore conflicts_;
  ConflictStats conflict_stats_;
  bool analysing_ = false;
  bool conflict_pruned_ = false;  ///< the last propagate() failed on a stored conflict
  /// Verification calls one analysis may spend minimising, and the total the analyses may
  /// spend per node explored, plus a start-up allowance.
  static constexpr int kConflictMinimizeChecks = 32;
  static constexpr Count kConflictChecksPerNode = 8;
  static constexpr Count kConflictChecksBase = 2000;
  /// The parallel search this worker belongs to, or null (#222).
  SharedSearch* shared_ = nullptr;
  const SubtreeSpec* seed_ = nullptr;
  Count nodes_reported_ = 0;
  /// Open nodes a worker must hold before it gives any away (#222).
  static constexpr std::size_t kMinOpenToDonate = 8;
  /// Nodes between pseudocost exchanges with the other workers (#222).
  static constexpr Count kPseudocostSyncNodes = 20;
  Count next_pseudocost_sync_ = 0;
  std::vector<double> pseudo_start_down_sum_;
  std::vector<double> pseudo_start_up_sum_;
  std::vector<Count> pseudo_start_down_count_;
  std::vector<Count> pseudo_start_up_count_;
  Timer timer_;
};

}  // namespace sankhya::mip
