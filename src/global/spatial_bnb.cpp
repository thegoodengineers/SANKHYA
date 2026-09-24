// SPDX-License-Identifier: Apache-2.0
// SANKHYA - spatial branch and bound over McCormick relaxations (#514).
//
// THE METHOD. Falk and Soland ("An algorithm for separable nonconvex programming problems",
// Management Science 15, 1969) for the branch and bound over boxes; McCormick (1976) for the
// relaxation of each product over a box; Tawarmalani and Sahinidis, "Convexification and
// Global Optimization in Continuous and Mixed-Integer Nonlinear Programming" (Kluwer 2002)
// and Belotti, Lee, Liberti, Margot and Wachter (Optim. Methods Softw. 24, 2009) for the
// branching and bound-tightening choices. Each node is a box:
//
//   1. FBBT tightens the box against every row and against objective <= incumbent; a box
//      it empties is pruned.
//   2. The McCormick relaxation over the box is an LP; its multipliers give a lower bound
//      for the box by weak duality (relaxation.cpp), so an LP solved only to tolerance
//      weakens the bound and never makes it wrong. An infeasible relaxation prunes the box.
//   3. Upper bounds come only from points checked against the ORIGINAL model, products and
//      all: the relaxation's own point when it happens to satisfy the products, and the
//      alternating LP started from it (local_search.cpp).
//   4. Branch on a column of the product whose relaxation value is furthest from the product
//      of its factors, at a point mixing the relaxation's value with the box midpoint.
//
// Best-bound node order. The proven bound is the least bound over every box still open or
// closed by bound; `optimal` is reported only when it is within mip_absolute_gap or
// mip_relative_gap of the incumbent - the same targets, and the same test, a MILP answer is
// held to - and otherwise the answer is `feasible` with the bound it did prove.
//
// SCOPE OF THIS FIRST SLICE: continuous columns, products in the rows and in the objective,
// every column of a product bounded after FBBT at the root. OBBT (#515), cuts beyond the
// McCormick envelope, integer columns and a written bound certificate are not here.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "global/global_internal.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/qcqp.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace global {
namespace {

struct Node {
  Box box;
  double bound = -kInfinity;  ///< valid lower bound for every feasible point of the box
  int depth = 0;
  Count id = 0;
  /// The parent's optimal basis of the relaxation, to start this node's LP from. Only the
  /// box moved, so it is usually a few dual pivots from this node's optimum.
  std::vector<BasisStatus> col_status;
  std::vector<BasisStatus> row_status;
};

/// Least bound first; among equals the older node (a deterministic order).
bool later(const std::unique_ptr<Node>& a, const std::unique_ptr<Node>& b) {
  if (a->bound != b->bound) return a->bound > b->bound;
  return a->id > b->id;
}

struct BranchChoice {
  Index column = -1;
  double point = 0.0;
};

class Search {
 public:
  Search(const QcqpModel& model, const Options& options, SolveControl* control)
      : model_(model),
        options_(options),
        control_(control),
        logger_(options.get_bool("log_to_console") ? stdout : nullptr),
        problem_(build_problem(model)) {
    time_limit_ = options.get_double("time_limit");
    node_limit_ = options.get_int("node_limit");
    relative_gap_ = options.get_double("mip_relative_gap");
    absolute_gap_ = options.get_double("mip_absolute_gap");
    tolerance_ = options.get_double("primal_feasibility_tolerance");
  }

  Solution run();

 private:
  [[nodiscard]] double seconds_left() const {
    return time_limit_ < std::numeric_limits<double>::max()
               ? time_limit_ - timer_.elapsed_seconds()
               : kInfinity;
  }
  /// A box whose bound is at or above this cannot hold a point better than the incumbent by
  /// more than the gap target.
  [[nodiscard]] double prune_level() const {
    if (!std::isfinite(upper_)) return kInfinity;
    return upper_ - std::max(absolute_gap_, relative_gap_ * std::max(1.0, std::fabs(upper_)));
  }
  void offer(const std::vector<double>& x, const char* source) {
    if (original_violation(problem_, x) > tolerance_) return;
    const double objective = min_objective(problem_, x);
    if (objective < upper_) {
      upper_ = objective;
      incumbent_ = x;
      incumbent_source_ = source;
      ++incumbents_;
      logger_.info("global: incumbent {:.10g} from {} at node {}", problem_.sigma * upper_,
                   source, nodes_);
    }
  }
  void push(std::unique_ptr<Node> node) {
    node->id = next_id_++;
    open_.push_back(std::move(node));
    std::push_heap(open_.begin(), open_.end(), later);
  }
  std::unique_ptr<Node> pop() {
    std::pop_heap(open_.begin(), open_.end(), later);
    std::unique_ptr<Node> node = std::move(open_.back());
    open_.pop_back();
    return node;
  }
  [[nodiscard]] BranchChoice choose_branch(const Box& box, const std::vector<double>& lp) const;
  [[nodiscard]] bool branchable(const Box& box, Index column) const;
  void process(std::unique_ptr<Node> node);
  Solution finish(LimitReason stopped_by);

  const QcqpModel& model_;
  const Options& options_;
  SolveControl* control_;
  Logger logger_;
  Timer timer_;
  Problem problem_;
  Box root_;

  double time_limit_ = kInfinity;
  std::int64_t node_limit_ = -1;
  double relative_gap_ = 0.0;
  double absolute_gap_ = 0.0;
  double tolerance_ = 0.0;

  std::vector<std::unique_ptr<Node>> open_;
  Count next_id_ = 0;
  double upper_ = kInfinity;
  std::vector<double> incumbent_;
  const char* incumbent_source_ = "";
  Count incumbents_ = 0;
  double closed_bound_ = kInfinity;  ///< least bound over boxes closed with a bound
  double root_bound_ = std::numeric_limits<double>::quiet_NaN();
  Count nodes_ = 0;
  Count lp_iterations_ = 0;
  Count lp_failures_ = 0;
  Count unsafe_bounds_ = 0;
  Count unresolved_ = 0;
  Count tightened_ = 0;
  Count heuristic_calls_ = 0;
  bool out_of_time_ = false;
};

bool Search::branchable(const Box& box, Index column) const {
  const auto u = static_cast<std::size_t>(column);
  const double width = box.upper[u] - box.lower[u];
  return width > tol::kGlobalMinBranchWidth * std::max(1.0, std::fabs(box.upper[u]));
}

BranchChoice Search::choose_branch(const Box& box, const std::vector<double>& lp) const {
  const Index n = problem_.n;
  const auto relative_width = [&](Index j) {
    const auto u = static_cast<std::size_t>(j);
    const double root = root_.upper[u] - root_.lower[u];
    return root > 0.0 ? (box.upper[u] - box.lower[u]) / root : 0.0;
  };
  // The product whose relaxation value is furthest from the product of its factors.
  Index best = -1;
  double worst = -1.0;
  if (!lp.empty()) {
    for (std::size_t p = 0; p < problem_.products.size(); ++p) {
      const Product& product = problem_.products[p];
      const double xa = lp[static_cast<std::size_t>(product.a)];
      const double xb = lp[static_cast<std::size_t>(product.b)];
      const double w = lp[static_cast<std::size_t>(n) + p];
      const double violation = std::fabs(w - xa * xb);
      if (violation <= tol::kGlobalProductTolerance * std::max(1.0, std::fabs(xa * xb)))
        continue;
      if (!branchable(box, product.a) && !branchable(box, product.b)) continue;
      if (violation > worst) {
        worst = violation;
        best = static_cast<Index>(p);
      }
    }
  }
  Index column = -1;
  if (best >= 0) {
    const Product& product = problem_.products[static_cast<std::size_t>(best)];
    const bool a_ok = branchable(box, product.a);
    const bool b_ok = branchable(box, product.b);
    column = (a_ok && (!b_ok || relative_width(product.a) >= relative_width(product.b)))
                 ? product.a
                 : product.b;
  } else {
    // No product is violated (or there is no relaxation point): refine the widest column
    // of a product, at its midpoint.
    double widest = 0.0;
    for (Index j = 0; j < n; ++j) {
      if (problem_.in_product[static_cast<std::size_t>(j)] == 0 || !branchable(box, j))
        continue;
      if (relative_width(j) > widest) {
        widest = relative_width(j);
        column = j;
      }
    }
  }
  if (column < 0) return {};
  const auto u = static_cast<std::size_t>(column);
  const double lower = box.lower[u];
  const double upper = box.upper[u];
  const double middle = 0.5 * (lower + upper);
  double point = middle;
  if (best >= 0) {
    point = tol::kGlobalBranchPointLpWeight * std::clamp(lp[u], lower, upper) +
            (1.0 - tol::kGlobalBranchPointLpWeight) * middle;
  }
  const double margin = tol::kGlobalBranchPointMargin * (upper - lower);
  point = std::clamp(point, lower + margin, upper - margin);
  return {column, point};
}

void Search::process(std::unique_ptr<Node> node) {
  ++nodes_;
  const FbbtOutcome tightening = fbbt(problem_, upper_, tolerance_, &node->box);
  tightened_ += tightening.tightened;
  if (tightening.infeasible) return;  // an empty box: closed, with bound +infinity

  const Model lp = build_relaxation(problem_, node->box);
  const Solution relaxed = solve_lp(lp, inner_lp_options(options_, seconds_left()),
                                    &node->col_status, &node->row_status);
  lp_iterations_ += relaxed.iterations;

  double bound = node->bound;
  std::vector<double> point;
  if (relaxed.status == SolveStatus::kInfeasible) return;
  if (relaxed.status == SolveStatus::kOptimal &&
      static_cast<Index>(relaxed.col_value.size()) == lp.num_cols()) {
    // Only a bound the multipliers prove is used. When they prove none (a column with no
    // finite bound on the side its reduced cost needs), the box keeps its parent's proved
    // bound: the LP's own objective is only as good as the tolerance it was solved to, and
    // pruning or declaring optimality on it would make the "proved" gap a claim (review of
    // #637). The box is still branched, and its children get a fresh chance to prove more.
    const double proved = dual_bound_from_multipliers(lp, relaxed.row_dual);
    if (std::isfinite(proved)) {
      bound = std::max(bound, proved);
    } else {
      ++unsafe_bounds_;  // counted and reported; the parent's bound stands
    }
    point = relaxed.col_value;
  } else {
    if (relaxed.stopped_by == LimitReason::kTime || seconds_left() <= 0.0) {
      out_of_time_ = true;
      push(std::move(node));  // still open: its bound is still owed
      return;
    }
    ++lp_failures_;  // the parent's bound still holds for this box; branch without a point
    logger_.verbose("global: node {} relaxation ended {} ({}); parent bound kept", nodes_,
                    to_string(relaxed.status), relaxed.message);
  }
  if (nodes_ == 1) root_bound_ = bound;
  if (bound >= prune_level()) {
    closed_bound_ = std::min(closed_bound_, bound);
    return;
  }

  if (!point.empty()) {
    const std::vector<double> x(point.begin(), point.begin() + problem_.n);
    offer(x, "the relaxation");
    const bool due = nodes_ <= tol::kGlobalHeuristicAlwaysNodes ||
                     nodes_ % tol::kGlobalHeuristicInterval == 0;
    if (due && bound < prune_level()) {
      ++heuristic_calls_;
      const LocalSearchResult found = relaxation_search(
          problem_, point, upper_, inner_lp_options(options_, seconds_left()), tolerance_);
      lp_iterations_ += found.lp_iterations;
      if (found.improved) offer(found.x, "the alternating LP");
    }
    if (bound >= prune_level()) {
      closed_bound_ = std::min(closed_bound_, bound);
      return;
    }
  }

  const BranchChoice choice = choose_branch(node->box, point);
  if (choice.column < 0) {
    // Every product column is as narrow as it can usefully be cut. The box keeps its bound,
    // which therefore stays in the proven bound; it is not called infeasible.
    ++unresolved_;
    closed_bound_ = std::min(closed_bound_, bound);
    return;
  }
  const auto u = static_cast<std::size_t>(choice.column);
  auto left = std::make_unique<Node>();
  left->box = node->box;
  left->box.upper[u] = choice.point;
  left->bound = bound;
  left->depth = node->depth + 1;
  left->col_status = relaxed.col_status;
  left->row_status = relaxed.row_status;
  auto right = std::make_unique<Node>();
  right->box = std::move(node->box);
  right->box.lower[u] = choice.point;
  right->bound = bound;
  right->depth = node->depth + 1;
  right->col_status = relaxed.col_status;
  right->row_status = relaxed.row_status;
  push(std::move(left));
  push(std::move(right));
}

Solution Search::run() {
  Solution solution;
  solution.algorithm = "spatial-branch-and-bound";
  root_.lower = problem_.col_lower;
  root_.upper = problem_.col_upper;
  const FbbtOutcome root_tightening = fbbt(problem_, kInfinity, tolerance_, &root_);
  tightened_ += root_tightening.tightened;
  if (root_tightening.infeasible) {
    solution.status = SolveStatus::kInfeasible;
    solution.message =
        "bound tightening at the root proves no point of the column box satisfies the rows";
    solution.solve_seconds = timer_.elapsed_seconds();
    return solution;
  }
  for (Index j = 0; j < problem_.n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (problem_.in_product[u] != 0 &&
        !(std::isfinite(root_.lower[u]) && std::isfinite(root_.upper[u]))) {
      const Model& lin = model_.linear;
      solution.status = SolveStatus::kNotSolved;
      solution.message = fmt::format(
          "column '{}' appears in a product and has no finite {} bound, even after bound "
          "tightening; the McCormick relaxation needs a bounded box",
          lin.col_names.empty() ? fmt::format("x{}", j) : lin.col_names[u],
          std::isfinite(root_.lower[u]) ? "upper" : "lower");
      solution.solve_seconds = timer_.elapsed_seconds();
      return solution;
    }
  }
  logger_.info(
      "global: {} columns, {} rows, {} products; spatial branch and bound over "
      "McCormick relaxations, gap target {:g} relative / {:g} absolute",
      problem_.n, problem_.m, problem_.products.size(), relative_gap_, absolute_gap_);

  auto root = std::make_unique<Node>();
  root->box = root_;
  push(std::move(root));
  LimitReason stopped_by = LimitReason::kNone;
  while (!open_.empty()) {
    if (control_ != nullptr && control_->interruption_requested()) {
      stopped_by = LimitReason::kInterrupt;
      break;
    }
    if (out_of_time_ || seconds_left() <= 0.0) {
      stopped_by = LimitReason::kTime;
      break;
    }
    if (node_limit_ >= 0 && nodes_ >= node_limit_) {
      stopped_by = LimitReason::kNodes;
      break;
    }
    std::unique_ptr<Node> node = pop();
    if (node->bound >= prune_level()) {
      closed_bound_ = std::min(closed_bound_, node->bound);
      continue;
    }
    process(std::move(node));
    if (nodes_ % 200 == 0) {
      const double open_bound = open_.empty() ? closed_bound_ : open_.front()->bound;
      logger_.info("global: {} nodes, {} open, bound {:.10g}, incumbent {:.10g}, {:.1f}s",
                   nodes_, open_.size(), problem_.sigma * std::min(open_bound, closed_bound_),
                   problem_.sigma * upper_, timer_.elapsed_seconds());
    }
  }
  if (out_of_time_ && stopped_by == LimitReason::kNone) stopped_by = LimitReason::kTime;
  return finish(stopped_by);
}

Solution Search::finish(LimitReason stopped_by) {
  Solution solution;
  solution.algorithm = "spatial-branch-and-bound";
  double lower = closed_bound_;
  for (const std::unique_ptr<Node>& node : open_) lower = std::min(lower, node->bound);
  const bool have_point = !incumbent_.empty();
  if (have_point) lower = std::min(lower, upper_);
  const double sigma = problem_.sigma;

  solution.nodes = nodes_;
  solution.iterations = lp_iterations_;
  solution.root_bound = sigma * root_bound_;
  solution.stopped_by = stopped_by;
  solution.solve_seconds = timer_.elapsed_seconds();
  const std::string effort = fmt::format(
      "{} nodes, {} FBBT tightenings, {} alternating-LP calls{}{}{}", nodes_, tightened_,
      heuristic_calls_,
      lp_failures_ > 0 ? fmt::format(", {} node LPs failed (parent bound kept)", lp_failures_)
                       : "",
      unsafe_bounds_ > 0
          ? fmt::format(", {} node LPs whose multipliers proved no bound (parent bound kept)",
                        unsafe_bounds_)
          : "",
      unresolved_ > 0
          ? fmt::format(", {} boxes too narrow to branch closed at their bound", unresolved_)
          : "");

  if (have_point) {
    solution.col_value = incumbent_;
    solution.row_activity = model_.row_activity(incumbent_);
    solution.objective = model_.objective(incumbent_);
    solution.dual_bound = sigma * lower;
    solution.primal_infeasibility_scaled =
        original_violation(problem_, incumbent_, &solution.primal_infeasibility);
    solution.absolute_gap = std::fabs(solution.objective - solution.dual_bound);
    solution.relative_gap =
        solution.absolute_gap / std::max(1.0, std::fabs(solution.objective));
    const bool closed =
        solution.absolute_gap <= absolute_gap_ || solution.relative_gap <= relative_gap_;
    solution.status = closed ? SolveStatus::kOptimal : SolveStatus::kFeasible;
    solution.message =
        closed ? fmt::format(
                     "global optimum to gap {:.3e} (absolute {:.3e}); incumbent from {}; "
                     "{}",
                     solution.relative_gap, solution.absolute_gap, incumbent_source_, effort)
               : fmt::format("global gap {:.3e} not closed{}; the bound {:.10g} is proved; {}",
                             solution.relative_gap,
                             stopped_by != LimitReason::kNone
                                 ? fmt::format(" (stopped by {})", to_string(stopped_by))
                                 : "",
                             solution.dual_bound, effort);
    return solution;
  }
  solution.objective = sigma * kInfinity;
  solution.dual_bound = sigma * lower;
  solution.absolute_gap = kInfinity;
  solution.relative_gap = kInfinity;
  if (stopped_by != LimitReason::kNone) {
    solution.status = status_for(stopped_by);
    solution.message =
        fmt::format("no feasible point found before the {}; {}", to_string(stopped_by), effort);
  } else if (unresolved_ == 0) {
    solution.status = SolveStatus::kInfeasible;
    solution.message = fmt::format(
        "every box was proved empty by bound tightening or by an infeasible relaxation; {}",
        effort);
  } else {
    solution.status = SolveStatus::kNotSolved;
    solution.message = fmt::format(
        "no feasible point found, and boxes too narrow to branch remain, so infeasibility is "
        "not proved; {}",
        effort);
  }
  return solution;
}

}  // namespace
}  // namespace global

Solution solve_global(const QcqpModel& model, const Options& options, SolveControl* control) {
  const std::string problem = model.validate();
  if (!problem.empty()) {
    Solution solution;
    solution.status = SolveStatus::kModelError;
    solution.algorithm = "spatial-branch-and-bound";
    solution.message = problem;
    return solution;
  }
  if (model.linear.has_integrality()) {
    Solution solution;
    solution.status = SolveStatus::kNotSolved;
    solution.algorithm = "spatial-branch-and-bound";
    solution.message =
        "the global method handles continuous models only; this one has integer columns";
    return solution;
  }
  const global::Problem shape = global::build_problem(model);
  if (shape.products.empty()) {
    // Nothing non-convex to branch on: the model IS its linear part.
    return solve(model.linear, options, control);
  }
  global::Search search(model, options, control);
  return search.run();
}

}  // namespace sankhya
