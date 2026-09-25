// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP by NLP-based branch and bound (NLP stage 3). See minlp_bnb.hpp for
// the method (Gupta and Ravindran 1985), what makes it valid and what is claimed.

#include "nlp/minlp_bnb.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "nlp/nlp_solve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::nlp {
namespace {

struct TreeNode {
  std::vector<double> lower, upper;
  std::vector<double> start;
  double bound = -kInfinity;  ///< minimisation space: the parent's relaxation value
  Count depth = 0;
};

struct Worse {
  bool operator()(const TreeNode& a, const TreeNode& b) const {
    return a.bound != b.bound ? a.bound > b.bound : a.depth < b.depth;  // best bound, deepest
  }
};

bool is_point(SolveStatus s) {
  return s == SolveStatus::kOptimal || s == SolveStatus::kLocallyOptimal;
}

/// The node's model: `model` with the node's column bounds.
void with_bounds(const TreeNode& node, NonlinearModel* work) {
  work->base.col_lower = node.lower;
  work->base.col_upper = node.upper;
}

/// The convex minimum-violation problem over the node's box: the same rows, each given a
/// surplus p and a slack n (both >= 0), the objective sum(p + n), no integrality. Its optimum
/// is the least total violation of any point in the box; convex whenever `node` is, since
/// adding p - n changes no curvature and the objective is linear.
NonlinearModel violation_model(const NonlinearModel& node) {
  const Model& b = node.base;
  const Index n = b.num_cols();
  const Index linear = b.num_rows();
  const Index rows = linear + static_cast<Index>(node.constraints.size());
  Model base = b;
  base.objective_offset = 0.0;
  base.hessian = SparseMatrix();
  base.resize_columns(n + 2 * rows);
  std::fill(base.col_cost.begin(), base.col_cost.end(), 0.0);
  for (Index j = 0; j < n; ++j)
    base.col_type[static_cast<std::size_t>(j)] = VarType::kContinuous;
  for (Index k = n; k < n + 2 * rows; ++k) {
    const auto u = static_cast<std::size_t>(k);
    base.col_cost[u] = 1.0;
    base.col_lower[u] = 0.0;
    base.col_upper[u] = kInfinity;
    base.col_type[u] = VarType::kContinuous;
  }
  base.sense = ObjSense::kMinimize;
  SparseMatrix matrix(linear, n + 2 * rows);
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = b.matrix.column(j);
    for (Index k = 0; k < column.size; ++k)
      matrix.add_entry(column.rows[k], j, column.values[k]);
  }
  for (Index i = 0; i < linear; ++i) {
    matrix.add_entry(i, n + i, 1.0);
    matrix.add_entry(i, n + rows + i, -1.0);
  }
  matrix.finalize(0.0);
  base.matrix = std::move(matrix);
  NonlinearModel out(std::move(base));
  out.graph = node.graph;
  out.graph.extend_variables(n + 2 * rows);
  for (std::size_t k = 0; k < node.constraints.size(); ++k) {
    NonlinearConstraint c = node.constraints[k];
    const Index i = linear + static_cast<Index>(k);
    c.expression = out.graph.sum({c.expression, out.graph.variable(n + i),
                                  out.graph.negate(out.graph.variable(n + rows + i))});
    out.constraints.push_back(std::move(c));
  }
  return out;
}

}  // namespace

Solution solve_minlp(const NonlinearModel& model, const Options& options,
                     SolveControl* control) {
  const Timer timer;
  Logger logger(options.get_bool("log_to_console") ? stdout : nullptr);
  LogLevel level = LogLevel::kInfo;
  if (parse_log_level(options.get_string("log_level"), &level)) logger.set_level(level);
  Solution out;
  out.algorithm = "minlp-nlp-bnb";
  const std::string problem = model.validate();
  if (!problem.empty()) {
    out.status = SolveStatus::kModelError;
    out.message = problem;
    return out;
  }
  if (!model.base.has_integrality()) return solve_nlp(model, options, control);
  const ConvexityReport convexity = model.convexity();
  const bool assumed = !convexity.convex && options.get_bool("nlp_assume_convex");
  if (!convexity.convex && !assumed) {
    out.status = SolveStatus::kNotSolved;
    out.message = fmt::format(
        "the continuous relaxation is not proved convex ({}); NLP-based branch and bound "
        "gives no valid bound on a nonconvex MINLP, so the model is refused "
        "(nlp_assume_convex=true runs it as a heuristic, and then nothing is called optimal)",
        convexity.reasons.empty() ? "no proof" : convexity.reasons.front());
    return out;
  }

  const double sense = model.base.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  const double int_tol = options.get_double("integrality_tolerance");
  const double rel_target = options.get_double("mip_relative_gap");
  const double abs_target = options.get_double("mip_absolute_gap");
  const std::int64_t node_limit = options.get_int("node_limit");
  const double time_limit = options.get_double("time_limit");
  Options quiet = options;
  quiet.set_bool("log_to_console", false);
  Logger silent(nullptr);
  const Index n = model.base.num_cols();
  std::vector<Index> integers;
  for (Index j = 0; j < n; ++j) {
    if (model.base.col_type[static_cast<std::size_t>(j)] == VarType::kInteger)
      integers.push_back(j);
  }

  TreeNode root;
  root.lower = model.base.col_lower;
  root.upper = model.base.col_upper;
  for (const Index j : integers) {
    auto& lo = root.lower[static_cast<std::size_t>(j)];
    auto& up = root.upper[static_cast<std::size_t>(j)];
    if (std::isfinite(lo)) lo = std::ceil(lo - int_tol);
    if (std::isfinite(up)) up = std::floor(up + int_tol);
  }
  root.start = model.start;
  std::priority_queue<TreeNode, std::vector<TreeNode>, Worse> open;
  open.push(root);

  NonlinearModel work = model;
  double incumbent = kInfinity;  // minimisation space
  Solution best;
  bool proof_lost = false;
  std::string lost_because;
  Count nodes = 0;
  Count nlp_iterations = 0;
  LimitReason stopped = LimitReason::kNone;
  double open_bound = kInfinity;

  const auto lose = [&](const std::string& why) {
    if (!proof_lost) lost_because = why;
    proof_lost = true;
  };
  const auto certify_empty = [&](const TreeNode& node) {
    with_bounds(node, &work);
    const NonlinearModel violation = violation_model(work);
    const Solution v = solve_nlp_relaxation(violation, quiet, {}, control, silent);
    nlp_iterations += v.iterations;
    return v.status == SolveStatus::kOptimal && v.objective > tol::kMinlpEmptyViolation;
  };

  while (!open.empty()) {
    if (control != nullptr && control->interruption_requested()) {
      stopped = LimitReason::kInterrupt;
      break;
    }
    if (timer.elapsed_seconds() > time_limit) {
      stopped = LimitReason::kTime;
      break;
    }
    if (node_limit >= 0 && nodes >= node_limit) {
      stopped = LimitReason::kNodes;
      break;
    }
    TreeNode node = open.top();
    open.pop();
    if (node.bound >= incumbent) continue;  // nothing below it can improve
    // The MIP gap test (src/mip/branch_and_bound.cpp): best-first order makes this node's
    // bound the best open one.
    if (std::isfinite(incumbent)) {
      const double gap = incumbent - node.bound;
      if (gap > 0.0 &&
          (gap <= abs_target || gap / std::max(1.0, std::fabs(incumbent)) <= rel_target)) {
        open.push(node);
        break;
      }
    }
    ++nodes;
    // Each relaxation gets what is left of the solve's budget, not a fresh one.
    quiet.set_double("time_limit", std::max(0.0, time_limit - timer.elapsed_seconds()));
    with_bounds(node, &work);
    const Solution r = solve_nlp_relaxation(work, quiet, node.start, control, silent);
    nlp_iterations += r.iterations;
    if (r.status == SolveStatus::kTimeLimit || r.status == SolveStatus::kInterrupted) {
      stopped =
          r.status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
      open.push(node);
      break;
    }
    const bool bound_valid = r.status == SolveStatus::kOptimal;
    if (!is_point(r.status)) {
      if (r.status == SolveStatus::kLocallyInfeasible || r.status == SolveStatus::kInfeasible) {
        if (assumed || !certify_empty(node)) {
          lose(fmt::format("node {}: an infeasible relaxation could not be certified empty",
                           nodes));
        }
        continue;
      }
      lose(fmt::format("node {}: the relaxation ended {} ({})", nodes, to_string(r.status),
                       r.message));
      continue;
    }
    if (!bound_valid) lose("a relaxation was solved only to a local optimum");
    const double value = sense * r.objective;
    if (value >= incumbent) continue;
    // The most fractional integer column.
    Index branch = -1;
    double most = int_tol;
    for (const Index j : integers) {
      const double v = r.col_value[static_cast<std::size_t>(j)];
      const double frac = std::fabs(v - std::round(v));
      if (frac > most) {
        most = frac;
        branch = j;
      }
    }
    if (branch < 0) {
      // Integral: fix the integers where they are and solve the rest for a verified point.
      TreeNode fixed = node;
      for (const Index j : integers) {
        const double v = std::round(r.col_value[static_cast<std::size_t>(j)]);
        fixed.lower[static_cast<std::size_t>(j)] = fixed.upper[static_cast<std::size_t>(j)] = v;
      }
      with_bounds(fixed, &work);
      const Solution s = solve_nlp_relaxation(work, quiet, r.col_value, control, silent);
      nlp_iterations += s.iterations;
      if (is_point(s.status) && sense * s.objective < incumbent) {
        incumbent = sense * s.objective;
        best = s;
        logger.verbose("MINLP node {}: incumbent {:.10g}", nodes, s.objective);
      } else if (!is_point(s.status)) {
        lose(fmt::format("node {}: the integer point's NLP ended {}", nodes,
                         to_string(s.status)));
      }
      continue;
    }
    const double v = r.col_value[static_cast<std::size_t>(branch)];
    TreeNode down = node, up = node;
    down.upper[static_cast<std::size_t>(branch)] = std::floor(v);
    up.lower[static_cast<std::size_t>(branch)] = std::ceil(v);
    for (TreeNode* child : {&down, &up}) {
      child->bound = bound_valid ? value : node.bound;
      child->depth = node.depth + 1;
      child->start = r.col_value;
      open.push(std::move(*child));
    }
  }
  open_bound = open.empty() ? incumbent : std::min(incumbent, open.top().bound);

  out = std::isfinite(incumbent) ? best : Solution{};
  out.algorithm = "minlp-nlp-bnb";
  out.nodes = nodes;
  out.iterations = nlp_iterations;
  out.stopped_by = stopped;
  const bool have = std::isfinite(incumbent);
  const bool proved = !proof_lost && !assumed;
  if (!have) {
    // No point: the worst representable objective and an empty col_value, so the answer
    // claims none (claims_a_point, #505). Found by test_minlp_minlplib.cpp: clay0203m stopped
    // by a 20 s limit before its first incumbent reported a "point" with objective 0.
    out.objective = sense * kInfinity;
    out.dual_bound = proof_lost || assumed ? sense * -kInfinity : sense * open_bound;
  }
  if (have) {
    out.objective = sense * incumbent;
    out.dual_bound = proved ? sense * open_bound : sense * -kInfinity;
    out.absolute_gap = proved ? std::fabs(incumbent - open_bound) : kInfinity;
    out.relative_gap =
        proved ? out.absolute_gap / std::max(1.0, std::fabs(incumbent)) : kInfinity;
  }
  if (stopped != LimitReason::kNone) {
    out.status = have ? SolveStatus::kFeasible : status_for(stopped);
    out.message = fmt::format("stopped by the {} after {} nodes{}", to_string(stopped), nodes,
                              have ? "; the best integer point found is reported" : "");
  } else if (have && proved) {
    out.status = SolveStatus::kOptimal;
    out.message = fmt::format(
        "a convex MINLP: the incumbent is within the gap target of the best bound after {} "
        "nodes of NLP-based branch and bound",
        nodes);
  } else if (have) {
    out.status = SolveStatus::kFeasible;
    out.message =
        assumed ? "convexity asserted (nlp_assume_convex), not proved: the best "
                  "integer point found, with no optimality claim and no bound"
                : "the best integer point found; not proved optimal because " + lost_because;
  } else if (proved) {
    out.status = SolveStatus::kInfeasible;
    out.message = fmt::format(
        "every node of the convex relaxation's tree was certified empty or pruned ({} nodes)",
        nodes);
  } else {
    out.status = SolveStatus::kNumericalError;
    out.message = "no integer point was found and the search could not certify every node: " +
                  (assumed ? std::string("convexity was asserted, not proved") : lost_because);
  }
  out.solve_seconds = timer.elapsed_seconds();
  logger.info("MINLP branch and bound: {} after {} nodes - {}", to_string(out.status), nodes,
              out.message);
  return out;
}

}  // namespace sankhya::nlp
