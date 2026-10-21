// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MIQCP by branch and bound over the conic interior point (#755, item 4).
//
// Land and Doig, "An automatic method of solving discrete programming problems",
// Econometrica 28 (1960); for the convex nonlinear case Gupta and Ravindran, "Branch and
// bound experiments in convex nonlinear integer programming", Management Science 31 (1985).
// Every node is the continuous convex problem with tightened integer bounds, solved by
// solve_conic(); its Wolfe dual bound (or its objective when that bound is not finite)
// prunes the node against the incumbent to mip_relative_gap / mip_absolute_gap. The search is
// depth first and branches on the most fractional column. A relaxation point that is
// integral to the integrality tolerance is an incumbent as it stands: solve_conic() already
// measured its feasibility on the model's own rows and cones.
//
// ponytail: no cuts, no presolve, no warm start; each node is a cold interior point solve.
// Fine for the small MIQCPs this is built for; the LP tree (src/mip) is the place to grow.
//
// A node the interior point could not settle (neither optimal nor infeasible) is not pruned
// and is not branched: the search records it and the answer is at best kFeasible, never
// kOptimal, so a numerical failure cannot become a wrong proof.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/options.hpp"
#include "sankhya/qcqp.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::conic {

Solution solve_miqcp(const QcqpModel& model, const Options& options, SolveControl* control) {
  const Timer timer;
  const Model& lin = model.linear;
  const double sigma = lin.sense_multiplier();
  const double rel_gap = options.get_double("mip_relative_gap");
  const double abs_gap = options.get_double("mip_absolute_gap");
  const std::int64_t node_limit = options.get_int("node_limit");
  const double time_limit = options.get_double("time_limit");

  struct Node {
    std::vector<double> lower, upper;
  };
  std::vector<Node> open{{lin.col_lower, lin.col_upper}};
  QcqpModel relaxed = model;
  std::fill(relaxed.linear.col_type.begin(), relaxed.linear.col_type.end(),
            VarType::kContinuous);

  Solution best;
  double incumbent = kInfinity;  // minimization space
  Count nodes = 0;
  Count iterations = 0;
  Count unsettled = 0;
  bool stopped = false;
  bool by_time = false;
  double open_bound = kInfinity;  // smallest bound among nodes left unexplored
  while (!open.empty()) {
    if ((node_limit >= 0 && nodes >= node_limit) || timer.elapsed_seconds() >= time_limit) {
      stopped = true;
      by_time = timer.elapsed_seconds() >= time_limit;
      break;
    }
    Node node = std::move(open.back());
    open.pop_back();
    ++nodes;
    relaxed.linear.col_lower = node.lower;
    relaxed.linear.col_upper = node.upper;
    const Solution s = solve_conic(relaxed, options, control);
    iterations += s.iterations;
    if (s.status == SolveStatus::kInfeasible) continue;
    if (s.status != SolveStatus::kOptimal) {
      ++unsettled;
      continue;
    }
    const double bound = std::min(
        sigma * s.objective, std::isfinite(s.dual_bound) ? sigma * s.dual_bound : kInfinity);
    const double tolerance = std::max(abs_gap, rel_gap * std::fabs(incumbent));
    if (incumbent < kInfinity && bound >= incumbent - tolerance) continue;

    Index branch = -1;
    double most = tol::kIntegrality;
    for (Index j = 0; j < lin.num_cols(); ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (lin.col_type[u] != VarType::kInteger) continue;
      const double v = s.col_value[u];
      const double frac = std::fabs(v - std::round(v));
      if (frac > most) {
        most = frac;
        branch = j;
      }
    }
    if (branch < 0) {
      if (sigma * s.objective < incumbent) {
        incumbent = sigma * s.objective;
        best = s;
      }
      continue;
    }
    const auto b = static_cast<std::size_t>(branch);
    Node down = node;
    down.upper[b] = std::floor(s.col_value[b]);
    node.lower[b] = std::ceil(s.col_value[b]);
    // Down pushed last: explored first. The bound rides along only for the final report.
    open.push_back(std::move(node));
    open.push_back(std::move(down));
    open_bound = std::min(open_bound, bound);
  }

  // The bound a stopped search proves is the weakest of what is still open; a finished one
  // proves the incumbent (or infeasibility). Children carry their parent's bound, not their
  // own, so an open node's bound is taken as the smallest parent bound ever branched on.
  const bool complete = !stopped && unsettled == 0;
  Solution out = incumbent < kInfinity ? best : Solution{};
  out.algorithm = "conic-bnb";
  out.nodes = nodes;
  out.iterations = iterations;
  out.solve_seconds = timer.elapsed_seconds();
  if (complete) {
    out.status = incumbent < kInfinity ? SolveStatus::kOptimal : SolveStatus::kInfeasible;
    out.dual_bound = incumbent < kInfinity ? out.objective : sigma * kInfinity;
  } else {
    const SolveStatus limit = by_time ? SolveStatus::kTimeLimit : SolveStatus::kNodeLimit;
    out.status = incumbent < kInfinity ? SolveStatus::kFeasible
                 : stopped             ? limit
                                       : SolveStatus::kNotSolved;
    // An unsettled node proves nothing below it; otherwise every open node descends from a
    // branched parent whose bound is at least open_bound.
    out.dual_bound =
        sigma * (unsettled > 0 || nodes == 0 ? -kInfinity : std::min(open_bound, incumbent));
  }
  if (incumbent < kInfinity) {
    out.absolute_gap = std::fabs(out.objective - out.dual_bound);
    out.relative_gap = out.absolute_gap / std::max(1.0, std::fabs(out.objective));
  }
  out.message = fmt::format(
      "MIQCP branch and bound on the conic interior point: {} node(s), {} the interior point "
      "could not settle, search {}",
      nodes, unsettled,
      complete ? "complete" : (stopped ? "stopped by a limit" : "incomplete"));
  return out;
}

}  // namespace sankhya::conic
