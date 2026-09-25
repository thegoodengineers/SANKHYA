// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: MIQP node relaxations by the QP interior point (#494).
//
// With `miqp_node_ipm` on, a node QP is solved by the proximal interior point of #490 instead
// of the first-order Condat-Vu engine, and the node is pruned on a bound that holds for ANY
// primal point and row multipliers the solve returns, not on the objective of its point.
//
// THE BOUND. In minimise space the node QP is min F(x) = s (c'x + x'Qx/2) over the rows and
// the node's box, s the sense, sQ positive semidefinite (the convexity check refuses anything
// else). For any point p, convexity gives F(x) >= F(p) + g'(x - p) with g = s (c + Q p), so
//
//     min F  >=  -s p'Qp/2  +  min { g'x : rows, box },
//
// and the LP on the right is bounded below, for any multipliers y, by the Neumaier-Shcherbina
// bound (src/core/safe_bound.hpp), which is computed with outward rounding. At an optimal
// (p, y) both inequalities are tight; away from it the bound is weaker but still a bound, so
// an IPM stopped early, or a point accurate only to its tolerance, cannot make the search
// fathom the subtree holding the optimum. What is NOT rounded outward is the linearisation
// itself (Q p and p'Qp, in round-to-nearest): its error is of the order of n ulps of
// |Q||p|, which can_prune()'s kMiqpNodeTolerance margin covers on the problems this serves.
//
// A NODE THE IPM CANNOT FINISH. #490's first slice has no infeasibility detection: an
// infeasible node runs to the iteration ceiling. The node QP and the node LP (the same rows
// and box, no objective) have the same feasible set, so the primal simplex decides it: an
// infeasible LP is an infeasible node, with its Farkas multipliers for conflict analysis;
// otherwise the node is re-solved by Condat-Vu, as without the option.
//
// NOT DONE: #494 also asks for a warm start from the parent's iterate. The IPM has no
// warm-start entry point yet (#490's "Not done"), so every node starts cold.
//
// References:
//   Friedlander & Orban, "A primal-dual regularized interior-point method for convex
//     quadratic programs", Math. Programming Computation 4 (2012) - the node solver (#490).
//   Neumaier & Shcherbina, "Safe bounds in linear and mixed-integer linear programming",
//     Math. Programming 99 (2004) - the bound from any multipliers.
//   Gupta & Ravindran, "Branch and bound experiments in convex nonlinear integer
//     programming", Management Science 31(12) (1985) - branch and bound over convex nodes.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/safe_bound.hpp"

namespace sankhya::mip {

Solution BranchAndBound::solve_qp_node_ipm(const Options& options) {
  // A strong-branching probe (probe_options_) only scores a column; its status is read for
  // kInfeasible alone, so it is not worth the fallback below.
  const bool node_solve = &options == &node_options_;
  Solution ipm = qp::solve_convex_qp_ipm(working_, node_solve ? node_ipm_options_ : options,
                                         logger_, control_);
  if (!node_solve) return ipm;
  ++miqp_ipm_nodes_;
  if (ipm.status == SolveStatus::kOptimal || ipm.status == SolveStatus::kTimeLimit ||
      ipm.status == SolveStatus::kInterrupted) {
    return ipm;
  }
  ++miqp_ipm_fallbacks_;
  logger_.verbose("MIQP node: the interior point returned {} ({}); deciding the node by its LP",
                  to_string(ipm.status), ipm.message);

  Model feasibility = working_;
  const Index n = feasibility.num_cols();
  feasibility.hessian.reset(n, n);
  feasibility.hessian.finalize();
  std::fill(feasibility.col_cost.begin(), feasibility.col_cost.end(), 0.0);
  Solution lp = solve_primal_simplex(feasibility, options, logger_, control_);
  if (lp.status == SolveStatus::kInfeasible || lp.status == SolveStatus::kTimeLimit ||
      lp.status == SolveStatus::kInterrupted) {
    if (lp.status == SolveStatus::kInfeasible) ++miqp_ipm_lp_infeasible_;
    return lp;
  }
  return qp::solve_convex_qp(working_, options, logger_, control_);
}

double BranchAndBound::safe_qp_node_bound(const Solution& relaxation) {
  constexpr double kNone = -std::numeric_limits<double>::infinity();
  const auto n = static_cast<std::size_t>(working_.num_cols());
  const auto m = static_cast<std::size_t>(working_.num_rows());
  const std::vector<double>& p = relaxation.col_value;
  if (p.size() != n || relaxation.row_dual.size() != m ||
      !std::all_of(p.begin(), p.end(), [](double v) { return std::isfinite(v); })) {
    ++miqp_safe_bound_infinite_;
    return kNone;
  }
  // Q p from the lower triangle: a stored off-diagonal entry stands for both of its
  // positions in the symmetric matrix, the diagonal for one (Model::evaluate_objective).
  std::vector<double> qp(n, 0.0);
  const SparseMatrix& h = working_.hessian;
  for (Index j = 0; j < h.num_cols(); ++j) {
    const ColumnView column = h.column(j);
    const auto uj = static_cast<std::size_t>(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto ui = static_cast<std::size_t>(column.rows[k]);
      if (ui >= n || uj >= n) continue;
      qp[ui] += column.values[k] * p[uj];
      if (ui != uj) qp[uj] += column.values[k] * p[ui];
    }
  }
  const double sense = working_.sense_multiplier();
  double pqp = 0.0;
  std::vector<double> gradient(n);
  for (std::size_t j = 0; j < n; ++j) {
    pqp += p[j] * qp[j];
    gradient[j] = sense * (working_.col_cost[j] + qp[j]);
  }
  std::vector<double> y(m);
  for (std::size_t i = 0; i < m; ++i) y[i] = sense * relaxation.row_dual[i];

  SafeBoundProblem problem;
  problem.matrix = &working_.matrix;
  problem.cost = gradient;
  problem.row_lower = working_.row_lower;
  problem.row_upper = working_.row_upper;
  problem.col_lower = working_.col_lower;
  problem.col_upper = working_.col_upper;
  const SafeBound linear = safe_dual_bound(problem, y);
  const double constant = -0.5 * sense * pqp;
  if (!std::isfinite(linear.value) || !std::isfinite(constant)) {
    ++miqp_safe_bound_infinite_;
    return kNone;
  }
  // One step outward for the final sum; the linearisation's own rounding is the header's.
  return std::nextafter(linear.value + constant, kNone);
}

void BranchAndBound::report_miqp_ipm() const {
  if (!miqp_node_ipm_) return;
  logger_.info(
      "MIQP node IPM (#494): {} node QP(s), {} not finished by the interior point ({} of them "
      "proved infeasible by the node LP, the rest re-solved by Condat-Vu); {} node(s) with no "
      "finite linearised bound",
      miqp_ipm_nodes_, miqp_ipm_fallbacks_, miqp_ipm_lp_infeasible_, miqp_safe_bound_infinite_);
}

}  // namespace sankhya::mip
