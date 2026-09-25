// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP by NLP-based branch and bound (NLP stage 3).
//
// THE METHOD. Gupta and Ravindran, "Branch and bound experiments in convex nonlinear integer
// programming", Management Science 31(12) (1985): solve the continuous relaxation of each
// node with the NLP engine (nlp_solve.hpp), prune a node whose relaxation cannot beat the
// incumbent or has no feasible point, take an integral relaxation as a candidate, and
// otherwise branch on an integer column x_j with a fractional value v into x_j <= floor(v)
// and x_j >= ceil(v). Nodes are taken best bound first. Related: Dakin, "A tree-search
// algorithm for mixed integer programming problems", The Computer Journal 8(3) (1965), the
// dichotomy branching; Duran and Grossmann (1986) and Fletcher and Leyffer (1994) are the
// outer-approximation alternative (#528), not this.
//
// VALID ONLY FOR A CONVEX RELAXATION, AND ONLY CLAIMED THERE. A node's relaxation value is a
// lower bound on every integer point below it only when the relaxation's optimum is GLOBAL,
// which the NLP engine asserts (status `optimal`) exactly when the model is proved convex
// by the composition rules. So the method runs only when NonlinearModel::convexity() proves
// the continuous relaxation convex, and REFUSES otherwise, with the reason. With the option
// nlp_assume_convex it runs anyway as a heuristic on the caller's word, and then reports the
// best point it found as `feasible`, never `optimal`, and no bound.
//
// PRUNING AN INFEASIBLE NODE IS CERTIFIED. The interior point's `locally_infeasible` is a
// local statement; for a convex relaxation it is made global here by solving the convex
// minimum-violation problem min sum(p + n) s.t. L <= g(x) + p - n <= U over the node's box
// to `optimal`: a positive minimum proves the node empty. A node that cannot be certified
// either way is not pruned silently - the proof is marked lost and the answer says so.
//
// WHAT `optimal` MEANS HERE. The incumbent is within mip_relative_gap / mip_absolute_gap of
// the best open node's bound (the MIP definition, src/mip/branch_and_bound.cpp), every node
// having been resolved with a certificate. The bounds are relaxation optima at KKT points to
// the project tolerances, as in every NLP-based branch and bound; the incumbent is the
// integer columns fixed at integers and the remaining NLP solved, verified by the KKT check.
//
// WHAT IS REUSED from the MIP tree: its gap test, its limits (node_limit, time_limit) and
// its tolerances and options; the tree itself is written here, since the MIP tree is built
// around LP relaxations with bases, cuts and pseudocosts that an NLP node does not have.
#pragma once

#include "nlp/nonlinear_model.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::nlp {

/// Solve a nonlinear model with integer columns (a model without them goes to solve_nlp()).
[[nodiscard]] Solution solve_minlp(const NonlinearModel& model, const Options& options,
                                   SolveControl* control = nullptr);

}  // namespace sankhya::nlp
