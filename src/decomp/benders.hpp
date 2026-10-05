// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Benders decomposition of an LP over linking columns (#525).
//
// THE METHOD. Benders, "Partitioning procedures for solving mixed-variables programming
// problems", Numerische Mathematik 4 (1962). With the linking columns y fixed, an LP whose rows
// fall into blocks is one small LP per block; the block's optimal value Q_b(y) is a convex
// piecewise-linear function of y, so a subgradient at any y gives a cut that holds everywhere:
//
//     theta_b >= Q_b(y^) + g_b'(y - y^),        g_b = - A_{b,y}' lambda_b
//
// where lambda_b are the block LP's optimal row duals at y^. A master LP over y and one
// epigraph column theta_b per block, with every cut so far, is a RELAXATION, so its optimum is
// a lower bound; the blocks at its y^ give a feasible point, so their sum is an upper bound.
// They meet after finitely many rounds for an LP.
//
// WHEN A BLOCK IS INFEASIBLE FOR y^, its elastic LP - the same rows each given a surplus and a
// slack, minimising their total - is always feasible, its value V_b(y) is convex and zero
// exactly where the block is feasible, and the same subgradient argument gives a FEASIBILITY
// cut  V_b(y^) + g_b'(y - y^) <= 0  that every feasible y satisfies and y^ does not (the l1
// elastic form; Benders's own cut is the same inequality in terms of an extreme ray).
//
// THE START. theta_b has no bound until a cut exists, so it starts at the optimum of block b's
// LP with the coupling relaxed over the linking columns' own bounds - a value no y can go
// below. A block that is unbounded there, or an unbounded linking column that matters, is a
// model this method does not take, and it declines.
//
// THE ANSWER IS NOT TRUSTED BECAUSE THE METHOD CONVERGED. The primal point is the linking
// columns and the blocks' optimal points at the last y^. The dual vector is built from the
// master's own multipliers: a block's row duals are the combination of the duals behind its
// cuts, weighted by the master's multipliers on them (the multipliers of the optimality cuts
// sum to one per block at a master optimum), and the master's own rows keep their multipliers.
// By construction the reduced cost of every linking column then equals the master's, and of
// every block column the block's. solve() measures that pair - primal and dual infeasibility,
// complementarity, and the gap against the master's bound - exactly as for any engine, and a
// pair that does not pass is discarded and the LP solved monolithically. Decline rather than
// lie.
//
// WHAT IT TAKES: a minimisation LP, no integer columns, no quadratic objective. Anything else,
// and any block LP that is unbounded or any master that is not optimal, is declined with the
// reason, and solve() runs the model monolithically as if the option were off.
#pragma once

#include <optional>
#include <string>

#include "decomp/structure.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::decomp {

struct BendersOutcome {
  /// The answer, when the method produced one. A solution with a limit status is final (the
  /// budget is spent); one with status `optimal` still has to pass solve()'s measurement.
  std::optional<Solution> solution;
  /// Why there is no solution, when there is none.
  std::string declined;
};

/// Solve `model` by Benders decomposition over `structure` (kind = linking columns).
[[nodiscard]] BendersOutcome solve_benders(const Model& model, const BlockStructure& structure,
                                           const Options& options, SolveControl* control,
                                           Logger& logger);

}  // namespace sankhya::decomp
