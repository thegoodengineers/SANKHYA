// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Dantzig-Wolfe decomposition of an LP over linking rows (#1048).
//
// THE METHOD. Dantzig and Wolfe, "Decomposition principle for linear programs", Operations
// Research 8(1), 1960; the textbook treatment followed here is Lasdon, "Optimization Theory for
// Large Systems", Macmillan 1970 (chapter 3), and for column generation in general Desrosiers
// and Lubbecke, "A primer in column generation", in "Column Generation", Springer 2005.
//
// With the linking rows set aside, the columns fall into blocks with rows of their own, and
// each block's own rows and column bounds describe a polyhedron X_b. Every point of X_b is a
// convex combination of its extreme points plus a nonnegative combination of its extreme rays,
// so the LP is the MASTER over weights lambda (points) and mu (rays):
//
//   min   sum c'x_p lambda_p + sum c'd_r mu_r + c_D'x_D
//   s.t.  l_0 <= sum (A_0 x_p) lambda_p + sum (A_0 d_r) mu_r + A_0D x_D <= u_0   (duals pi)
//         sum_{p in b} lambda_p = 1                     for each block b       (duals sigma_b)
//         lambda, mu >= 0,  x_D within its bounds
//
// where x_D are the columns that are in linking rows only (or in no row), kept in the master
// as they are. The RESTRICTED master holds the points and rays found so far and is solved by
// the dual simplex. Its duals price each block: the block LP  min (c_b - A_0b' pi)'x  over X_b
// either finds a point (or a ray) whose reduced cost  value - sigma_b  is negative, which
// enters the master, or proves that none exists. When no block has one, the restricted
// master's optimum is the LP's. For any pi, the restricted master's value plus the blocks'
// (nonpositive) reduced costs is a lower bound (the Lagrangian bound, Lasdon section 3.4),
// so the gap between the two is measured every round and the method stops when it closes.
//
// THE START. Every block's first point is its optimum for its own costs, and the master is
// made feasible in a first phase that minimises the violation of the linking rows (one
// surplus and one slack column each, the textbook phase one, Lasdon section 3.3) with the
// blocks priced at zero cost; the second phase drops those columns and prices at the true
// costs. A block with no point makes the LP infeasible, and that is left to the monolithic
// engines to prove, as is a master the first phase cannot make feasible.
//
// THE ANSWER IS NOT TRUSTED BECAUSE THE METHOD CONVERGED. The primal point is the master's
// direct columns and, for each block, the combination of its points and rays with the
// master's weights. The dual vector is the master's pi on the linking rows and each block
// LP's own row duals, from the last round, on its rows: a block column's reduced cost is then
// exactly that block LP's. solve() measures that pair as it measures any engine's answer and
// discards it for the monolithic solve if it does not pass.
//
// WHAT IT TAKES: a minimisation LP with linking-row structure. It is reached only with
// `decomposition=dantzig-wolfe`; under `off` (the default) and `auto` it is never run.
#pragma once

#include "decomp/benders.hpp"
#include "decomp/structure.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::decomp {

/// Solve `model` by Dantzig-Wolfe decomposition over `structure` (kind = linking rows). The
/// outcome has the same meaning as Benders': a solution with status `optimal` still has to pass
/// solve()'s measurement, one with a limit status is final, none means solve monolithically.
[[nodiscard]] BendersOutcome solve_dantzig_wolfe(const Model& model,
                                                 const BlockStructure& structure,
                                                 const Options& options, SolveControl* control,
                                                 Logger& logger);

}  // namespace sankhya::decomp
