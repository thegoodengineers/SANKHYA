// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex MINLP by outer approximation (#528).
//
// THE METHOD. Duran and Grossmann, "An outer-approximation algorithm for a class of
// mixed-integer nonlinear programs", Mathematical Programming 36(3) (1986): a MILP MASTER over
// first-order linearizations of the convex nonlinear rows and objective, alternated with the
// convex NLP obtained by fixing the master's integer columns. The master is a RELAXATION of
// the MINLP - a convex function lies above every tangent - so its optimum is a lower bound;
// each NLP is a RESTRICTION, so its value is an upper bound. They are alternated, adding the
// linearizations at each NLP solution to the master, until the master's bound meets the best
// NLP value. The master is solved by the project's own MILP engine (solve(), branch and cut),
// the NLP by the interior point (nlp_solve.hpp): no new numerical kernel.
//
// THREE KINDS OF CUT, each valid for a convex model whatever the point it is taken at:
//   * at the NLP's optimum for the fixed integers (Duran and Grossmann): every nonlinear row
//     and the objective, linearized there;
//   * when the NLP for those integers is INFEASIBLE, at the minimizer of the convex
//     minimum-violation problem (minlp_bnb.hpp's violation_model): the linearizations there
//     cut off that integer assignment (Fletcher and Leyffer, "Solving mixed integer nonlinear
//     programs by outer approximation", Math. Programming 66 (1994));
//   * at the master's own point, for whatever it violates (the extended cutting plane method,
//     Westerlund and Pettersson, Computers & Chemical Engineering 19 (1995)): this is what
//     keeps every round making progress when the constraint qualification the classical
//     convergence proof needs does not hold, since a point that violates a row is separated
//     from it by that row's linearization there.
//
// WHICH SIDE OF A ROW IS LINEARIZED. NonlinearModel::convexity() proves the relaxation convex
// by requiring, per nonlinear row, a convex left side where the upper bound is finite and a
// concave one where the lower bound is (both finite: affine). So the linearization of the
// left side is a valid cut exactly on the sides that are finite, and an equality or range row
// is only ever one the left side of which is affine - its linearization is exact.
//
// THE OBJECTIVE is moved into the master through an epigraph column t >= f(x), when it is
// not linear, so the master's objective is linear: min t, with a cut f(x_k) + grad f(x_k)'(x -
// x_k) <= t at every point. Its lower bound is the continuous relaxation's value less
// kMinlpOaEpigraphSlack (tolerances.hpp), which is the least f can be anywhere on the
// feasible set - without it the master over the first few tangents can be unbounded.
//
// WHAT `optimal` MEANS. As for the tree (minlp_bnb.hpp): proved convex, and the incumbent -
// an NLP solution at fixed integers whose KKT check passed - within mip_relative_gap /
// mip_absolute_gap of the largest bound any master proved (its dual_bound, not its
// objective). A bound above a verified feasible value, beyond that tolerance, is never
// resolved in favour of either: it means a cut was not valid at the tolerance, and the answer
// is reported `feasible` with that said. Convexity asserted rather than proved
// (nlp_assume_convex) gives a heuristic: the best point found, never `optimal`, no bound.
#pragma once

#include "nlp/nonlinear_model.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::nlp {

/// Solve a nonlinear model with integer columns by outer approximation. Called by
/// solve_minlp() when `minlp_method=oa`, after its convexity gate; `convexity_assumed` is true
/// when the relaxation was not proved convex but the caller asserted it.
[[nodiscard]] Solution solve_minlp_oa(const NonlinearModel& model, const Options& options,
                                      SolveControl* control, bool convexity_assumed);

}  // namespace sankhya::nlp
