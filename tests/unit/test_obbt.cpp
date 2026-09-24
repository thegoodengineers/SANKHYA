// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for optimality-based bound tightening at the root (#515).
//
// We verify that obbt_root() tightens at least one bound on a tiny LP where
// the result is known analytically, and that the tightened bounds are strictly
// better than the original ones.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "mip/obbt.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

using mip::obbt_root;
using mip::ObbtResult;

Logger make_logger() {
  return Logger(nullptr);
}

Options obbt_options() {
  Options opt;
  opt.set_bool("log_to_console", false);
  opt.set_bool("mip_obbt", true);
  opt.set_int("mip_obbt_max_iters", 200);
  return opt;
}

/// Build a 2-variable LP:
///   x0 + x1 <= 4
///   x0, x1 in [0, 10]
///   minimise x0 + x1
///
/// The feasible region is bounded above (x0 <= 4, x1 <= 4 individually when
/// the other variable is 0), but the explicit upper bounds are 10.  OBBT with
/// no incumbent tightens nothing because the unconstrained max of x_j over
/// {x0+x1 <= 4, x >= 0} is still 4 which is below 10 — actually tightens!
///   max x0 s.t. x0 + x1 <= 4, x >= 0  ->  x0 = 4 (tight)
///   max x1 s.t. x0 + x1 <= 4, x >= 0  ->  x1 = 4 (tight)
/// So upper bounds should be tightened from 10 to 4.
TEST(Obbt, TightensUpperBoundsFromSingleConstraint) {
  Model model;
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {4.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();

  const double orig_ub0 = model.col_upper[0];
  const double orig_ub1 = model.col_upper[1];

  Logger logger1 = make_logger();
  const ObbtResult result = obbt_root(model, obbt_options(), logger1);

  EXPECT_GT(result.lp_solves, 0) << "OBBT should have solved at least one LP";
  EXPECT_GT(result.bounds_tightened, 0) << "OBBT should have tightened at least one bound";

  // Upper bounds must be strictly tighter.
  EXPECT_LT(model.col_upper[0], orig_ub0 - 1e-6)
      << "upper bound of x0 should have decreased from 10";
  EXPECT_LT(model.col_upper[1], orig_ub1 - 1e-6)
      << "upper bound of x1 should have decreased from 10";

  // Tightened values should be close to 4.
  EXPECT_NEAR(model.col_upper[0], 4.0, 1e-6);
  EXPECT_NEAR(model.col_upper[1], 4.0, 1e-6);
}

/// 3-variable LP:
///   x0 + x1 + x2 <= 3
///   x0 >= 1  (lower bound)
///   x1 >= 1
///   x0, x1, x2 in [0, 10]
/// With an objective cutoff (minimise the sum, incumbent 3): only strictly improving points
/// remain, x0 + x1 + x2 <= 3 - eps. With x0 >= 1 and x1 >= 1 that gives x2 <= 1 - eps, so
/// OBBT tightens x2's upper bound below 1.
TEST(Obbt, TightensWithIncumbentCutoff) {
  Model model;
  model.col_cost = {1.0, 1.0, 1.0};
  model.col_lower = {1.0, 1.0, 0.0};
  model.col_upper = {10.0, 10.0, 10.0};
  model.col_type = {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {3.0};
  model.matrix.reset(1, 3);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 1.0);
  model.matrix.finalize();

  const double orig_ub2 = model.col_upper[2];
  const double incumbent = 3.0;  // known feasible objective = 1+1+1

  Logger logger2 = make_logger();
  const ObbtResult result = obbt_root(model, obbt_options(), logger2, incumbent);

  EXPECT_GT(result.bounds_tightened, 0) << "OBBT should tighten at least one bound";
  // x2's upper bound must be strictly tighter than the original 10.
  EXPECT_LT(model.col_upper[2], orig_ub2 - 1e-6);
  // It should be ~0 (2 - 1 - 1 - epsilon).
  EXPECT_LT(model.col_upper[2], 1.0);
}

/// The cutoff follows the model's sense (review of #631). Maximise x0 + x1 + x2 over
/// x0 + x1 + x2 <= 3, x in [0, 10], with incumbent 2: an improving point has sum >= 2 + eps,
/// and x2 >= 0 already. Written the minimise way (sum <= 2 - eps) the row would have cut away
/// every improving point, and OBBT would have tightened x2's upper bound to about 2. The
/// correct row leaves x2 able to reach 3.
TEST(Obbt, TheCutoffFollowsTheObjectiveSense) {
  Model model;
  model.sense = ObjSense::kMaximize;
  model.col_cost = {1.0, 1.0, 1.0};
  model.col_lower = {0.0, 0.0, 0.0};
  model.col_upper = {10.0, 10.0, 10.0};
  model.col_type = {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {3.0};
  model.matrix.reset(1, 3);
  for (Index j = 0; j < 3; ++j) model.matrix.add_entry(0, j, 1.0);
  model.matrix.finalize();
  Logger logger5 = make_logger();
  (void)obbt_root(model, obbt_options(), logger5, 2.0);
  EXPECT_GE(model.col_upper[2], 3.0 - 1e-6) << "the improving point (0, 0, 3) was cut off";
  EXPECT_LE(model.col_upper[2], 3.0 + 1e-6);
}

/// An integer column's probed bound is rounded to an integer it cannot exclude, and never
/// past a point the LP admits: 2 x0 <= 7 with x0 integer in [0, 10] gives x0 <= 3.
TEST(Obbt, AnIntegerColumnGetsAnIntegerBound) {
  Model model;
  model.col_cost = {1.0};
  model.col_lower = {0.0};
  model.col_upper = {10.0};
  model.col_type = {VarType::kInteger};
  model.row_lower = {-kInfinity};
  model.row_upper = {7.0};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 2.0);
  model.matrix.finalize();
  Logger logger6 = make_logger();
  (void)obbt_root(model, obbt_options(), logger6);
  EXPECT_EQ(model.col_upper[0], 3.0);
  EXPECT_EQ(model.col_lower[0], 0.0);
}

/// When mip_obbt is false, obbt_root should be a no-op and return zero solves.
TEST(Obbt, SkipsWhenDisabled) {
  Model model;
  model.col_cost = {1.0};
  model.col_lower = {0.0};
  model.col_upper = {10.0};
  model.col_type = {VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {5.0};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();

  Options opt = obbt_options();
  opt.set_bool("mip_obbt", false);

  Logger logger3 = make_logger();
  const ObbtResult result = obbt_root(model, opt, logger3);
  EXPECT_EQ(result.lp_solves, 0);
  EXPECT_EQ(result.bounds_tightened, 0);
  EXPECT_NEAR(model.col_upper[0], 10.0, 1e-12);
}

/// Work-limit test: with mip_obbt_max_iters=2 and 5 free variables, OBBT
/// stops early and the result's lp_solves does not exceed the limit.
TEST(Obbt, RespectsMaxIters) {
  const int n = 5;
  Model model;
  model.col_cost.assign(n, 1.0);
  model.col_lower.assign(n, 0.0);
  model.col_upper.assign(n, 10.0);
  model.col_type.assign(n, VarType::kContinuous);
  // sum x_j <= 3
  model.row_lower = {-kInfinity};
  model.row_upper = {3.0};
  model.matrix.reset(1, n);
  for (int j = 0; j < n; ++j) model.matrix.add_entry(0, j, 1.0);
  model.matrix.finalize();

  Options opt = obbt_options();
  opt.set_int("mip_obbt_max_iters", 2);

  Logger logger4 = make_logger();
  const ObbtResult result = obbt_root(model, opt, logger4);
  EXPECT_LE(result.lp_solves, 2);
}

}  // namespace
}  // namespace sankhya
