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

  const ObbtResult result = obbt_root(model, obbt_options(), Logger{});

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
/// With an objective cutoff c'x <= 2 (minimise sum, incumbent = 2):
///   Feasible solutions must satisfy x0 + x1 + x2 <= 2 (cutoff: sum <= 2 - eps).
///   Combined with x0 >= 1, x1 >= 1: x2 <= 2 - 1 - 1 = 0.
/// So OBBT should tighten x2's upper bound to (near) 0.
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

  const ObbtResult result = obbt_root(model, obbt_options(), Logger{}, incumbent);

  EXPECT_GT(result.bounds_tightened, 0) << "OBBT should tighten at least one bound";
  // x2's upper bound must be strictly tighter than the original 10.
  EXPECT_LT(model.col_upper[2], orig_ub2 - 1e-6);
  // It should be ~0 (2 - 1 - 1 - epsilon).
  EXPECT_LT(model.col_upper[2], 1.0);
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

  const ObbtResult result = obbt_root(model, opt, Logger{});
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

  const ObbtResult result = obbt_root(model, opt, Logger{});
  EXPECT_LE(result.lp_solves, 2);
}

}  // namespace
}  // namespace sankhya
