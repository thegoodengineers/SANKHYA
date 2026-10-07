// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the active-set finish for the QP interior point (#980), behind qp_ipm_finish.
//
// Every optimum below is worked out by hand in its comment, and every answer the finish
// accepts is also put through the in-process KKT check (#590's own gate), so a test passes
// on the conditions and not only on a number that happens to match.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/kkt_check.hpp"
#include "qp/qp_ipm_finish.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

void expect_backed(const Model& model, const Solution& s) {
  const KktVerdict verdict = check_qp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
}

/// min x1^2 + x2^2 - 2 x1 - 4 x2  s.t.  x1 + x2 <= 2,  x >= 0. Optimum (0.5, 1.5), objective
/// -4.5: the same instance test_qp_ipm.cpp uses, with no bound close to active, so the
/// active-set finish has nothing to pin and must be a no-op either way.
Model inequality_qp() {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {-2.0, -4.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {2.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 1, 2.0);
  model.hessian.finalize();
  return model;
}

/// min 1000 x1^2 - 2000 x1 + x2^2 - 2 x2  s.t.  x1 >= 1,  x2 >= 0,  x1 + x2 <= 10. The
/// unconstrained minimum of the first term alone is x1 = 1 exactly (derivative 2000 x1 -
/// 2000 = 0), so the lower bound x1 >= 1 sits exactly ON the unconstrained optimum: a
/// multiplier that has to carry whatever the interior point's iterate leaves of the
/// 1000-scaled term against however close its x1 lands to 1, the shape the issue's own
/// instances have (a row whose multiplier is 1e+03 needs its activity on the bound to
/// 1e-10). x2's unconstrained optimum is 1, strictly interior and not pinned. The true
/// optimum is x = (1, 1), objective 1000 - 2000 + 1 - 2 = -1001, row dual 0 (the x1 + x2 <=
/// 10 row is slack), z on x1's lower bound = 0 at the exact optimum (the bound is active but
/// the multiplier is 0: x1 = 1 is simultaneously unconstrained-optimal and bound-active,
/// a degenerate but legal KKT point) and z on x2's lower bound = 0 (x2 = 1 is interior).
Model tight_bound_large_curvature_qp() {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {-2000.0, -2.0};
  model.col_lower = {1.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {10.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 2000.0);
  model.hessian.add_entry(1, 1, 2.0);
  model.hessian.finalize();
  return model;
}

TEST(QpIpmFinish, OptionDefaultsOff) {
  const Options options = quiet();
  EXPECT_FALSE(options.get_bool("qp_ipm_finish"));
}

TEST(QpIpmFinish, OffLeavesTheSolveByteForByteUnchanged) {
  const Model model = inequality_qp();
  Logger logger(nullptr);

  Options off_default = quiet();
  const Solution baseline = qp::solve_convex_qp_ipm(model, off_default, logger);

  Options off_explicit = quiet();
  off_explicit.set_bool("qp_ipm_finish", false);
  const Solution explicit_off = qp::solve_convex_qp_ipm(model, off_explicit, logger);

  ASSERT_EQ(baseline.status, SolveStatus::kOptimal) << baseline.message;
  ASSERT_EQ(explicit_off.status, SolveStatus::kOptimal) << explicit_off.message;
  EXPECT_EQ(baseline.iterations, explicit_off.iterations);
  EXPECT_EQ(baseline.message, explicit_off.message);
  ASSERT_EQ(baseline.col_value.size(), explicit_off.col_value.size());
  for (std::size_t j = 0; j < baseline.col_value.size(); ++j) {
    EXPECT_DOUBLE_EQ(baseline.col_value[j], explicit_off.col_value[j]);
  }
  ASSERT_EQ(baseline.row_dual.size(), explicit_off.row_dual.size());
  for (std::size_t i = 0; i < baseline.row_dual.size(); ++i) {
    EXPECT_DOUBLE_EQ(baseline.row_dual[i], explicit_off.row_dual[i]);
  }
  EXPECT_DOUBLE_EQ(baseline.objective, explicit_off.objective);
}

TEST(QpIpmFinish, NeverDegradesAPointThatIsAlreadyFullyOptimal) {
  // No column is close to a bound here (both are strictly interior, see inequality_qp's own
  // comment), so the finish's own active-set test finds nothing to pin and takes no action:
  // turning the option on must reproduce the unpolished answer bit for bit.
  const Model model = inequality_qp();
  Logger logger(nullptr);

  Options off = quiet();
  const Solution without_finish = qp::solve_convex_qp_ipm(model, off, logger);

  Options on = quiet();
  on.set_bool("qp_ipm_finish", true);
  const Solution with_finish = qp::solve_convex_qp_ipm(model, on, logger);

  ASSERT_EQ(without_finish.status, SolveStatus::kOptimal) << without_finish.message;
  ASSERT_EQ(with_finish.status, SolveStatus::kOptimal) << with_finish.message;
  ASSERT_EQ(without_finish.col_value.size(), with_finish.col_value.size());
  for (std::size_t j = 0; j < without_finish.col_value.size(); ++j) {
    EXPECT_DOUBLE_EQ(without_finish.col_value[j], with_finish.col_value[j]);
  }
  expect_backed(model, with_finish);
}

TEST(QpIpmFinish, ATightlyActiveBoundWithALargeMultiplierStillVerifiesEitherWay) {
  // This is the shape the issue names: a bound whose unconstrained optimum sits exactly on
  // it, scaled by a large Hessian coefficient (1e+03, the same order the issue's qship12s
  // example names). With the finish off, the interior point may or may not close the
  // complementarity product to the gate on its own; with it on, the active-set finish either
  // pins x1 to its bound and solves the reduced equality system exactly (closing the product
  // to machine precision) or - the property under test - leaves the interior point's own
  // point alone when the reduced solve does not check out. Either way the returned answer
  // must still be the hand-worked optimum and pass the in-process KKT gate: "never worse" is
  // not just a claim about the product, it is a claim about the whole answer.
  const Model model = tight_bound_large_curvature_qp();
  Logger logger(nullptr);

  Options off = quiet();
  const Solution without_finish = qp::solve_convex_qp_ipm(model, off, logger);
  ASSERT_EQ(without_finish.status, SolveStatus::kOptimal) << without_finish.message;
  EXPECT_NEAR(without_finish.objective, -1001.0, 1e-5);
  EXPECT_NEAR(without_finish.col_value[0], 1.0, 1e-5);
  EXPECT_NEAR(without_finish.col_value[1], 1.0, 1e-5);
  expect_backed(model, without_finish);

  Options on = quiet();
  on.set_bool("qp_ipm_finish", true);
  const Solution with_finish = qp::solve_convex_qp_ipm(model, on, logger);
  ASSERT_EQ(with_finish.status, SolveStatus::kOptimal) << with_finish.message;
  EXPECT_NEAR(with_finish.objective, -1001.0, 1e-5);
  EXPECT_NEAR(with_finish.col_value[0], 1.0, 1e-5);
  EXPECT_NEAR(with_finish.col_value[1], 1.0, 1e-5);
  expect_backed(model, with_finish);

  // Whichever point was returned, it must be at least as good as the one without the finish:
  // never a worse objective (both are the true optimum to the hand-worked value, so this is
  // an equality within solver tolerance, not a looser bound).
  EXPECT_NEAR(with_finish.objective, without_finish.objective, 1e-6);
}

TEST(QpIpmFinish, AnInteriorOptimumIsLeftAloneWithTheOptionOn) {
  // x1's lower bound is 1 but its optimum is strictly interior (x1 = 10): the interior point
  // never lands near the bound, its products close on their own, and the finish does not
  // run. The guard itself is exercised directly by the ActiveSetFinish tests below.
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {-10.0};
  model.col_lower = {1.0};
  model.col_upper = {kInfinity};
  model.col_type = {VarType::kContinuous};
  model.row_lower = {};
  model.row_upper = {};
  model.matrix.reset(0, 1);
  model.matrix.finalize();
  model.hessian.reset(1, 1);
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.finalize();
  // min 0.5 x1^2 - 10 x1 s.t. x1 >= 1: unconstrained optimum x1 = 10, strictly interior
  // (bound 1 is not active at all), objective -50.
  Logger logger(nullptr);

  Options on = quiet();
  on.set_bool("qp_ipm_finish", true);
  const Solution s = qp::solve_convex_qp_ipm(model, on, logger);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.col_value[0], 10.0, 1e-6);
  EXPECT_NEAR(s.objective, -50.0, 1e-6);
  expect_backed(model, s);
}

// ---- active_set_finish called directly, so both of its outcomes are actually reached -------

struct Bounds {
  std::vector<bool> lower, upper;
};

Bounds finite_bounds(const qp::ipm_detail::Standard& s) {
  Bounds b;
  for (std::size_t j = 0; j < s.lower.size(); ++j) {
    b.lower.push_back(is_finite_bound(s.lower[j]));
    b.upper.push_back(is_finite_bound(s.upper[j]));
  }
  return b;
}

TEST(QpIpmFinish, ActiveSetFinishSnapsANearlyActiveBoundExactlyOntoIt) {
  // tight_bound_large_curvature_qp's optimum (1, 1), offered with x1 a hair above its bound,
  // as an interior point leaves it: the finish pins x1 = 1 EXACTLY and re-solves the rest.
  // Internal columns: x1, x2, then the slack w of the one-sided row x1 + x2 - w = 0.
  const qp::ipm_detail::Standard s =
      qp::ipm_detail::standardize(tight_bound_large_curvature_qp());
  ASSERT_EQ(s.cols, 3);
  const Bounds b = finite_bounds(s);
  std::vector<double> v = {1.0 + 1e-7, 1.0, 2.0 + 1e-7}, y = {0.0}, zl(3, 0.0), zu(3, 0.0);
  const Options options;
  ASSERT_TRUE(qp::ipm_detail::active_set_finish(
      s, b.lower, b.upper, options.get_double("qp_ipm_tolerance"),
      options.get_double("primal_feasibility_tolerance"),
      options.get_double("qp_ipm_regularization"), &v, &y, &zl, &zu));
  EXPECT_EQ(v[0], 1.0);  // on the bound to the bit, which is what closes the product
  EXPECT_NEAR(v[1], 1.0, 1e-7);
  EXPECT_NEAR(v[2], 2.0, 1e-7);
  EXPECT_GE(zl[0], -1e-7);
}

TEST(QpIpmFinish, ActiveSetFinishRefusesAPinWhoseMultiplierHasTheWrongSign) {
  // min 0.5 x1^2 - 10 x1 + 0.5 x2^2, x1 >= 1, x2 free: the optimum is (10, 0). Offered the
  // point (1, 0), the finish pins x1 at 1, where stationarity needs z = x1 - 10 = -9 on a
  // LOWER bound: the wrong sign. The guard must refuse and leave every argument untouched.
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {-10.0, 0.0};
  model.col_lower = {1.0, -kInfinity};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.matrix.reset(0, 2);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.add_entry(1, 1, 1.0);
  model.hessian.finalize();
  const qp::ipm_detail::Standard s = qp::ipm_detail::standardize(model);
  const Bounds b = finite_bounds(s);
  std::vector<double> v = {1.0, 0.0}, y, zl(2, 0.0), zu(2, 0.0);
  const std::vector<double> v_before = v;
  const Options options;
  EXPECT_FALSE(qp::ipm_detail::active_set_finish(
      s, b.lower, b.upper, options.get_double("qp_ipm_tolerance"),
      options.get_double("primal_feasibility_tolerance"),
      options.get_double("qp_ipm_regularization"), &v, &y, &zl, &zu));
  EXPECT_EQ(v, v_before);
  EXPECT_EQ(zl, std::vector<double>(2, 0.0));
}

}  // namespace
}  // namespace sankhya
