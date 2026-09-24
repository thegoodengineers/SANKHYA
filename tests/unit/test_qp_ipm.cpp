// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the proximal interior point for convex QP (#490), and the signed LDL^T it runs on.
//
// Every optimum below is worked out by hand in its comment, and every answer is also put
// through the in-process KKT check, so a test passes on the conditions and not only on a
// number that happens to match.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/kkt_check.hpp"
#include "la/ldl.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// min x1^2 + x2^2 - 2 x1 - 4 x2  s.t.  x1 + x2 <= 2,  x >= 0. Optimum (0.5, 1.5), objective
/// -4.5, row dual -1 (a <= row at a minimum), reduced costs zero.
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

/// min (x1^2 + x2^2 + x3^2) / 2  s.t.  x1 + x2 + x3 = 3,  1 <= x1 - x2 <= 2,  x1 free,
/// x2 >= 0, x3 fixed at 0.5. With x3 = 0.5, x1 + x2 = 2.5 and the unconstrained split (1.25,
/// 1.25) breaks x1 - x2 >= 1, so the ranged row is active at its lower end: x = (1.75, 0.75,
/// 0.5), objective (3.0625 + 0.5625 + 0.25) / 2 = 1.9375. Stationarity on x1 and x2:
/// 1.75 = y1 + y2 and 0.75 = y1 - y2, so y = (1.25, 0.5), and y2 > 0 prices the lower end.
Model equality_ranged_free_fixed_qp() {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {0.0, 0.0, 0.0};
  model.col_lower = {-kInfinity, 0.0, 0.5};
  model.col_upper = {kInfinity, kInfinity, 0.5};
  model.col_type = {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {3.0, 1.0};
  model.row_upper = {3.0, 2.0};
  model.matrix.reset(2, 3);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, -1.0);
  model.matrix.finalize();
  model.hessian.reset(3, 3);
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.add_entry(1, 1, 1.0);
  model.hessian.add_entry(2, 2, 1.0);
  model.hessian.finalize();
  return model;
}

void expect_backed(const Model& model, const Solution& s) {
  const KktVerdict verdict = check_qp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
}

TEST(QpIpm, SolvesAnInequalityQpToItsKnownOptimum) {
  const Model model = inequality_qp();
  Logger logger(nullptr);
  const Solution s = qp::solve_convex_qp_ipm(model, quiet(), logger);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.algorithm, "qp-ipm");
  EXPECT_NEAR(s.objective, -4.5, 1e-8);
  EXPECT_NEAR(s.col_value[0], 0.5, 1e-7);
  EXPECT_NEAR(s.col_value[1], 1.5, 1e-7);
  EXPECT_NEAR(s.row_dual[0], -1.0, 1e-7);
  EXPECT_GT(s.iterations, 0);
  expect_backed(model, s);
}

TEST(QpIpm, HandlesEqualityRangedFreeAndFixedTogether) {
  const Model model = equality_ranged_free_fixed_qp();
  Logger logger(nullptr);
  const Solution s = qp::solve_convex_qp_ipm(model, quiet(), logger);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, 1.9375, 1e-8);
  EXPECT_NEAR(s.col_value[0], 1.75, 1e-7);
  EXPECT_NEAR(s.col_value[1], 0.75, 1e-7);
  EXPECT_DOUBLE_EQ(s.col_value[2], 0.5);  // substituted out, reported exactly
  EXPECT_NEAR(s.row_dual[0], 1.25, 1e-7);
  EXPECT_NEAR(s.row_dual[1], 0.5, 1e-7);
  expect_backed(model, s);
}

TEST(QpIpm, AMaximizationIsReportedInTheModelsOwnSense) {
  // max -(the objective above): the same point, the objective and the duals negated.
  Model model = equality_ranged_free_fixed_qp();
  model.sense = ObjSense::kMaximize;
  model.hessian.reset(3, 3);
  model.hessian.add_entry(0, 0, -1.0);
  model.hessian.add_entry(1, 1, -1.0);
  model.hessian.add_entry(2, 2, -1.0);
  model.hessian.finalize();
  Logger logger(nullptr);
  const Solution s = qp::solve_convex_qp_ipm(model, quiet(), logger);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, -1.9375, 1e-8);
  EXPECT_NEAR(s.col_value[0], 1.75, 1e-7);
  EXPECT_NEAR(s.row_dual[0], -1.25, 1e-7);
  EXPECT_NEAR(s.row_dual[1], -0.5, 1e-7);
  expect_backed(model, s);
}

TEST(QpIpm, TheNonConvexRefusalIsUnchanged) {
  Model model = inequality_qp();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 1, -1.0);
  model.hessian.finalize();
  Logger logger(nullptr);
  const Solution ipm = qp::solve_convex_qp_ipm(model, quiet(), logger);
  const Solution condat_vu = qp::solve_convex_qp(model, quiet(), logger);
  EXPECT_EQ(ipm.status, SolveStatus::kModelError);
  EXPECT_EQ(ipm.status, condat_vu.status);
  EXPECT_EQ(ipm.message, condat_vu.message);
  EXPECT_NE(ipm.message.find("not convex"), std::string::npos) << ipm.message;
}

TEST(QpIpm, SolveRunsItOnlyWhenAskedAndThroughTheSameGate) {
  const Model model = inequality_qp();
  Options options = quiet();
  EXPECT_EQ(options.get_string("qp_algorithm"), "condat-vu");  // default off
  const Solution by_default = solve(model, options);
  EXPECT_EQ(by_default.algorithm, "qp-condat-vu");

  options.set_string("qp_algorithm", "ipm");
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.algorithm, "qp-ipm");
  EXPECT_NEAR(s.objective, -4.5, 1e-8);
  EXPECT_EQ(s.message.find("#590"), std::string::npos) << s.message;
  expect_backed(model, s);
}

TEST(QpIpm, AnIterationLimitIsALimitNotAnOptimum) {
  const Model model = inequality_qp();
  Options options = quiet();
  options.set_int("iteration_limit", 2);
  Logger logger(nullptr);
  const Solution s = qp::solve_convex_qp_ipm(model, options, logger);
  EXPECT_EQ(s.status, SolveStatus::kIterationLimit) << s.message;
  EXPECT_EQ(s.iterations, 2);
}

TEST(QpIpm, AnUnknownAlgorithmNameIsRefused) {
  Options options = quiet();
  std::string error;
  EXPECT_FALSE(options.set_from_string("qp_algorithm", "admm", &error));
  EXPECT_FALSE(error.empty());
}

// ---- the signed factorization ----------------------------------------------------------

/// K = [-2 1; 1 3], lower triangle. Quasi-definite: D = (-2, 3.5) in the natural order.
SparseMatrix small_quasidefinite() {
  SparseMatrix k;
  k.reset(2, 2);
  k.add_entry(0, 0, -2.0);
  k.add_entry(1, 0, 1.0);
  k.add_entry(1, 1, 3.0);
  k.finalize();
  return k;
}

TEST(SparseLdlQuasidefinite, FactorsAndSolvesWithNegativePivots) {
  const SparseMatrix k = small_quasidefinite();
  SparseLdl ldl;
  ASSERT_TRUE(ldl.analyze(k));
  ASSERT_TRUE(ldl.factorize_quasidefinite(k, {-1, 1}, 1e-12));
  EXPECT_EQ(ldl.regularized_pivots(), 0);
  // K x = (0, 7) has x = (1, 2): -2 + 2 = 0, 1 + 6 = 7.
  std::vector<double> b = {0.0, 7.0};
  ldl.solve(b.data());
  EXPECT_NEAR(b[0], 1.0, 1e-14);
  EXPECT_NEAR(b[1], 2.0, 1e-14);
}

TEST(SparseLdlQuasidefinite, APivotOfTheWrongSignIsCountedNotUsed) {
  const SparseMatrix k = small_quasidefinite();
  SparseLdl ldl;
  ASSERT_TRUE(ldl.analyze(k));
  // Declaring both pivots positive: the -2 has the wrong sign and is lifted and counted, the
  // signal the interior point raises its proximal parameters on.
  ASSERT_TRUE(ldl.factorize_quasidefinite(k, {1, 1}, 1e-12));
  EXPECT_GE(ldl.regularized_pivots(), 1);
  // And the positive-definite factorize() treats it the same way it always has.
  ASSERT_TRUE(ldl.factorize(k, 1e-12));
  EXPECT_GE(ldl.regularized_pivots(), 1);
}

TEST(SparseLdlQuasidefinite, TheWrongNumberOfSignsIsRefused) {
  const SparseMatrix k = small_quasidefinite();
  SparseLdl ldl;
  ASSERT_TRUE(ldl.analyze(k));
  EXPECT_FALSE(ldl.factorize_quasidefinite(k, {-1}, 1e-12));
}

}  // namespace
}  // namespace sankhya
