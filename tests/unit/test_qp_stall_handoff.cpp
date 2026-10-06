// SPDX-License-Identifier: Apache-2.0
// SANKHYA - #981: the QP interior point's stall test generalized to the cold run
// (qp_ipm_stall_handoff), and the handoff each way to the first-order engine (#493)
// built on it.
//
// Every optimum below is worked out by hand, as test_qp_ipm.cpp's are.

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// min x1^2 + x2^2 - 2 x1 - 4 x2  s.t.  x1 + x2 <= 2,  x >= 0. Optimum (0.5, 1.5), objective
/// -4.5, row dual -1 (a <= row at a minimum), reduced costs zero. The same model
/// test_qp_ipm.cpp uses, so a known-by-hand optimum is available without repeating the
/// derivation here.
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
/// x2 >= 0, x3 fixed at 0.5. Optimum (1.75, 0.75, 0.5), objective 1.9375 (test_qp_ipm.cpp's
/// derivation). None of these numbers is exact in binary, so the residual floor a run
/// plateaus at is genuine floating-point noise, not an exact zero a trivial model can hit.
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

// ---- (d) the option defaults off, and off changes nothing -------------------------------

TEST(QpIpmStallHandoff, DefaultsOff) {
  EXPECT_FALSE(Options().get_bool("qp_ipm_stall_handoff"));
}

TEST(QpIpmStallHandoff, OffLeavesAnOrdinaryConvergentSolveExactlyAsItWas) {
  const Model model = inequality_qp();
  Logger logger(nullptr);
  const Solution without = qp::solve_convex_qp_ipm(model, quiet(), logger);
  Options with_option = quiet();
  with_option.set_bool("qp_ipm_stall_handoff", false);
  const Solution still_off = qp::solve_convex_qp_ipm(model, with_option, logger);
  ASSERT_EQ(without.status, SolveStatus::kOptimal) << without.message;
  ASSERT_EQ(still_off.status, SolveStatus::kOptimal) << still_off.message;
  EXPECT_EQ(without.iterations, still_off.iterations);
  EXPECT_DOUBLE_EQ(without.objective, still_off.objective);
}

/// x1 + x2 in [0, 1] each, but the single row demands x1 + x2 = 5: infeasible by construction
/// (the widest the row can reach is 2). With qp_ipm_detect_infeasibility off (its default),
/// the file header says this engine "runs to the iteration ceiling or a non-finite iterate"
/// on a model like this, exactly the genuine stall #981 is about - not a converging run cut
/// short, a run that CANNOT converge. Zero Hessian keeps the model trivially convex (the
/// convexity test never has to run the power iteration on anything).
Model infeasible_qp() {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {1.0, 1.0};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {5.0};
  model.row_upper = {5.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

// ---- (a) the stall test tells a genuine stall (a model that cannot converge) from --------
//      slow-but-real progress --------------------------------------------------------------
//
// With the option off, this is exactly the pre-#981 code path (the ceiling guard's condition
// is unchanged when stall_handoff is false) - the run either counts all the way to
// kIterationCeiling = 200 or, as happens on this model, its proximal parameters are driven to
// zero chasing a complementarity that can never be satisfied and the iterate stops being
// finite first. Confirming that outcome is unchanged is the regression test for (d): nothing
// about qp_ipm_stall_handoff's generalization runs when it is off.

TEST(QpIpmStallHandoff, OffAnInfeasibleModelRunsToNumericalErrorExactlyAsBefore) {
  const Model model = infeasible_qp();
  Logger logger(nullptr);
  Options options = quiet();
  options.set_bool("qp_ipm_stall_handoff", false);
  qp::QpIpmWarmResult result;
  const Solution s = qp::solve_convex_qp_ipm(model, options, logger, nullptr, nullptr, &result);
  EXPECT_EQ(s.status, SolveStatus::kNumericalError);
  EXPECT_NE(s.message.find("stopped being finite"), std::string::npos) << s.message;
  EXPECT_FALSE(result.stalled);
}

TEST(QpIpmStallHandoff, OnTheSameModelIsRecognizedAsAGenuineStallInstead) {
  const Model model = infeasible_qp();
  Logger logger(nullptr);
  Options options = quiet();
  options.set_bool("qp_ipm_stall_handoff", true);
  qp::QpIpmWarmResult result;
  const Solution s = qp::solve_convex_qp_ipm(model, options, logger, nullptr, nullptr, &result);
  EXPECT_EQ(s.status, SolveStatus::kIterationLimit);
  EXPECT_TRUE(result.stalled) << s.message;
  EXPECT_NE(s.message.find("stalled"), std::string::npos) << s.message;
  EXPECT_LT(s.iterations, 200);  // caught well before the old fixed ceiling
  // #981: the point to hand to the other engine is filled, unlike `next` (the
  // kQpIpmWarmSaveLevel save point), which a model that never gets close never reaches.
  ASSERT_FALSE(result.final_point.empty());
  EXPECT_EQ(result.final_point.col_value.size(), 2u);
}

// ---- (b) the IPM-to-first-order handoff: the plumbing from a genuine stall's point --------
//      into Condat-Vu, and - on a model that DOES have an optimum - the correct answer -------

TEST(QpIpmStallHandoff, TheStalledPointCanBeHandedToTheFirstOrderEngineWithoutCrashing) {
  // The model has no optimum, so this only exercises that the hand-off itself (sizes,
  // QpFirstOrderWarmStart's row_dual sign convention, op->upload) is sound on a real stalled
  // point; see the next test for a model where "correct answer" means something.
  const Model model = infeasible_qp();
  Logger logger(nullptr);
  Options options = quiet();
  options.set_bool("qp_ipm_stall_handoff", true);
  qp::QpIpmWarmResult result;
  const Solution stalled = qp::solve_convex_qp_ipm(model, options, logger, nullptr, nullptr,
                                                    &result);
  ASSERT_TRUE(result.stalled);
  ASSERT_FALSE(result.final_point.empty());

  qp::QpFirstOrderWarmStart warm{result.final_point.col_value, result.final_point.row_dual};
  const Solution handed = qp::solve_convex_qp(model, quiet(), logger, nullptr, &warm);
  // Neither engine can prove infeasibility here (qp_ipm_detect_infeasibility and its
  // Condat-Vu equivalent are untouched by #981); the only claim is that the warm-started run
  // produces a finite point of the box it was given, not kOptimal.
  EXPECT_NE(handed.status, SolveStatus::kOptimal);
  ASSERT_EQ(handed.col_value.size(), 2u);
  for (const double v : handed.col_value) {
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GE(v, 0.0);
    EXPECT_LE(v, 1.0);
  }
}

TEST(QpIpmStallHandoff, OnAModelThatDoesConvergeTheHandoffIsNeverTriggeredAndTheAnswerStands) {
  // #981's handoff is read back through QpIpmWarmResult::stalled, which is only ever true
  // together with kIterationLimit; an ordinary optimal solve must never look stalled.
  const Model model = inequality_qp();
  Logger logger(nullptr);
  Options options = quiet();
  options.set_bool("qp_ipm_stall_handoff", true);
  qp::QpIpmWarmResult result;
  const Solution s = qp::solve_convex_qp_ipm(model, options, logger, nullptr, nullptr, &result);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_FALSE(result.stalled);
}

TEST(QpIpmStallHandoff, AFirstOrderWarmStartOfTheWrongSizeIsIgnoredAndTheColdStartRuns) {
  const Model model = inequality_qp();
  Logger logger(nullptr);
  qp::QpFirstOrderWarmStart warm;
  warm.col_value = {1.0};  // one entry short of the model's two columns
  const Solution with_bad = qp::solve_convex_qp(model, quiet(), logger, nullptr, &warm);
  const Solution cold = qp::solve_convex_qp(model, quiet(), logger);
  ASSERT_EQ(with_bad.status, SolveStatus::kOptimal) << with_bad.message;
  EXPECT_EQ(with_bad.iterations, cold.iterations)
      << "a mismatched first-order warm start should fall back to the cold start exactly";
  EXPECT_NEAR(with_bad.objective, -4.5, 1e-6);
}

TEST(QpIpmStallHandoff, AFirstOrderWarmStartNearTheOptimumConvergesInFewerIterationsThanCold) {
  const Model model = inequality_qp();
  Logger logger(nullptr);
  const Solution cold = qp::solve_convex_qp(model, quiet(), logger);
  ASSERT_EQ(cold.status, SolveStatus::kOptimal) << cold.message;
  qp::QpFirstOrderWarmStart warm{cold.col_value, cold.row_dual};
  const Solution started = qp::solve_convex_qp(model, quiet(), logger, nullptr, &warm);
  ASSERT_EQ(started.status, SolveStatus::kOptimal) << started.message;
  EXPECT_LE(started.iterations, cold.iterations)
      << "starting at the optimum should never need MORE iterations than the cold start";
}

// ---- (c) the reverse handoff: Condat-Vu's point, stopped short by an explicit ------------
//      iteration_limit, finishes at the correct optimum through the interior point ---------

TEST(QpIpmStallHandoff, ACondatVuPointStoppedShortFinishesAtTheOptimumThroughTheInteriorPoint) {
  const Model model = inequality_qp();
  Logger logger(nullptr);
  Options short_run = quiet();
  short_run.set_int("iteration_limit", 3);  // nowhere near Condat-Vu's own tolerance
  const Solution first_order = qp::solve_convex_qp(model, short_run, logger);
  ASSERT_EQ(first_order.status, SolveStatus::kIterationLimit) << first_order.message;
  // Short of the optimum, or this test proves nothing about the handoff.
  ASSERT_GT(std::fabs(first_order.objective - (-4.5)), 1e-3);
  ASSERT_EQ(first_order.col_value.size(), 2u);
  ASSERT_EQ(first_order.row_dual.size(), 1u);

  qp::QpIpmWarmStart warm;
  warm.col_value = first_order.col_value;
  warm.row_dual = first_order.row_dual;
  warm.retry_cold = false;
  const Solution finished = qp::solve_convex_qp_ipm(model, quiet(), logger, nullptr, &warm);
  ASSERT_EQ(finished.status, SolveStatus::kOptimal) << finished.message;
  EXPECT_NEAR(finished.objective, -4.5, 1e-8);
  EXPECT_NEAR(finished.col_value[0], 0.5, 1e-7);
  EXPECT_NEAR(finished.col_value[1], 1.5, 1e-7);
}

}  // namespace
}  // namespace sankhya
