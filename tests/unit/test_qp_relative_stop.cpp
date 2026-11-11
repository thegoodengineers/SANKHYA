// SPDX-License-Identifier: Apache-2.0
// SANKHYA - qp_relative_stop: the QP first-order engine (Condat-Vu) may end its loop on the
// relative measures, and the verdict does not move - `optimal` still needs the absolute
// ones, so a point that met only the relative request is reported as stopped.
//
// The optimum below is worked out by hand.
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"

namespace sankhya {
namespace {

Options condat_vu(bool relative, double tolerance) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("qp_algorithm", "condat-vu");
  options.set_bool("qp_relative_stop", relative);
  options.set_double("qp_tolerance", tolerance);
  return options;
}

/// min (x1^2 + x2^2)/2 - 1e6 x1 - 2e6 x2  s.t.  x1 + x2 <= 1e6, x >= 0. The unconstrained
/// minimum (1e6, 2e6) violates the row, so it is active with multiplier l: x1 = 1e6 - l and
/// x2 = 2e6 - l sum to 1e6 at l = 1e6, giving x = (0, 1e6) with x1 on its bound at a zero
/// reduced cost; objective 0.5e12 - 2e12 = -1.5e12. The numbers are large on purpose: an
/// absolute tolerance of 1e-9 on activities of 1e6 is what the relative measure exists for.
Model large_scale_qp() {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {-1e6, -2e6};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {1e6};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.add_entry(1, 1, 1.0);
  model.hessian.finalize();
  return model;
}

TEST(QpRelativeStop, OffByDefault) {
  EXPECT_FALSE(Options().get_bool("qp_relative_stop"));
}

TEST(QpRelativeStop, ARelativeStopIsNeverReportedOptimalAboveTheAbsoluteTolerance) {
  // A tolerance of 1e-9 on activities of 1e6: the relative request is met long before the
  // absolute one, so the loop ends early and must not claim optimality.
  const Model model = large_scale_qp();
  Logger logger(nullptr, LogLevel::kError);
  Options options = condat_vu(true, 1e-9);
  options.set_int("iteration_limit", 200000);
  const Solution s = qp::solve_convex_qp(model, options, logger);
  if (s.status == SolveStatus::kOptimal) {
    // Only when the absolute measures were met as well: the point is optimal to them.
    EXPECT_LE(s.primal_infeasibility, 1e-9 * 10.0) << s.message;
  } else {
    EXPECT_EQ(s.status, SolveStatus::kIterationLimit) << s.message;
    EXPECT_NE(s.message.find("qp_relative_stop"), std::string::npos) << s.message;
    EXPECT_NE(s.message.find("not an optimality claim"), std::string::npos) << s.message;
  }
  ASSERT_EQ(s.col_value.size(), 2u);
  EXPECT_NEAR(s.col_value[0] + s.col_value[1], 1e6, 1e6 * 1e-6);
}

TEST(QpRelativeStop, OnASmallWellScaledQpTheAnswerIsTheSame) {
  // min x1^2 + x2^2 - 2 x1 - 4 x2  s.t.  x1 + x2 <= 2, x >= 0: optimum (0.5, 1.5), -4.5. At
  // this scale relative and absolute measures nearly coincide and both runs end optimal.
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
  Logger logger(nullptr, LogLevel::kError);
  const Solution off = qp::solve_convex_qp(model, condat_vu(false, 1e-8), logger);
  const Solution on = qp::solve_convex_qp(model, condat_vu(true, 1e-8), logger);
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(off.objective, -4.5, 1e-6);
  ASSERT_TRUE(on.status == SolveStatus::kOptimal || on.status == SolveStatus::kIterationLimit)
      << on.message;
  ASSERT_EQ(on.col_value.size(), 2u);
  EXPECT_NEAR(on.col_value[0], 0.5, 1e-5);
  EXPECT_NEAR(on.col_value[1], 1.5, 1e-5);
}

}  // namespace
}  // namespace sankhya
