// SPDX-License-Identifier: Apache-2.0
// SANKHYA - #1178: the QP interior point's reversible drop of the dual regularization
// (qp_ipm_adaptive_dual_drop) and the step rejection it rests on: a trial iterate that is not
// finite, or a drop whose step misses no less of the residual, never becomes the iterate.
//
// Every optimum below is worked out by hand, as test_qp_ipm.cpp's are.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"

namespace sankhya {
namespace {

Options drop_on() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("qp_ipm_adaptive_dual_regularization", true);
  options.set_bool("qp_ipm_adaptive_dual_drop", true);
  options.set_bool("qp_ipm_stall_handoff", false);
  return options;
}

/// min x1^2 + x2^2 - 2 x1 - 4 x2  s.t.  x1 + x2 <= 2 twice, x >= 0. The row is written
/// twice, so the row block of the KKT matrix is rank-deficient and the row duals are not
/// unique (any split of -1 between the two rows). Optimum (0.5, 1.5), objective -4.5, the
/// model of test_qp_ipm.cpp with its row duplicated.
Model duplicated_row_qp() {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {-2.0, -4.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {-kInfinity, -kInfinity};
  model.row_upper = {2.0, 2.0};
  model.matrix.reset(2, 2);
  for (Index r = 0; r < 2; ++r) {
    model.matrix.add_entry(r, 0, 1.0);
    model.matrix.add_entry(r, 1, 1.0);
  }
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 1, 2.0);
  model.hessian.finalize();
  return model;
}

/// min x1^2 + x2^2  s.t.  x1 + x2 = 1  and  x1 + x2 = 1 + gap, x >= 0: the two equality
/// rows disagree by `gap`, so b - M v keeps a component no step can close, and that is
/// exactly the miss the dual-regularization cut reads. With the drop on, each drop makes
/// |dy| grow as 1 / delta while the miss stays, which is the cascade qfffff80 showed.
Model inconsistent_rows_qp(double gap) {
  Model model;
  model.sense = ObjSense::kMinimize;
  model.col_cost = {0.0, 0.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.row_lower = {1.0, 1.0 + gap};
  model.row_upper = {1.0, 1.0 + gap};
  model.matrix.reset(2, 2);
  for (Index r = 0; r < 2; ++r) {
    model.matrix.add_entry(r, 0, 1.0);
    model.matrix.add_entry(r, 1, 1.0);
  }
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 1, 2.0);
  model.hessian.finalize();
  return model;
}

/// The engine with its log captured at verbose level.
std::string solve_logged(const Model& model, const Options& options, Solution* out) {
  std::FILE* stream = std::tmpfile();
  EXPECT_NE(stream, nullptr);
  if (stream == nullptr) return {};
  {
    Logger logger(stream, LogLevel::kVerbose);
    *out = qp::solve_convex_qp_ipm(model, options, logger);
  }
  std::fflush(stream);
  std::rewind(stream);
  std::string text;
  char buffer[4096];
  while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) text += buffer;
  std::fclose(stream);
  return text;
}

bool all_finite(const std::vector<double>& x) {
  for (double e : x) {
    if (!std::isfinite(e)) return false;
  }
  return true;
}

TEST(QpIpmDualDrop, DefaultsOff) {
  EXPECT_FALSE(Options().get_bool("qp_ipm_adaptive_dual_drop"));
}

TEST(QpIpmDualDrop, OnARankDeficientRowBlockTheDropStillReachesTheHandOptimum) {
  Solution with_drop;
  solve_logged(duplicated_row_qp(), drop_on(), &with_drop);
  ASSERT_EQ(with_drop.status, SolveStatus::kOptimal);
  EXPECT_NEAR(with_drop.objective, -4.5, 1e-6);
  ASSERT_EQ(with_drop.col_value.size(), 2u);
  EXPECT_NEAR(with_drop.col_value[0], 0.5, 1e-5);
  EXPECT_NEAR(with_drop.col_value[1], 1.5, 1e-5);
  EXPECT_TRUE(all_finite(with_drop.col_value));
  EXPECT_TRUE(all_finite(with_drop.row_dual));
}

TEST(QpIpmDualDrop, AnUnclosableResidualForcesADropCascadeAndTheRunEndsOnAFinitePoint) {
  Options options = drop_on();
  options.set_int("iteration_limit", 80);
  Solution s;
  const std::string log = solve_logged(inconsistent_rows_qp(1e-6), options, &s);
  // The drop happened and was undone by its own judgement, before any step of it was taken.
  EXPECT_NE(log.find("delta dropped from"), std::string::npos) << log;
  EXPECT_NE(log.find("undone at iteration"), std::string::npos) << log;
  // Whatever the run ends as, it is not a non-finite point and not a numerical failure: the
  // iterate that leaves the loop is the last finite one.
  EXPECT_NE(s.status, SolveStatus::kNumericalError) << log;
  EXPECT_NE(s.status, SolveStatus::kOptimal) << log;
  ASSERT_EQ(s.col_value.size(), 2u);
  EXPECT_TRUE(all_finite(s.col_value)) << log;
  EXPECT_TRUE(all_finite(s.row_dual)) << log;
  EXPECT_TRUE(std::isfinite(s.objective)) << log;
}

}  // namespace
}  // namespace sankhya
