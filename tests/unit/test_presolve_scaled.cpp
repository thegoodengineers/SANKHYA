// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve keeps every coefficient the model states (#792).
//
// A row and a column scaled by 2^-20 put a 2^-40 = 9.1e-13 coefficient into the matrix. It
// is data, and an exact change of variables away from 1.0. Presolve used to finalize its
// reduced model with the default absolute drop of 1e-11 and solved a different model (Netlib
// adlittle under the stress set of #762 lost five coefficients that way). These tests hold
// the reduced model to every coefficient, and the solve to the unscaled optimum.

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kScale = 0x1p-20;  // exact: only the exponent changes

/// min -x - y  s.t.  x + 2y <= 4,  3x + y <= 6,  x, y >= 0  (optimum -2.8 at (1.6, 1.2)),
/// with x replaced by kScale * x' and the first row multiplied by kScale.
Model scaled_lp() {
  Model model;
  model.col_cost = {-kScale, -1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInf, kInf};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {-kInf, -kInf};
  model.row_upper = {4.0 * kScale, 6.0};
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, kScale * kScale);  // 9.1e-13
  model.matrix.add_entry(0, 1, 2.0 * kScale);
  model.matrix.add_entry(1, 0, 3.0 * kScale);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.finalize(0.0);
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  return options;
}

TEST(PresolveScaled, TheReducedModelKeepsACoefficientBelowTheOldAbsoluteDrop) {
  const Model model = scaled_lp();
  ASSERT_EQ(model.matrix.num_nonzeros(), 4);
  Logger logger(nullptr);
  const presolve::Result reduced = presolve::presolve(model, quiet(), logger);
  EXPECT_EQ(reduced.model.matrix.num_nonzeros(), 4) << "a 9.1e-13 coefficient is data";
}

TEST(PresolveScaled, TheScaledModelSolvesToTheUnscaledOptimum) {
  const Solution solution = solve(scaled_lp(), quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -2.8, 1e-9);
  EXPECT_NEAR(solution.col_value[0] * kScale, 1.6, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 1.2, 1e-9);
}

}  // namespace
}  // namespace sankhya
