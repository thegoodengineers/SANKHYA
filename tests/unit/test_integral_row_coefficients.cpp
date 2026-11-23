// SPDX-License-Identifier: Apache-2.0
// SANKHYA - #1341: a row's bounds are rounded to integers only when its coefficients are
// integers, not when they are near ones.
//
// tighten_integral_rows (Chvatal 1973) rounds the bounds of a row over integer columns
// because the activity is then an integer at every integer point. It took a coefficient
// within 1e-6 of an integer for an integer. With 2 + 2^-21 on a column that reaches 400 the
// activity is 1.9e-4 away from an integer, the bound 1000.00019 was rounded up to 1001, and
// the point that meets the row with equality was cut off: the search proved 501 where the
// optimum is 500. Every number below is exact in a double.

#include <vector>

#include <gtest/gtest.h>

#include "mip/cuts.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kNear = 1.0 / 2097152.0;  // 2^-21 = 4.8e-7, inside the 1e-6 tolerance

/// min sum x:  first x0 + 2 (x1 + ...) >= lower,  every column integer in [0, 500].
Model covering_row(double first, double lower, Index cols) {
  Model model;
  model.resize_columns(cols);
  model.resize_rows(1);
  model.row_lower = {lower};
  model.row_upper = {kInfinity};
  model.matrix.reset(1, cols);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_cost[u] = 1.0;
    model.col_upper[u] = 500.0;
    model.col_type[u] = VarType::kInteger;
    model.matrix.add_entry(0, j, j == 0 ? first : 2.0);
  }
  model.matrix.finalize();
  return model;
}

TEST(IntegralRowCoefficients, ANearIntegerCoefficientDoesNotCutOffTheOptimum) {
  // (2 + 2^-21) x + 2 y >= 1000 + 400 * 2^-21. x + y = 500 needs x >= 400, and (400, 100)
  // meets the row with equality; x + y = 499 would need 2^-21 (x - 400) >= 2.
  for (const Index cols : {Index{2}, Index{3}}) {
    const Model model = covering_row(2.0 + kNear, 1000.0 + 400.0 * kNear, cols);
    ASSERT_EQ(model.validate(), "");
    for (const bool presolve : {true, false}) {
      for (const int threads : {0, 1}) {  // 0: one worker per core
        Options options;
        options.set_bool("log_to_console", false);
        options.set_bool("presolve", presolve);
        options.set_int("mip_threads", threads);
        const Solution solved = solve(model, options);
        ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
        EXPECT_NEAR(solved.objective, 500.0, 1e-6)
            << cols << " columns, presolve " << presolve << ", mip_threads " << threads;
      }
    }
  }
}

TEST(IntegralRowCoefficients, OnlyARowOfExactIntegersIsRounded) {
  Logger logger(nullptr);
  Model near = covering_row(2.0 + kNear, 1000.0 + 400.0 * kNear, 2);
  const mip::RowTightening none = mip::tighten_integral_rows(&near, logger);
  EXPECT_EQ(none.rows_tightened, 0);
  EXPECT_DOUBLE_EQ(near.row_lower[0], 1000.0 + 400.0 * kNear);

  // The control: with the coefficient exactly 2 the same bound is rounded up, rightly.
  Model exact = covering_row(2.0, 1000.0 + 400.0 * kNear, 2);
  const mip::RowTightening one = mip::tighten_integral_rows(&exact, logger);
  EXPECT_EQ(one.rows_tightened, 1);
  EXPECT_DOUBLE_EQ(exact.row_lower[0], 1001.0);
}

}  // namespace
}  // namespace sankhya
