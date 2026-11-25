// SPDX-License-Identifier: Apache-2.0
// SANKHYA - #1335: an integral row whose bounds hold no integer is an infeasible model, not
// a malformed one.
//
// tighten_integral_rows (Chvatal 1973) rounds the bounds of a row over integer columns with
// integral coefficients inward. When no integer lies between them the rounded bounds cross,
// which proves the model has no integer point. The crossed row used to go on to the search,
// whose Model::validate() refused it, and the solve ended model_error. The general-form
// oracle fuzz (#1334) drew the first model below; the exact answer is infeasible.

#include <vector>

#include <gtest/gtest.h>

#include "mip/cuts.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

/// One row over the given columns; every column integer unless `integral` says otherwise.
Model one_row(const std::vector<double>& coefficients, double row_lower, double row_upper,
              const std::vector<double>& col_lower, const std::vector<double>& col_upper,
              const std::vector<bool>& integral) {
  Model model;
  const auto cols = static_cast<Index>(coefficients.size());
  model.resize_columns(cols);
  model.resize_rows(1);
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.row_lower = {row_lower};
  model.row_upper = {row_upper};
  model.matrix.reset(1, cols);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (integral[u]) model.col_type[u] = VarType::kInteger;
    model.matrix.add_entry(0, j, coefficients[u]);
  }
  model.matrix.finalize();
  return model;
}

Options quiet_options() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

void expect_infeasible(const Model& model, const Options& options) {
  ASSERT_EQ(model.validate(), "");
  const Solution solved = solve(model, options);
  EXPECT_EQ(solved.status, SolveStatus::kInfeasible) << solved.message;
  // What "no point" looks like on a MILP (#505): the worst objective and bound of the sense.
  const double nothing = model.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;
  EXPECT_EQ(solved.objective, nothing) << solved.message;
  EXPECT_EQ(solved.dual_bound, nothing) << solved.message;
}

TEST(EmptyIntegralRow, TheFuzzModelIsInfeasibleNotAModelError) {
  // -6 x0 + 3/2 x1 - 6 x2 = -6 with x1 fixed at -1 and x0, x2 integer, at most 1. Presolve
  // folds x1 into the row, which leaves 6 (x0 + x2) = 4.5: bounds [-4.5, -4.5] round to
  // [-4, -5].
  const Model model = one_row({-6.0, 1.5, -6.0}, -6.0, -6.0, {-kInfinity, -1.0, -kInfinity},
                              {1.0, -1.0, 1.0}, {true, false, true});
  expect_infeasible(model, quiet_options());
  Options one_thread = quiet_options();
  one_thread.set_int("mip_threads", 1);
  expect_infeasible(model, one_thread);
}

TEST(EmptyIntegralRow, AnEqualityAtOneHalfOverTwoBinariesIsInfeasibleInEitherSense) {
  Model model = one_row({1.0, 1.0}, 0.5, 0.5, {0.0, 0.0}, {1.0, 1.0}, {true, true});
  model.col_cost = {1.0, 2.0};
  for (const bool presolve : {true, false}) {
    Options options = quiet_options();
    options.set_bool("presolve", presolve);
    model.sense = ObjSense::kMinimize;
    expect_infeasible(model, options);
    model.sense = ObjSense::kMaximize;
    expect_infeasible(model, options);
  }
}

TEST(EmptyIntegralRow, ARangeBetweenTwoIntegersIsInfeasible) {
  // 0.2 <= x - y <= 0.8 over two general integers.
  const Model model = one_row({1.0, -1.0}, 0.2, 0.8, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  expect_infeasible(model, quiet_options());
}

TEST(EmptyIntegralRow, TheRoundingNamesTheRowAndItsBoundsAsGiven) {
  Logger logger(nullptr);
  Model model = one_row({1.0, -1.0}, 0.2, 0.8, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  const mip::RowTightening empty = mip::tighten_integral_rows(&model, logger);
  EXPECT_EQ(empty.empty_row, 0);
  EXPECT_DOUBLE_EQ(empty.empty_lower, 0.2);
  EXPECT_DOUBLE_EQ(empty.empty_upper, 0.8);

  // A row with an integer in its bounds, and one on dust around an integer, are not empty.
  Model wide = one_row({1.0, -1.0}, 0.2, 1.8, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  EXPECT_EQ(mip::tighten_integral_rows(&wide, logger).empty_row, -1);
  Model dust = one_row({1.0, -1.0}, -2.7755575615628914e-17, -2.7755575615628914e-17,
                       {0.0, 0.0}, {5.0, 5.0}, {true, true});
  EXPECT_EQ(mip::tighten_integral_rows(&dust, logger).empty_row, -1);
}

TEST(EmptyIntegralRow, ARowThatArrivesCrossedIsNotCalledEmpty) {
  // #1350: x - y in [1, 1 - 9e-16], the equality x - y = 1 as a parallel-row merge wrote
  // it. The integer 1 is where both bounds are, to rounding; the rounding moves neither of
  // them, so the crossing is not its finding and proves nothing. The row stays as given,
  // for validate() to refuse.
  Logger logger(nullptr);
  const double below_one = 0.9999999999999991;
  Model crossed = one_row({1.0, -1.0}, 0.0, 2.0, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  crossed.row_lower[0] = 1.0;
  crossed.row_upper[0] = below_one;
  EXPECT_EQ(mip::tighten_integral_rows(&crossed, logger).empty_row, -1);
  EXPECT_EQ(crossed.row_lower[0], 1.0);
  EXPECT_EQ(crossed.row_upper[0], below_one);
  EXPECT_NE(crossed.validate(), "");
}

TEST(EmptyIntegralRow, ARowWithAnIntegerInItsBoundsStillSolves) {
  // The control: 0.2 <= x - y <= 1.8 is x - y = 1, and min x + y is then 1 at (1, 0).
  Model model = one_row({1.0, -1.0}, 0.2, 1.8, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  model.col_cost = {1.0, 1.0};
  const Solution solved = solve(model, quiet_options());
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, 1.0, 1e-9);
}

}  // namespace
}  // namespace sankhya
