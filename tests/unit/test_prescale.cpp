// SPDX-License-Identifier: Apache-2.0
// SANKHYA - equilibration of the model by powers of two, and its exact inverse (#792).

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "core/prescale.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// min -x - y  s.t.  x + 2y <= 4,  3x + y <= 6,  x, y >= 0, with x and the first row
/// scaled by 2^-20 (the same model tests/unit/test_presolve_scaled.cpp uses).
Model badly_scaled() {
  const double s = 0x1p-20;
  Model model;
  model.col_cost = {-s, -1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInf, kInf};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {-kInf, -kInf};
  model.row_upper = {4.0 * s, 6.0};
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, s * s);
  model.matrix.add_entry(0, 1, 2.0 * s);
  model.matrix.add_entry(1, 0, 3.0 * s);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.finalize(0.0);
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

bool is_power_of_two(double v) {
  int exponent = 0;
  return v > 0.0 && std::frexp(v, &exponent) == 0.5;
}

TEST(Prescale, FactorsArePowersOfTwoAndTheMatrixIsEquilibrated) {
  const PrescaledModel p = prescale_by_powers_of_two(badly_scaled(), 20);
  for (const double r : p.row) EXPECT_TRUE(is_power_of_two(r)) << r;
  for (const double c : p.column) EXPECT_TRUE(is_power_of_two(c)) << c;
  double largest = 0.0;
  double smallest = kInf;
  for (const double v : p.model.matrix.values()) {
    largest = std::max(largest, std::fabs(v));
    smallest = std::min(smallest, std::fabs(v));
  }
  // The original spans 2^-40 .. 1; after equilibration everything is within a few powers
  // of two of 1.
  EXPECT_LE(largest / smallest, 64.0);
  EXPECT_EQ(p.model.matrix.num_nonzeros(), 4);
}

TEST(Prescale, ThePointAndTheDualsMapBackExactly) {
  const Model model = badly_scaled();
  const PrescaledModel p = prescale_by_powers_of_two(model, 20);
  // The optimum of the scaled model, in its own units: x' = 1.6 / (2^-20 * S_x), y' = 1.2 /
  // S_y.
  Solution s;
  s.status = SolveStatus::kOptimal;
  s.col_value = {1.6 / 0x1p-20 / p.column[0], 1.2 / p.column[1]};
  s.row_dual = {-0.2 / 0x1p-20 / p.row[0], -0.2 / p.row[1]};
  s.col_dual = {0.0, 0.0};
  s.row_activity = {4.0 * 0x1p-20 * p.row[0], 6.0 * p.row[1]};
  unscale_solution(p, &s);
  EXPECT_DOUBLE_EQ(s.col_value[0] * 0x1p-20, 1.6);
  EXPECT_DOUBLE_EQ(s.col_value[1], 1.2);
  EXPECT_DOUBLE_EQ(s.row_dual[0] * 0x1p-20, -0.2);
  EXPECT_DOUBLE_EQ(s.row_dual[1], -0.2);
  EXPECT_DOUBLE_EQ(s.row_activity[0], 4.0 * 0x1p-20);
  EXPECT_DOUBLE_EQ(s.row_activity[1], 6.0);
}

// #792, scaled modszk1 and pilot.we: Ruiz alone stops at a fixed point that keeps part of a
// diagonal rescaling. A 4-cycle of ones (rows {0,1}, {0,2}, {1,3}, {2,3}) with its rows and
// columns scaled by powers of two up to 2^20 comes out of 20 Ruiz rounds with entries spanning
// a factor of 1024; the geometric passes in front take it back to all ones.
TEST(Prescale, ADiagonalRescalingIsUndoneNotInherited) {
  const double rows[] = {0x1p-20, 0x1p13, 0x1p-7, 0x1p20};
  const double cols[] = {0x1p17, 0x1p-20, 0x1p5, 0x1p-11};
  Model model = badly_scaled();
  model.col_cost.assign(4, -1.0);
  model.col_lower.assign(4, 0.0);
  model.col_upper.assign(4, kInf);
  model.col_type.assign(4, VarType::kContinuous);
  model.row_lower.assign(4, -kInf);
  model.row_upper.assign(4, 1.0);
  model.matrix.reset(4, 4);
  const int pattern[4][2] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
  for (int j = 0; j < 4; ++j) {
    for (const int i : pattern[j]) model.matrix.add_entry(i, j, rows[i] * cols[j]);
  }
  model.matrix.finalize(0.0);
  model.hessian.reset(4, 4);
  model.hessian.finalize();
  const PrescaledModel p = prescale_by_powers_of_two(model, 20);
  for (const double v : p.model.matrix.values()) EXPECT_EQ(std::fabs(v), 1.0) << v;
}

TEST(Prescale, AnEquilibratedModelIsLeftAlone) {
  Model model = badly_scaled();
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.finalize(0.0);
  const PrescaledModel p = prescale_by_powers_of_two(model, 20);
  for (const double r : p.row) EXPECT_EQ(r, 1.0);
  for (const double c : p.column) EXPECT_EQ(c, 1.0);
}

// #792, scaled_e226: Netlib e226 has RHS -7.113 on its objective row, a constant of +7.113.
// The prescaled retry solved it to -11.6389 = Koch's -18.7519 + 7.113; the stress runner then
// subtracted the constant a second time and graded it wrong. The solver side is pinned here:
// the equilibrated model keeps the constant, and the answer measured back on the original
// model reports c'x plus it, as every other LP route does.
TEST(Prescale, TheObjectiveConstantSurvivesEveryLpRoute) {
  Model model = badly_scaled();
  model.objective_offset = 7.113;
  const PrescaledModel p = prescale_by_powers_of_two(model, 20);
  EXPECT_EQ(p.model.objective_offset, model.objective_offset);

  Options options;
  options.set_bool("log_to_console", false);
  Solution retry = solve(p.model, options);
  ASSERT_EQ(retry.status, SolveStatus::kOptimal);
  unscale_solution(p, &retry);
  retry.recompute_quality(model);
  EXPECT_NEAR(retry.objective, -2.8 + 7.113, 1e-9);

  for (const char* algorithm : {"auto", "simplex", "dual-simplex", "ipm"}) {
    for (const bool presolve : {true, false}) {
      Options route;
      route.set_bool("log_to_console", false);
      route.set_string("algorithm", algorithm);
      route.set_bool("presolve", presolve);
      const Solution s = solve(model, route);
      ASSERT_EQ(s.status, SolveStatus::kOptimal) << algorithm << " presolve " << presolve;
      double linear = 0.0;
      for (std::size_t j = 0; j < s.col_value.size(); ++j) {
        linear += model.col_cost[j] * s.col_value[j];
      }
      EXPECT_NEAR(s.objective, linear + 7.113, 1e-9) << algorithm << " presolve " << presolve;
      EXPECT_NEAR(s.objective, -2.8 + 7.113, 1e-6) << algorithm << " presolve " << presolve;
    }
  }
}

}  // namespace
}  // namespace sankhya
