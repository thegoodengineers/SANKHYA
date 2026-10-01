// SPDX-License-Identifier: Apache-2.0
// SANKHYA - equilibration of the model by powers of two, and its exact inverse (#792).

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "core/prescale.hpp"
#include "sankhya/model.hpp"

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

}  // namespace
}  // namespace sankhya
