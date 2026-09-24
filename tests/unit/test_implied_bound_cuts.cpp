// SPDX-License-Identifier: Apache-2.0
// SANKHYA - implied-bound cuts (#499).
//
// Three obligations: the cut separates the point it is built for (a row the LP already
// satisfies, tightened by the column's own bound); it is valid at both values of the binary,
// checked exactly on small random rows; and a search with it on reaches the exact MILP
// optimum.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "mip/implied_bound_cuts.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

/// x continuous in [x_lo, x_hi], y binary, one row a_x x + a_y y in [row_lo, row_hi].
Model two_variable(double a_x, double a_y, double row_lo, double row_hi, double x_lo,
                   double x_hi) {
  Model m;
  m.resize_columns(2);
  m.col_lower = {x_lo, 0.0};
  m.col_upper = {x_hi, 1.0};
  m.col_type = {VarType::kContinuous, VarType::kInteger};
  m.resize_rows(1);
  m.row_lower = {row_lo};
  m.row_upper = {row_hi};
  m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, a_x);
  m.matrix.add_entry(0, 1, a_y);
  m.matrix.finalize();
  return m;
}

TEST(ImpliedBoundCuts, TheColumnBoundMakesItTighterThanTheRow) {
  // x + 2 y <= 4, x <= 3: y = 0 gives x <= 3 (the cap), y = 1 gives x <= 2. The cut
  // x + y <= 3 cuts off (3, 0.5), which satisfies the row (3 + 1 = 4).
  const Model m = two_variable(1.0, 2.0, -kInfinity, 4.0, 0.0, 3.0);
  const std::vector<Cut> cuts = implied_bound_cuts(m, {3.0, 0.5});
  ASSERT_EQ(cuts.size(), 1u);
  const Cut& c = cuts[0];
  EXPECT_EQ(c.family, CutFamily::kImpliedBound);
  EXPECT_NEAR(c.coeff[0], 1.0, 1e-12);
  EXPECT_NEAR(c.coeff[1], 1.0, 1e-6);
  EXPECT_NEAR(c.rhs, 3.0, 1e-6);
}

TEST(ImpliedBoundCuts, WithNoCapTheCutIsTheRowAndSeparatesNothing) {
  // x <= 10 caps neither case, so the cut is x + 2 y <= 4 itself, satisfied by (3, 0.5).
  const Model m = two_variable(1.0, 2.0, -kInfinity, 4.0, 0.0, 10.0);
  EXPECT_TRUE(implied_bound_cuts(m, {3.0, 0.5}).empty());
}

TEST(ImpliedBoundCuts, ALowerSideGivesALowerBoundCut) {
  // x - 4 y >= 0 (x >= 4 when y = 1), x >= 1: y = 0 gives x >= 1 (the cap), y = 1 gives
  // x >= 4. The cut x >= 1 + 3 y cuts off (2, 0.5), which satisfies the row (2 - 2 = 0).
  const Model m = two_variable(1.0, -4.0, 0.0, kInfinity, 1.0, 10.0);
  const std::vector<Cut> cuts = implied_bound_cuts(m, {2.0, 0.5});
  ASSERT_EQ(cuts.size(), 1u);
  // -x + 3 y <= -1
  EXPECT_NEAR(cuts[0].coeff[0], -1.0, 1e-12);
  EXPECT_NEAR(cuts[0].coeff[1], 3.0, 1e-6);
  EXPECT_NEAR(cuts[0].rhs, -1.0, 1e-6);
}

TEST(ImpliedBoundCuts, AnIntegralBinaryOrALongerRowGetsNoCut) {
  const Model m = two_variable(1.0, 2.0, -kInfinity, 4.0, 0.0, 3.0);
  EXPECT_TRUE(implied_bound_cuts(m, {3.0, 1.0}).empty());
  EXPECT_TRUE(implied_bound_cuts(m, {3.0, 0.0}).empty());
  Model three = two_variable(1.0, 2.0, -kInfinity, 4.0, 0.0, 3.0);
  three.resize_columns(3);
  three.matrix.reset(1, 3);
  three.matrix.add_entry(0, 0, 1.0);
  three.matrix.add_entry(0, 1, 2.0);
  three.matrix.add_entry(0, 2, 1.0);
  three.matrix.finalize();
  EXPECT_TRUE(implied_bound_cuts(three, {3.0, 0.5, 0.0}).empty());
}

TEST(ImpliedBoundCuts, EveryCutHoldsAtBothValuesOfTheBinary) {
  // Random integer rows and bounds. For y = 0 and y = 1 the feasible x form an interval,
  // computed exactly in rationals; the cut must hold at both of its ends.
  using oracle::Rational;
  std::mt19937_64 rng(4990);
  std::uniform_int_distribution<int> coef(-6, 6);
  std::uniform_int_distribution<int> bound(-8, 8);
  int checked = 0;
  for (int trial = 0; trial < 4000; ++trial) {
    const int a_x = coef(rng);
    const int a_y = coef(rng);
    if (a_x == 0 || a_y == 0) continue;
    int lo = bound(rng);
    int hi = bound(rng);
    if (lo > hi) std::swap(lo, hi);
    int x_lo = bound(rng);
    int x_hi = bound(rng);
    if (x_lo > x_hi) std::swap(x_lo, x_hi);
    const bool ranged = trial % 3 == 0;
    const double row_lo = ranged || trial % 3 == 1 ? lo : -kInfinity;
    const double row_hi = ranged || trial % 3 == 2 ? hi : kInfinity;
    const Model m = two_variable(a_x, a_y, row_lo, row_hi, x_lo, x_hi);
    // A fractional y and an x far outside: every candidate cut is violated there.
    for (const double x : {-50.0, 50.0}) {
      for (const Cut& cut : implied_bound_cuts(m, {x, 0.5})) {
        for (int y = 0; y <= 1; ++y) {
          // The exact feasible x-interval at this y: the column box and the row.
          Rational lo_x(x_lo);
          Rational hi_x(x_hi);
          const Rational rest = Rational(a_y) * Rational(y);
          if (std::isfinite(row_hi)) {
            const Rational edge = (Rational(hi) - rest) / Rational(a_x);
            if (a_x > 0) {
              hi_x = std::min(hi_x, edge);
            } else {
              lo_x = std::max(lo_x, edge);
            }
          }
          if (std::isfinite(row_lo)) {
            const Rational edge = (Rational(lo) - rest) / Rational(a_x);
            if (a_x > 0) {
              lo_x = std::max(lo_x, edge);
            } else {
              hi_x = std::min(hi_x, edge);
            }
          }
          if (lo_x > hi_x) continue;  // this value of y is infeasible: nothing to hold
          for (const Rational& xv : {lo_x, hi_x}) {
            const double activity = cut.coeff[0] * xv.to_double() + cut.coeff[1] * y;
            EXPECT_LE(activity, cut.rhs + 1e-9)
                << "row " << a_x << " x + " << a_y << " y in [" << row_lo << ", " << row_hi
                << "], x in [" << x_lo << ", " << x_hi << "], y = " << y;
          }
          ++checked;
        }
      }
    }
  }
  EXPECT_GT(checked, 500);
}

TEST(ImpliedBoundCuts, ASearchWithThemReachesTheExactOptimum) {
  // Fixed-charge pairs: x_k <= U_k continuous, y_k binary, x_k + a_k y_k <= b_k with U_k < b_k
  // (so the column bound caps the y = 0 case), and a demand row sum x_k + sum d_k y_k >= D.
  std::mt19937_64 rng(4991);
  std::uniform_int_distribution<std::int64_t> small(1, 6);
  int compared = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Index pairs = 2 + trial % 3;
    oracle::GeneratedLp lp;
    lp.num_cols = 2 * pairs;
    lp.num_rows = pairs + 1;
    lp.a.assign(static_cast<std::size_t>(lp.num_rows),
                std::vector<std::int64_t>(static_cast<std::size_t>(lp.num_cols), 0));
    lp.b.assign(static_cast<std::size_t>(lp.num_rows), 0);
    lp.c.assign(static_cast<std::size_t>(lp.num_cols), 0);
    lp.upper.assign(static_cast<std::size_t>(lp.num_cols), 0);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    std::int64_t demand = 0;
    const auto demand_row = static_cast<std::size_t>(pairs);
    for (Index k = 0; k < pairs; ++k) {
      const auto row = static_cast<std::size_t>(k);
      const auto x = static_cast<std::size_t>(2 * k);
      const auto y = static_cast<std::size_t>(2 * k + 1);
      const std::int64_t cap = small(rng) + 2;
      const std::int64_t b = cap + small(rng);
      const std::int64_t a = small(rng);
      lp.upper[x] = cap;
      lp.upper[y] = 1;
      lp.integral[y] = 1;
      // x + a y <= b, as -x - a y >= -b
      lp.a[row][x] = -1;
      lp.a[row][y] = -a;
      lp.b[row] = -b;
      lp.a[demand_row][x] = 1;
      lp.a[demand_row][y] = small(rng);
      lp.c[x] = small(rng);
      lp.c[y] = small(rng) * 3;
      demand += cap;
    }
    lp.b[demand_row] = demand / 2 + 1;
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal) continue;
    const Model model = oracle::to_model(lp);
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("presolve", false);
    options.set_bool("enable_root_cuts", true);
    options.set_bool("mip_implied_bound_cuts", true);
    const Solution got = solve(model, options);
    ASSERT_EQ(got.status, SolveStatus::kOptimal) << lp.to_text();
    EXPECT_NEAR(got.objective, exact.objective.to_double(), 1e-6) << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 40);
}

}  // namespace
}  // namespace sankhya::mip
