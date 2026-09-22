// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover cuts (#419).
//
// The obligation cuts.hpp sets for every family: a closed-form case worked by hand, an
// exact-arithmetic validity gate, and a negative control the harness must be able to fail.
//
// The shared rational MILP oracle (tests/oracles) settles PURE INTEGER programmes; a flow
// cover cut lives on a set with CONTINUOUS flow variables, so the gate here is a different
// exact method suited to that structure: for every 0/1 assignment of the row's binaries (a
// finite enumeration), the worst-case continuous point is the vertex of a one-constraint
// knapsack LP, which has a closed form (fill variables in decreasing benefit-per-budget order
// until the budget is exhausted) computed in exact rational arithmetic. A cut is valid exactly
// when that worst case never exceeds the right-hand side, for every binary assignment - which
// is precisely what the flow cover theorem promises. This is analogous to
// test_combinatorial_cuts.cpp's enumeration gate: no oracle is needed because the structure
// itself makes the exact worst case computable directly.

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mip/flow_cover_cuts.hpp"
#include "oracles/rational.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {
namespace {

using oracle::Rational;

constexpr double kInf = std::numeric_limits<double>::infinity();

/// A single-node flow set on `capacity[j]` items sharing one budget `b`, with row coefficient
/// `a[j]` on each flow variable (x_0..x_{k-1}, continuous, then y_0..y_{k-1}, binary).
Model flow_model(const std::vector<double>& a, const std::vector<double>& capacity, double b) {
  const std::size_t k = a.size();
  Model m;
  const std::size_t n = 2 * k;
  m.col_cost.assign(n, 0.0);
  m.col_lower.assign(n, 0.0);
  m.col_upper.assign(n, kInf);
  m.col_type.assign(n, VarType::kContinuous);
  for (std::size_t j = 0; j < k; ++j) {
    m.col_upper[k + j] = 1.0;
    m.col_type[k + j] = VarType::kInteger;
  }
  m.matrix.reset(static_cast<Index>(k + 1), static_cast<Index>(n));
  for (std::size_t j = 0; j < k; ++j) {
    m.matrix.add_entry(0, static_cast<Index>(j), a[j]);
  }
  for (std::size_t j = 0; j < k; ++j) {
    const Index row = static_cast<Index>(1 + j);
    m.matrix.add_entry(row, static_cast<Index>(j), 1.0);
    m.matrix.add_entry(row, static_cast<Index>(k + j), -capacity[j]);
  }
  m.matrix.finalize();
  m.row_lower.assign(k + 1, -kInf);
  m.row_upper.assign(k + 1, kInf);
  m.row_upper[0] = b;
  for (std::size_t j = 0; j < k; ++j) m.row_upper[1 + j] = 0.0;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

Solution at(const std::vector<double>& x, const std::vector<double>& y) {
  Solution s;
  s.col_value = x;
  s.col_value.insert(s.col_value.end(), y.begin(), y.end());
  return s;
}

/// The exact worst-case value of sum cut.coeff[x_j] * x_j, maximized over the continuous
/// feasible region { x >= 0 : sum a_j x_j <= b, x_j <= capacity[j] * y_j }, for a FIXED 0/1
/// `y`. A one-constraint knapsack LP: fill items in decreasing benefit-per-unit-budget order
/// (coeff[j] / a[j], skipping non-positive coefficients, which are optimally left at 0) until
/// the budget is spent.
Rational worst_case_flow_value(const std::vector<double>& a,
                               const std::vector<double>& capacity, double b,
                               const std::vector<int>& y, const std::vector<double>& x_coeff) {
  const std::size_t k = a.size();
  std::vector<std::size_t> order;
  for (std::size_t j = 0; j < k; ++j) {
    if (y[j] != 0 && x_coeff[j] > 0.0) order.push_back(j);
  }
  std::sort(order.begin(), order.end(), [&](std::size_t p, std::size_t q) {
    return x_coeff[p] / a[p] > x_coeff[q] / a[q];
  });
  Rational remaining_budget(static_cast<Rational::Int>(std::llround(b)));
  Rational total(0);
  for (const std::size_t j : order) {
    const Rational aj(static_cast<Rational::Int>(std::llround(a[j])));
    const Rational uj(static_cast<Rational::Int>(std::llround(capacity[j])));
    const Rational needed = aj * uj;
    if (remaining_budget.is_negative() || remaining_budget.is_zero()) break;
    const Rational cj(static_cast<Rational::Int>(std::llround(x_coeff[j])));
    if (needed > remaining_budget) {
      // Partial fill: x_j = remaining_budget / a_j.
      total = total + cj * (remaining_budget / aj);
      remaining_budget = Rational(0);
      break;
    }
    total = total + cj * uj;
    remaining_budget = remaining_budget - needed;
  }
  return total;
}

/// Assert `cut` is valid over the ENTIRE flow set, by enumerating every 0/1 assignment of the
/// binaries and checking the exact worst-case continuous point against `cut.rhs`.
::testing::AssertionResult cut_is_exactly_valid(const Cut& cut, const std::vector<double>& a,
                                                const std::vector<double>& capacity, double b) {
  const std::size_t k = a.size();
  std::vector<double> x_coeff(k), y_coeff(k);
  for (std::size_t j = 0; j < k; ++j) {
    x_coeff[j] = cut.coeff[j];
    y_coeff[j] = cut.coeff[k + j];
  }
  const Rational rhs(static_cast<Rational::Int>(std::llround(cut.rhs)));
  for (std::size_t mask = 0; mask < (std::size_t{1} << k); ++mask) {
    std::vector<int> y(k);
    Rational y_term(0);
    for (std::size_t j = 0; j < k; ++j) {
      y[j] = static_cast<int>((mask >> j) & 1);
      if (y[j] != 0)
        y_term = y_term + Rational(static_cast<Rational::Int>(std::llround(y_coeff[j])));
    }
    const Rational lhs = worst_case_flow_value(a, capacity, b, y, x_coeff) + y_term;
    if (rhs < lhs) {
      return ::testing::AssertionFailure()
             << "cut separates a feasible point at binary mask " << mask << ": lhs "
             << lhs.to_double() << " > rhs " << rhs.to_double();
    }
  }
  return ::testing::AssertionSuccess();
}

// ---- Closed-form case: the textbook fixed-charge example -----------------------------------

TEST(FlowCoverCuts, TextbookExampleGivesTheKnownInequality) {
  // sum x_j <= 10, x_j <= u_j y_j, u = (6, 5, 4, 3). Cover C = {0, 1, 2}: excess
  // lambda = 6+5+4-10 = 5, C+ = {0} (u_0 = 6 > 5). The flow cover inequality is
  //   x0 + x1 + x2 + (6 - 5)(1 - y0) <= 10  =>  x0 + x1 + x2 - y0 <= 9.
  const std::vector<double> a{1, 1, 1, 1};
  const std::vector<double> capacity{6, 5, 4, 3};
  const Model m = flow_model(a, capacity, 10.0);
  // An LP point where y0..y2 are near 1 (so the cover {0,1,2} is chosen) and y3 is off, with
  // x3 slightly SLACK inside its own variable upper bound (0 < 3 * 0.05) rather than tight at
  // it - a tight leftover bound is exactly what the partial lifting in flow_cover_cuts.cpp
  // folds in for free, which would add extra (still valid) terms this closed-form check does
  // not expect.
  const Solution s = at({5.9, 4.9, 3.9, 0.0}, {0.98, 0.97, 0.96, 0.05});
  const std::vector<Cut> cuts = generate_flow_cover_cuts(m, s, m.col_lower, m.col_upper);
  ASSERT_FALSE(cuts.empty());
  bool found = false;
  for (const Cut& cut : cuts) {
    EXPECT_TRUE(cut_is_exactly_valid(cut, a, capacity, 10.0));
    if (std::fabs(cut.coeff[0] - 1.0) < 1e-9 && std::fabs(cut.coeff[1] - 1.0) < 1e-9 &&
        std::fabs(cut.coeff[2] - 1.0) < 1e-9 && std::fabs(cut.coeff[3]) < 1e-9 &&
        std::fabs(cut.coeff[4] - (-1.0)) < 1e-9 && std::fabs(cut.coeff[5]) < 1e-9 &&
        std::fabs(cut.coeff[6]) < 1e-9 && std::fabs(cut.coeff[7]) < 1e-9 &&
        std::fabs(cut.rhs - 9.0) < 1e-9) {
      found = true;
    }
  }
  EXPECT_TRUE(found) << "expected x0 + x1 + x2 - y0 <= 9 among the generated cuts";
}

// ---- The harness can fail -------------------------------------------------------------------

TEST(FlowCoverCuts, TheCheckerRejectsADeliberatelyInvalidCut) {
  const std::vector<double> a{1, 1, 1, 1};
  const std::vector<double> capacity{6, 5, 4, 3};
  Cut invalid;
  invalid.coeff = {1, 1, 1, 0, -1, 0, 0, 0};
  invalid.rhs = 8.0;  // tightened by 1 from the true x0+x1+x2-y0<=9
  // At y=(1,1,1,0), the budget (10) binds before the combined capacity (6+5+4=15) does, so
  // the worst-case continuous point reaches x0+x1+x2=10 exactly; with y0=1 the lhs is 10-1=9,
  // which exceeds the tightened 8. The harness must find this counterexample on its own.
  EXPECT_FALSE(cut_is_exactly_valid(invalid, a, capacity, 10.0));
}

// ---- Validity, by enumeration over random fixed-charge models ------------------------------

TEST(FlowCoverCuts, NoCutSeparatesAnyFeasiblePointOfRandomFixedChargeModels) {
  std::mt19937 rng(419);
  std::uniform_int_distribution<int> capacity_dist(2, 9);
  std::uniform_int_distribution<int> coeff_dist(1, 2);
  std::uniform_real_distribution<double> frac(0.55, 0.99);
  int cuts_seen = 0;
  int models_with_cuts = 0;
  for (int trial = 0; trial < 1500; ++trial) {
    const std::size_t k = 2 + static_cast<std::size_t>(trial % 4);  // 2..5 flow items
    std::vector<double> a(k), capacity(k);
    double total_capacity = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
      a[j] = static_cast<double>(coeff_dist(rng));
      capacity[j] = static_cast<double>(capacity_dist(rng));
      total_capacity += a[j] * capacity[j];
    }
    if (total_capacity < 4.0) continue;
    const double b = std::floor(total_capacity * 0.6);
    if (b < 1.0) continue;
    const Model m = flow_model(a, capacity, b);

    std::vector<double> x(k), y(k);
    for (std::size_t j = 0; j < k; ++j) {
      y[j] = frac(rng);
      x[j] = y[j] * capacity[j] * frac(rng);
    }
    const Solution s = at(x, y);
    const std::vector<Cut> cuts = generate_flow_cover_cuts(m, s, m.col_lower, m.col_upper);
    if (!cuts.empty()) ++models_with_cuts;
    for (const Cut& cut : cuts) {
      ++cuts_seen;
      EXPECT_TRUE(cut_is_exactly_valid(cut, a, capacity, b))
          << "trial " << trial << ": a flow cover cut separates a feasible point";
    }
  }
  EXPECT_GT(cuts_seen, 50) << "the sweep should actually exercise the generator";
  EXPECT_GT(models_with_cuts, 20);
}

}  // namespace
}  // namespace sankhya::mip
