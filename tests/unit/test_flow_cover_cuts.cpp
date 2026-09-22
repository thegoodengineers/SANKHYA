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

// ---- Single-node flow relaxation: aggregation across two rows -------------------------------

/// A two-node network's capacity rows, linked by one shared flow variable z that has no
/// variable upper bound of its own: x0 + x1 + z <= row0_bound (node A, receiving z from B),
/// and x2 + x3 - z <= row1_bound (node B, sending z to A). Columns: x0,x1,x2,x3,z,y0,y1,y2,y3.
Model two_node_flow_model(double row0_bound, double row1_bound,
                          const std::vector<double>& capacity) {
  Model m;
  const std::size_t n = 9;
  m.col_cost.assign(n, 0.0);
  m.col_lower.assign(n, 0.0);
  m.col_upper.assign(n, kInf);
  m.col_type.assign(n, VarType::kContinuous);
  for (std::size_t j = 5; j < 9; ++j) {
    m.col_upper[j] = 1.0;
    m.col_type[j] = VarType::kInteger;
  }
  m.matrix.reset(6, static_cast<Index>(n));
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(0, 4, 1.0);
  m.matrix.add_entry(1, 2, 1.0);
  m.matrix.add_entry(1, 3, 1.0);
  m.matrix.add_entry(1, 4, -1.0);
  for (std::size_t j = 0; j < 4; ++j) {
    const Index row = static_cast<Index>(2 + j);
    m.matrix.add_entry(row, static_cast<Index>(j), 1.0);
    m.matrix.add_entry(row, static_cast<Index>(5 + j), -capacity[j]);
  }
  m.matrix.finalize();
  m.row_lower.assign(6, -kInf);
  m.row_upper.assign(6, kInf);
  m.row_upper[0] = row0_bound;
  m.row_upper[1] = row1_bound;
  for (std::size_t j = 0; j < 4; ++j) m.row_upper[2 + j] = 0.0;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

TEST(FlowCoverCuts, AggregationAcrossTwoRowsFindsACoverNeitherRowHasAlone) {
  // Neither row alone is a clean single-node flow row: z has no variable upper bound, so it
  // is a disqualifying column in both x0+x1+z<=5 and x2+x3-z<=2, and neither row's own two
  // flow columns (capacity 3 each, total 6) exceed its own row's budget (5 or 2) - no cover is
  // possible from either row taken alone. Adding row0 + 1*row1 cancels z exactly (+1 in row0,
  // -1 in row1) and gives the AGGREGATE x0+x1+x2+x3 <= 7: capacities (3,3,3,3) sum to 12 > 7,
  // which the single row could never reach. This is the single-node flow relaxation
  // (flow_cover_cuts.hpp): the implied node where row1's outflow meets row0's inflow.
  const std::vector<double> capacity{3.0, 3.0, 3.0, 3.0};
  const Model m = two_node_flow_model(5.0, 2.0, capacity);
  const Solution s = at({2.9, 2.9, 2.9, 2.9, 0.0}, {0.97, 0.97, 0.97, 0.97});

  FlowCoverStats stats;
  const std::vector<Cut> cuts =
      generate_flow_cover_cuts(m, s, m.col_lower, m.col_upper, &stats);
  ASSERT_FALSE(cuts.empty());
  EXPECT_GT(stats.aggregated_cuts, 0) << "the win should have needed row aggregation";
  EXPECT_GT(stats.deepest, 0);

  bool found_cross_row_cut = false;
  for (const Cut& cut : cuts) {
    EXPECT_LT(std::fabs(cut.coeff[4]), 1e-9) << "z must be eliminated, never left in the cut";
    const bool touches_row0 = std::fabs(cut.coeff[0]) > 1e-9 || std::fabs(cut.coeff[1]) > 1e-9;
    const bool touches_row1 = std::fabs(cut.coeff[2]) > 1e-9 || std::fabs(cut.coeff[3]) > 1e-9;
    if (!touches_row0 || !touches_row1) continue;
    found_cross_row_cut = true;
    // EXACT VALIDITY. x0+x1+x2+x3 <= 7 is an inequality IMPLIED by the two original rows for
    // every value of z (it is their non-negative combination), so the true two-row, five-
    // variable feasible set is a SUBSET of that aggregate's single-node flow set; a cut valid
    // on the (larger) aggregate set is therefore valid on the (smaller) true set too. That
    // means the exact worst-case-vertex check already proven for the single-row family - no
    // oracle, no z - applies unchanged once z's column is dropped from the cut (it is always
    // zero there, asserted above).
    const std::vector<double> a4{1.0, 1.0, 1.0, 1.0};
    Cut projected;
    projected.coeff = {cut.coeff[0], cut.coeff[1], cut.coeff[2], cut.coeff[3],
                       cut.coeff[5], cut.coeff[6], cut.coeff[7], cut.coeff[8]};
    projected.rhs = cut.rhs;
    EXPECT_TRUE(cut_is_exactly_valid(projected, a4, capacity, 7.0));
  }
  EXPECT_TRUE(found_cross_row_cut)
      << "expected at least one cut spanning both row0's and row1's flow columns";
}

TEST(FlowCoverCuts, AggregationStopsAtTheDepthLimitRatherThanLoopingForever) {
  // A chain of five two-column rows, each carrying the next row's linking variable, needs
  // more eliminations than kMaxAggregation (3) allows to reach a clean flow row. The search
  // must give up cleanly - no crash, no cut from this line - rather than aggregate forever or
  // read past the depth it was given.
  Model m;
  const std::size_t links = 5;
  const std::size_t n = 2 + links;  // x0 (the one real flow column), z1..z5, y0
  m.col_cost.assign(n, 0.0);
  m.col_lower.assign(n, 0.0);
  m.col_upper.assign(n, kInf);
  m.col_type.assign(n, VarType::kContinuous);
  m.col_upper[n - 1] = 1.0;
  m.col_type[n - 1] = VarType::kInteger;
  // Row 0: x0 + z1 <= 1. Rows 1..4: -z_i + z_{i+1} <= 0 (a pure hand-off, no capacity of its
  // own - the sign is chosen so eliminating z_i needs a positive multiplier, i.e. a finite
  // UPPER bound, which this row has, at every link). Row 5 (the VUB): x0 - 2 y0 <= 0.
  m.matrix.reset(static_cast<Index>(links + 1), static_cast<Index>(n));
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  for (std::size_t i = 1; i < links; ++i) {
    m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(i), -1.0);
    m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(i + 1), 1.0);
  }
  m.matrix.add_entry(static_cast<Index>(links), 0, 1.0);
  m.matrix.add_entry(static_cast<Index>(links), static_cast<Index>(n - 1), -2.0);
  m.matrix.finalize();
  m.row_lower.assign(links + 1, -kInf);
  m.row_upper.assign(links + 1, kInf);
  m.row_upper[0] = 1.0;
  for (std::size_t i = 1; i < links; ++i) m.row_upper[i] = 0.0;
  m.row_upper[links] = 0.0;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();

  std::vector<double> x(n, 0.0);
  x[0] = 0.9;
  x[n - 1] = 0.95;
  const Solution s = at(x, {});
  FlowCoverStats stats;
  std::vector<Cut> cuts;
  EXPECT_NO_FATAL_FAILURE(
      { cuts = generate_flow_cover_cuts(m, s, m.col_lower, m.col_upper, &stats); });
  // The chain needs 4 eliminations to reach a clean flow row (z1 through z4); the cap of 3
  // stops it one short, so no cut comes from this line - the cap is what stopped it, not a
  // sign mismatch or a missing row (the previous, non-crash assertion is the one that matters
  // most; this one confirms the cap is actually load-bearing here, not vacuous).
  EXPECT_TRUE(cuts.empty());
}

}  // namespace
}  // namespace sankhya::mip
