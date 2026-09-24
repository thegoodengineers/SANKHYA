// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cut filter's policy: the support floor and the efficacy test (#496).
//
// Both are off by default, and the default policy has to be the filter as it was: a cut
// over more than 0.2 n columns is too dense whatever the count, and a cut is taken on its
// absolute violation. With the floor, a cut small in the count passes on a small model
// while the same fraction on a large model still fails; with efficacy, the same geometric
// cut passes or fails independently of the scale its coefficients are written in.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mip/cuts.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

Model padded(Index n) {
  Model model;
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  model.matrix.reset(0, n);
  model.matrix.finalize();
  return model;
}

Solution point(Index n, double value) {
  Solution s;
  s.col_value.assign(static_cast<std::size_t>(n), value);
  return s;
}

/// A cut over the first `support` columns, coefficient `scale` each, violated by the point
/// at `value` per column by `scale * support * value - rhs`.
mip::Cut dense_cut(Index n, Index support, double scale, double rhs) {
  mip::Cut c;
  c.coeff.assign(static_cast<std::size_t>(n), 0.0);
  for (Index j = 0; j < support; ++j) c.coeff[static_cast<std::size_t>(j)] = scale;
  c.rhs = rhs;
  return c;
}

TEST(CutFilterPolicy, TheDefaultPolicyIsTheFilterAsItWas) {
  // 40 of 100 columns is 0.4 > 0.2: too dense with the default policy, whatever the count.
  const Model m = padded(100);
  const auto res =
      mip::filter_and_deduplicate_cuts(m, point(100, 0.5), {dense_cut(100, 40, 1.0, 1.0)});
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(res[0].reason, mip::CutFilterReason::kTooDense);
}

TEST(CutFilterPolicy, TheFloorAdmitsASmallCountOnASmallModelAndNotTheSameFractionOnALargeOne) {
  mip::CutFilterPolicy policy;
  policy.support_floor = 100;
  // 40 nonzeros on 100 columns: 0.4 of n, but under the floor of 100: accepted.
  const Model small = padded(100);
  auto res = mip::filter_and_deduplicate_cuts(small, point(100, 0.5),
                                              {dense_cut(100, 40, 1.0, 1.0)}, policy);
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(res[0].reason, mip::CutFilterReason::kAccepted);
  // 400 nonzeros on 1,000 columns: the same fraction, over the floor: still too dense.
  const Model large = padded(1000);
  res = mip::filter_and_deduplicate_cuts(large, point(1000, 0.5),
                                         {dense_cut(1000, 400, 1.0, 1.0)}, policy);
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(res[0].reason, mip::CutFilterReason::kTooDense);
  // 150 nonzeros on 1,000 columns: 0.15 of n, accepted by the fraction as before.
  res = mip::filter_and_deduplicate_cuts(large, point(1000, 0.5),
                                         {dense_cut(1000, 150, 1.0, 1.0)}, policy);
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(res[0].reason, mip::CutFilterReason::kAccepted);
}

TEST(CutFilterPolicy, EfficacyIsScaleFreeWhereTheAbsoluteViolationIsNot) {
  // The cut x1 + x2 <= 0.9999 at the point (0.5, 0.5): violation 1e-4, efficacy
  // 1e-4 / sqrt(2) = 7.1e-5, under kCutMinEfficacy. Written as 1e4 x1 + 1e4 x2 <= 9999 it
  // is the same hyperplane: absolute violation 1.0 (passes the absolute test), efficacy
  // unchanged (still fails). The efficacy test does not care how the row is written.
  const Model m = padded(10);
  const Solution x = point(10, 0.5);
  const mip::Cut unit = dense_cut(10, 2, 1.0, 0.9999);
  const mip::Cut scaled = dense_cut(10, 2, 1e4, 9999.0);
  mip::CutFilterPolicy absolute;
  auto res = mip::filter_and_deduplicate_cuts(m, x, {unit, scaled}, absolute);
  ASSERT_EQ(res.size(), 2u);
  // 1e-4 > kCutViolationTolerance (1e-5): the unit form passes the absolute test too.
  EXPECT_EQ(res[0].reason, mip::CutFilterReason::kAccepted);
  EXPECT_EQ(res[1].reason, mip::CutFilterReason::kDuplicate)
      << "the scaled form is the same cut";
  mip::CutFilterPolicy efficacy;
  efficacy.efficacy = true;
  res = mip::filter_and_deduplicate_cuts(m, x, {unit, scaled}, efficacy);
  ASSERT_EQ(res.size(), 2u);
  EXPECT_EQ(res[0].reason, mip::CutFilterReason::kInsufficientViolation);
  EXPECT_EQ(res[1].reason, mip::CutFilterReason::kInsufficientViolation);
  // A cut the point is 0.2 / sqrt(2) = 0.14 away from passes either way.
  res = mip::filter_and_deduplicate_cuts(m, x, {dense_cut(10, 2, 1.0, 0.8)}, efficacy);
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(res[0].reason, mip::CutFilterReason::kAccepted);
}

TEST(CutFilterPolicy, ASearchWithBothOnStillReachesTheExactOptimum) {
  // The unit tests above say which cuts the policy lets through; this says the search that
  // takes them still gets the right answer. Binary knapsack and covering rows, 8 to 12
  // columns: 0.2 n is 1 or 2 nonzeros there, so the default filter refuses nearly every cut
  // and the floor admits them, which is where the policy changes what reaches the LP. Every
  // root family, tree rounds to depth 4, the floor at 100 and the efficacy test, each
  // answer compared with the exact rational branch and bound, and the same with both off.
  std::mt19937_64 rng(20260926);
  std::uniform_int_distribution<Index> width(8, 12);
  std::uniform_int_distribution<Index> height(2, 4);
  std::uniform_int_distribution<std::int64_t> weight(1, 9);
  std::uniform_int_distribution<std::int64_t> profit(1, 40);
  std::uniform_int_distribution<int> percent(0, 99);
  Options on;
  on.set_bool("log_to_console", false);
  for (const char* name :
       {"enable_root_cuts", "enable_mir_cuts", "enable_clique_cuts", "enable_zero_half_cuts"}) {
    on.set_bool(name, true);
  }
  on.set_int("tree_cut_depth", 4);
  Options off = on;
  on.set_int("cut_support_floor", 100);
  on.set_bool("cut_efficacy_test", true);
  int solved = 0;
  std::int64_t cuts_on = 0;
  std::int64_t cuts_off = 0;
  for (int trial = 0; trial < 120; ++trial) {
    oracle::GeneratedLp lp;
    lp.num_cols = width(rng);
    const Index knapsacks = height(rng);
    const Index covers = height(rng) - 1;
    lp.num_rows = knapsacks + covers;
    const auto n = static_cast<std::size_t>(lp.num_cols);
    lp.integral.assign(n, 1);
    lp.upper.assign(n, 1);
    lp.c.resize(n);
    for (std::size_t j = 0; j < n; ++j) lp.c[j] = -profit(rng);  // maximise profit
    for (Index i = 0; i < knapsacks; ++i) {
      std::vector<std::int64_t> row(n, 0);
      std::int64_t sum = 0;
      for (std::size_t j = 0; j < n; ++j) {
        if (percent(rng) < 70) row[j] = weight(rng);
        sum += row[j];
      }
      for (std::int64_t& a : row) a = -a;  // sum a x <= b, as -a x >= -b
      lp.a.push_back(row);
      lp.b.push_back(-std::max<std::int64_t>(1, sum / 2));
    }
    for (Index i = 0; i < covers; ++i) {
      std::vector<std::int64_t> row(n, 0);
      for (std::size_t j = 0; j < n; ++j) row[j] = percent(rng) < 30 ? 1 : 0;
      lp.a.push_back(row);
      lp.b.push_back(1);
    }
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal) continue;
    Model model = oracle::to_model(lp);
    for (std::size_t j = 0; j < n; ++j) {
      model.col_type[j] = VarType::kInteger;
      model.col_upper[j] = 1.0;
    }
    const Solution with = solve(model, on);
    const Solution without = solve(model, off);
    const double expected = exact.objective.to_double();
    const double tolerance = 1e-6 * std::max(1.0, std::fabs(expected));
    ASSERT_EQ(with.status, SolveStatus::kOptimal) << lp.to_text();
    EXPECT_NEAR(with.objective, expected, tolerance) << lp.to_text();
    ASSERT_EQ(without.status, SolveStatus::kOptimal) << lp.to_text();
    EXPECT_NEAR(without.objective, expected, tolerance) << lp.to_text();
    cuts_on += with.cuts_applied;
    cuts_off += without.cuts_applied;
    ++solved;
  }
  EXPECT_GE(solved, 80);
  EXPECT_GT(cuts_on, cuts_off) << "the policy changed nothing on these instances";
  std::printf(
      "[  INFO    ] cut filter policy: %d instances at the exact optimum; cut rows applied "
      "%lld with the floor and efficacy, %lld without\n",
      solved, static_cast<long long>(cuts_on), static_cast<long long>(cuts_off));
}

}  // namespace
}  // namespace sankhya
