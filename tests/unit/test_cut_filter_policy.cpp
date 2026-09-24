// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cut filter's policy: the support floor and the efficacy test (#496).
//
// Both are off by default, and the default policy has to be the filter as it was: a cut
// over more than 0.2 n columns is too dense whatever the count, and a cut is taken on its
// absolute violation. With the floor, a cut small in the count passes on a small model
// while the same fraction on a large model still fails; with efficacy, the same geometric
// cut passes or fails independently of the scale its coefficients are written in.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "mip/cuts.hpp"
#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

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

}  // namespace
}  // namespace sankhya
