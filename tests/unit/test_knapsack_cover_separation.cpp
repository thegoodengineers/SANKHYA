// SPDX-License-Identifier: Apache-2.0
// SANKHYA - knapsack cover separation at the LP point (#496).
//
// The cover chooser used to read the row alone, so its cut was valid and never violated:
// on the 30-instance MIPLIB set every cover candidate was refused for weak violation
// (bench/results/miplib-5daee10.csv). With the LP point the cover is chosen to be violated,
// the Crowder-Johnson-Padberg greedy. What can go wrong is that a separated cut stops being
// valid (it cannot: the base cover inequality is valid for any cover, and the lifting is
// unchanged), that a point which violates no cover still gets a cut, or that the cut
// returned is not violated by the point it was built for. These cases pin all three, and
// the row-only form the existing tests use is left as it was.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "mip/cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya {
namespace {

Model binary_knapsack(const std::vector<double>& a, double b) {
  const auto n = static_cast<Index>(a.size());
  Model m;
  m.col_cost.assign(static_cast<std::size_t>(n), 1.0);
  m.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  m.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  m.row_lower.push_back(-kInfinity);
  m.row_upper.push_back(b);
  m.matrix.reset(1, n);
  for (Index j = 0; j < n; ++j) m.matrix.add_entry(0, j, a[static_cast<std::size_t>(j)]);
  m.matrix.finalize();
  return m;
}

double lhs_minus_rhs(const mip::KnapsackCoverCut& cut, const std::vector<double>& x) {
  double lhs = 0.0;
  for (std::size_t k = 0; k < cut.col_index.size(); ++k) {
    lhs += cut.coeff[k] * x[static_cast<std::size_t>(cut.col_index[k])];
  }
  return lhs - cut.rhs;
}

TEST(KnapsackCoverSeparation, AFractionalPointThatViolatesACoverGetsACutItViolates) {
  // 5 x1 + 4 x2 + 3 x3 + 2 x4 <= 6 at the point (0.6, 0.6, 0.2, 0), which satisfies the row
  // (3.0 + 2.4 + 0.6 = 6.0). The cover {x1, x2} (5 + 4 > 6) gives x1 + x2 <= 1, which
  // 0.6 + 0.6 = 1.2 violates by 0.2; lifting adds x3 with coefficient 1 (no cover item
  // fits beside it in 6 - 3 = 3), so the cut is x1 + x2 + x3 <= 1, violated by 0.4.
  const Model m = binary_knapsack({5.0, 4.0, 3.0, 2.0}, 6.0);
  const std::vector<double> x{0.6, 0.6, 0.2, 0.0};
  const auto cut = mip::generate_knapsack_cover_cut(m, 0, &x);
  ASSERT_TRUE(cut.has_value()) << "the point violates the cover {x1, x2}; a cut is due";
  EXPECT_GT(lhs_minus_rhs(*cut, x), 1e-9) << "the returned cut is not violated by the point";
  // The row-only chooser picks the same largest-coefficient cover here, so the two forms
  // agree on this instance; the difference is on the next case.
}

TEST(KnapsackCoverSeparation, TheCoverFollowsThePointNotTheCoefficients) {
  // 6 x1 + 5 x2 + 4 x3 + 4 x4 <= 9. The row-only cover is {x1, x2} (6 + 5 > 9), x1 + x2 <= 1.
  // At (0, 0, 0.95, 0.95) no cover is violated: {x1, x2}, {x1, x3}, {x1, x4}, {x1, x3, x4}
  // and {x2, x3, x4} read 0, 0.95, 0.95, 1.9 and 1.9 against right-hand sides 1, 1, 1, 2, 2
  // ({x2, x3} is 9, not a cover). So no cut.
  const Model m = binary_knapsack({6.0, 5.0, 4.0, 4.0}, 9.0);
  const std::vector<double> quiet{0.0, 0.0, 0.95, 0.95};
  EXPECT_FALSE(mip::generate_knapsack_cover_cut(m, 0, &quiet).has_value())
      << "no cover is violated at this point, so no cut may be returned";
  // At (0.1, 0.7, 0.8, 0.8): the row-only cover {x1, x2} reads 0.8 <= 1, quiet; the cover
  // {x2, x3, x4} (5 + 4 + 4 = 13 > 9), x2 + x3 + x4 <= 2, reads 2.3: violated by 0.3.
  const std::vector<double> loud{0.1, 0.7, 0.8, 0.8};
  const auto cut = mip::generate_knapsack_cover_cut(m, 0, &loud);
  ASSERT_TRUE(cut.has_value()) << "the point violates the cover {x2, x3, x4}";
  EXPECT_GT(lhs_minus_rhs(*cut, loud), 1e-9);
  // and the row-only form, which knows no point, still hands back its unviolated cut.
  const auto row_only = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(row_only.has_value());
  EXPECT_LE(lhs_minus_rhs(*row_only, loud), 1e-9);
}

TEST(KnapsackCoverSeparation, AnIntegerFeasiblePointGetsNoCut) {
  const Model m = binary_knapsack({5.0, 4.0, 3.0, 2.0}, 6.0);
  for (unsigned mask = 0; mask < 16; ++mask) {
    std::vector<double> x(4, 0.0);
    double weight = 0.0;
    const double a[] = {5.0, 4.0, 3.0, 2.0};
    for (unsigned j = 0; j < 4; ++j) {
      if (mask & (1u << j)) {
        x[j] = 1.0;
        weight += a[j];
      }
    }
    if (weight > 6.0) continue;  // not a feasible point of the row
    EXPECT_FALSE(mip::generate_knapsack_cover_cut(m, 0, &x).has_value())
        << "mask " << mask << ": a feasible binary point violates no valid cut";
  }
}

TEST(KnapsackCoverSeparation, ASeparatedCutIsValidAtEveryFeasibleBinaryPoint) {
  const std::vector<double> a{7.0, 6.0, 5.0, 4.0, 3.0, 2.0};
  const double b = 10.0;
  const Model m = binary_knapsack(a, b);
  const std::vector<double> points[] = {{0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
                                        {0.9, 0.1, 0.9, 0.1, 0.9, 0.1},
                                        {0.3, 0.8, 0.8, 0.3, 0.0, 1.0}};
  int cuts = 0;
  for (const auto& x : points) {
    const auto cut = mip::generate_knapsack_cover_cut(m, 0, &x);
    if (!cut.has_value()) continue;
    ++cuts;
    EXPECT_GT(lhs_minus_rhs(*cut, x), 1e-9);
    for (unsigned mask = 0; mask < 64; ++mask) {
      std::vector<double> z(6, 0.0);
      double weight = 0.0;
      for (unsigned j = 0; j < 6; ++j) {
        if (mask & (1u << j)) {
          z[j] = 1.0;
          weight += a[j];
        }
      }
      if (weight > b) continue;
      EXPECT_LE(lhs_minus_rhs(*cut, z), 1e-12) << "cut violated at feasible point " << mask;
    }
  }
  EXPECT_GE(cuts, 1) << "at least one of the three points violates a cover";
}

}  // namespace
}  // namespace sankhya
