// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GMI safety (#496 item 4; Cornuejols, Margot and Nannicini, "On the safety of
// Gomory cut generators", Math. Programming Computation 5, 2013).
//
// Two properties, each against the unsafe generator on the same basis: a source row whose
// basic value is within kGmiMinFractionality of an integer gives no cut, and every cut that
// is still emitted is the unsafe one with a LARGER right-hand side (the same coefficients,
// so it is weaker and cannot cut off a point the unsafe cut kept). Validity in a search is
// the debug-solution fuzz (test_debug_solution.cpp), which runs with gmi_safety on.

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "mip/cuts.hpp"
#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

/// x0 + 1.2 x1 + 2.1 x2 >= 0 with the row at its lower bound, x1 = 2 and x2 = 0 at their
/// lower bounds, so the basic x0 sits at -2.4 (f0 = 0.6): the hand case of test_cuts.cpp,
/// whose cut is -2.5 x0 - 10/3 x1 - 65/12 x2 <= -5/3.
struct Case {
  Model model;
  Solution solution;
};

Case fractional_case() {
  Case c;
  Model& m = c.model;
  m.resize_columns(3);
  m.resize_rows(1);
  m.matrix.reset(1, 3);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.2);
  m.matrix.add_entry(0, 2, 2.1);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 2.0, 0.0};
  m.col_upper = {kInfinity, kInfinity, kInfinity};
  m.row_lower = {0.0};
  m.row_upper = {kInfinity};
  Solution& s = c.solution;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower, BasisStatus::kAtLower};
  s.col_value = {-2.4, 2.0, 0.0};
  s.row_status = {BasisStatus::kAtLower};
  s.row_activity = {0.0};
  return c;
}

/// x0 + x1 >= 0 at its lower bound with the continuous x1 at its lower bound 2.005, so the
/// integer x0 is basic at -2.005: f0 = 0.995, within 0.01 of the integer above.
Case nearly_integral_case() {
  Case c;
  Model& m = c.model;
  m.resize_columns(2);
  m.resize_rows(1);
  m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kContinuous};
  m.col_lower = {-kInfinity, 2.005};
  m.col_upper = {kInfinity, 10.0};
  m.row_lower = {0.0};
  m.row_upper = {kInfinity};
  Solution& s = c.solution;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {-2.005, 2.005};
  s.row_status = {BasisStatus::kAtLower};
  s.row_activity = {0.0};
  return c;
}

TEST(GmiSafety, TheEmittedCutIsTheUnsafeOneWithARelaxedRhs) {
  const Case c = fractional_case();
  const std::vector<mip::Cut> unsafe = mip::generate_gmi_cuts(c.model, c.solution, false);
  const std::vector<mip::Cut> safe = mip::generate_gmi_cuts(c.model, c.solution, true);
  ASSERT_EQ(unsafe.size(), 1U);
  ASSERT_EQ(safe.size(), 1U);
  EXPECT_NEAR(unsafe[0].rhs, -5.0 / 3.0, 1e-12);
  ASSERT_EQ(safe[0].coeff.size(), unsafe[0].coeff.size());
  for (std::size_t j = 0; j < safe[0].coeff.size(); ++j) {
    EXPECT_EQ(safe[0].coeff[j], unsafe[0].coeff[j]) << "column " << j;
  }
  const double relax =
      tol::kGmiRhsRelaxAbsolute + tol::kGmiRhsRelaxRelative * std::fabs(unsafe[0].rhs);
  EXPECT_GT(safe[0].rhs, unsafe[0].rhs);
  EXPECT_NEAR(safe[0].rhs - unsafe[0].rhs, relax, 1e-15);
  // Still violated at the LP point, by the unsafe violation less the relaxation.
  double lhs = 0.0;
  for (std::size_t j = 0; j < safe[0].coeff.size(); ++j) {
    lhs += safe[0].coeff[j] * c.solution.col_value[j];
  }
  EXPECT_GT(lhs, safe[0].rhs + 1e-4);
}

TEST(GmiSafety, ANearlyIntegralSourceRowGivesNoCut) {
  const Case c = nearly_integral_case();
  const double f0 = c.solution.col_value[0] - std::floor(c.solution.col_value[0]);
  ASSERT_GT(f0, 1.0 - tol::kGmiMinFractionality);
  ASSERT_LT(f0, 1.0 - tol::kIntegrality);
  EXPECT_EQ(mip::generate_gmi_cuts(c.model, c.solution, false).size(), 1U)
      << "the unsafe generator should still cut here, or the test proves nothing";
  EXPECT_TRUE(mip::generate_gmi_cuts(c.model, c.solution, true).empty());
}

}  // namespace
}  // namespace sankhya
