// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact rational verification of a reported optimal basis (#521).
//
// Same discipline as test_certificate.cpp: the controlling cases here are NEGATIVE controls.
// A checker that has never rejected anything is not evidence that it can reject something,
// so ExactVerify.ARejectedBasisFailsNotVerifies hand-builds a basis that is not actually
// optimal and confirms verify_basis_exact() says so rather than rubber-stamping it.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "exact/exact_verify.hpp"
#include "exact/rational.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

Options quiet_exact() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("exact", true);
  return options;
}

}  // namespace

namespace exact {
namespace {

TEST(ExactRational, FromDoubleRoundTripsBitExactly) {
  // 0.1 is NOT exactly 1/10 in binary; from_double must recover the ACTUAL double, and
  // to_double() must hand back the identical bit pattern, not an approximation of 0.1.
  for (const double value : {0.1, 1.8, -3.5, 0.0, 1.0, -1.0, 1e10, -1e-6, 0.30000000000000004,
                             123456789.987654321}) {
    const Rational r = Rational::from_double(value);
    EXPECT_EQ(r.to_double(), value) << "value=" << value;
  }
}

TEST(ExactRational, ArithmeticIsExactWhereDoubleWouldRound) {
  // 0.1 + 0.2 != 0.3 in double arithmetic (a famous, deliberately chosen example); the exact
  // sum of the TWO DOUBLES 0.1 and 0.2 is also not exactly representable as a double, so it
  // must NOT equal from_double(0.3) - if it did, this class would be silently rounding.
  const Rational sum = Rational::from_double(0.1) + Rational::from_double(0.2);
  EXPECT_NE(sum, Rational::from_double(0.3));
  // But it IS exactly representable as the double 0.1+0.2 already computes (both take the
  // same rounding path), so to_double() must recover exactly that.
  EXPECT_EQ(sum.to_double(), 0.1 + 0.2);
}

TEST(ExactRational, OverflowThrowsRatherThanWraps) {
  // 1e300 already exceeds what __int128 can hold as an exact fraction (its magnitude alone
  // is far past 2^127), so from_double(1e300) throws immediately - correct, but not what
  // this test is for. 1e25 is comfortably representable (from_double must NOT throw here);
  // its SQUARE, 1e50, is what overflows __int128 multiplication.
  const Rational large = Rational::from_double(1e25);
  EXPECT_THROW(large * large, RationalOverflow);
}

TEST(ExactRational, FromDoubleItselfDeclinesAMagnitudeBeyondInt128) {
  EXPECT_THROW((void)Rational::from_double(1e300), RationalOverflow);
}

// A tiny LP with an exact, hand-checkable optimum: minimize x + y subject to x + 2y >= 4,
// 2x + y >= 4, x,y >= 0. The optimum is x=y=4/3, objective 8/3 - not an integer, so this
// also exercises a genuinely fractional exact result, not just integers dressed up as
// rationals.
Model tiny_lp() {
  return make_lp({{1.0, 2.0}, {2.0, 1.0}}, {4.0, 4.0}, {kInfinity, kInfinity}, {1.0, 1.0},
                 {0.0, 0.0}, {kInfinity, kInfinity});
}

TEST(ExactVerify, ARealOptimalBasisVerifiesWithTheExactAnswer) {
  const Model model = tiny_lp();
  const Solution solution = solve(model, quiet_exact());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal);
  ASSERT_EQ(solution.exact_status, Solution::ExactVerification::kVerified)
      << solution.exact_message;

  const Rational objective = Rational(8, 3);
  ASSERT_FALSE(solution.exact_objective.empty());
  const auto slash = solution.exact_objective.find('/');
  ASSERT_NE(slash, std::string::npos);
  const long long num = std::stoll(solution.exact_objective.substr(0, slash));
  const long long den = std::stoll(solution.exact_objective.substr(slash + 1));
  EXPECT_EQ(Rational(num, den), objective);

  // Every reported exact column value round-trips to something close to the double answer -
  // "close" because the double engine's own answer is only accurate to its own tolerances,
  // while the exact one is exact; they must agree to several more digits than the double
  // engine promises, not to the last bit.
  ASSERT_EQ(solution.exact_col_value.size(), solution.col_value.size());
  for (std::size_t j = 0; j < solution.col_value.size(); ++j) {
    const auto s = solution.exact_col_value[j].find('/');
    const long long n2 = std::stoll(solution.exact_col_value[j].substr(0, s));
    const long long d2 = std::stoll(solution.exact_col_value[j].substr(s + 1));
    EXPECT_NEAR(static_cast<double>(n2) / static_cast<double>(d2), solution.col_value[j], 1e-9);
  }
}

TEST(ExactVerify, ARejectedBasisFailsNotVerifies) {
  // Same model as above, but a WRONG basis handed in directly (bypassing solve() entirely,
  // which would never produce this on its own): claim x is basic and y is at its lower bound
  // 0, with row 0's slack basic and row 1's slack nonbasic at its lower bound 4 - a valid
  // BASIS shape (2 basic variables for 2 rows) but not the optimal, nor even feasible, point.
  const Model model = tiny_lp();
  Solution solution;
  solution.status = SolveStatus::kOptimal;
  solution.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  solution.row_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  solution.col_value = {4.0, 0.0};
  solution.row_activity = {4.0, 8.0};

  const ExactResult result = verify_basis_exact(model, solution);
  EXPECT_EQ(result.verdict, ExactVerdict::kFailed) << result.message;
}

TEST(ExactVerify, ASingularBasisIsFailedNotCrashed) {
  // Two parallel rows (row 1 = 2 * row 0): x and y's columns are then [1,2] and [1,2],
  // identical up to scale, so claiming BOTH structural columns basic (no row slack basic)
  // makes the "basis" matrix [[1,1],[2,2]] exactly singular.
  const Model model = make_lp({{1.0, 1.0}, {2.0, 2.0}}, {0.0, 0.0}, {4.0, 8.0}, {1.0, 1.0},
                              {0.0, 0.0}, {kInfinity, kInfinity});
  Solution solution;
  solution.status = SolveStatus::kOptimal;
  solution.col_status = {BasisStatus::kBasic, BasisStatus::kBasic};
  solution.row_status = {BasisStatus::kAtLower, BasisStatus::kAtLower};
  solution.col_value = {0.0, 0.0};
  solution.row_activity = {0.0, 0.0};

  const ExactResult result = verify_basis_exact(model, solution);
  EXPECT_EQ(result.verdict, ExactVerdict::kFailed) << result.message;
}

TEST(ExactVerify, AQuadraticObjectiveDeclines) {
  Model model = tiny_lp();
  model.hessian.reset(model.num_cols(), model.num_cols());
  model.hessian.add_entry(0, 0, 1.0);
  model.hessian.finalize();
  const Solution solution = solve(model, quiet_exact());
  // solve() itself refuses a non-convex-checked or unsupported path in various ways depending
  // on the QP engine's own scope; what matters here is only that exact verification, if it
  // runs at all, never claims kVerified on a quadratic model.
  EXPECT_NE(solution.exact_status, Solution::ExactVerification::kVerified);
}

TEST(ExactVerify, NoBasisIsDeclinedNotCrashed) {
  const Model model = tiny_lp();
  Solution solution;
  solution.status = SolveStatus::kOptimal;
  // col_status/row_status left empty, as PDHG or the interior point would leave them.
  const ExactResult result = verify_basis_exact(model, solution);
  EXPECT_EQ(result.verdict, ExactVerdict::kDeclined);
}

}  // namespace
}  // namespace exact
}  // namespace sankhya
