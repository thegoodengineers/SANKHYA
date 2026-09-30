// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact rational verification of a reported optimal basis (#521).
//
// Same discipline as test_certificate.cpp: the controlling cases here are NEGATIVE controls.
// A checker that has never rejected anything is not evidence that it can reject something,
// so ExactVerify.ARejectedBasisFailsNotVerifies hand-builds a basis that is not actually
// optimal and confirms verify_basis_exact() says so rather than rubber-stamping it.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "exact/bigint.hpp"
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

TEST(ExactRational, ArithmeticPastOneHundredTwentyEightBitsIsExact) {
  // Under __int128 this product overflowed and the modules declined; since #757 it is exact.
  const Rational large = Rational::from_double(1e25);
  const Rational square = large * large;
  EXPECT_EQ(square / large, large);
  EXPECT_EQ(square.to_double(), 1e25 * 1e25);
  EXPECT_EQ(Rational::from_double(1e300).to_double(), 1e300);
  EXPECT_THROW((void)Rational::from_double(kInfinity), RationalOverflow);
  EXPECT_THROW((void)(large / Rational(0)), RationalOverflow);
}

TEST(ExactBigInt, DivisionSatisfiesItsDefinitionOnRandomLargeNumbers) {
  // a = q b + r with |r| < |b| and r carrying the sign of a: the definition of truncating
  // division, checked on numbers of up to twelve limbs, where Algorithm D's correction
  // steps are exercised, and against the built-in arithmetic where it can hold the value.
  std::uint64_t state = 757;
  const auto next = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<long long>(state >> 33);
  };
  const auto random_big = [&](int limbs) {
    BigInt v(next() % 1000 - 500);
    for (int k = 0; k < limbs; ++k) v = v * BigInt(1LL << 31) + BigInt(next());
    return v;
  };
  for (int trial = 0; trial < 400; ++trial) {
    const BigInt a = random_big(trial % 12);
    BigInt b = random_big((trial * 7) % 6);
    if (b.is_zero()) b = BigInt(3);
    const BigInt q = a / b;
    const BigInt r = a % b;
    ASSERT_EQ(q * b + r, a) << "trial " << trial;
    ASSERT_LT(r.abs(), b.abs()) << "trial " << trial;
    ASSERT_TRUE(r.is_zero() || r.sign() == a.sign()) << "trial " << trial;
  }
  for (long long x : {0LL, 1LL, -1LL, 12345678901LL, -98765432109876LL}) {
    for (long long y : {1LL, -7LL, 4294967296LL, 99991LL}) {
      EXPECT_EQ(BigInt(x) / BigInt(y), BigInt(x / y));
      EXPECT_EQ(BigInt(x) % BigInt(y), BigInt(x % y));
    }
  }
  EXPECT_EQ((BigInt(1LL << 62) * BigInt(1LL << 62)).to_string(),
            "21267647932558653966460912964485513216");
  EXPECT_EQ(BigInt(-1000000000000LL).to_string(), "-1000000000000");
  EXPECT_EQ(BigInt::gcd(BigInt(-84), BigInt(36)), BigInt(12));
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
  // Called directly: through solve() a QP never reaches the exact check, so asserting on the
  // solve's status could not fail (review of #622). Given an LP's own optimal basis, the same
  // basis with a Hessian on the model must be declined, never verified.
  const Model lp = tiny_lp();
  const Solution solution = solve(lp, quiet_exact());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal);
  Model qp = lp;
  qp.hessian.reset(qp.num_cols(), qp.num_cols());
  qp.hessian.add_entry(0, 0, 1.0);
  qp.hessian.finalize();
  const ExactResult result = verify_basis_exact(qp, solution);
  EXPECT_EQ(result.verdict, ExactVerdict::kDeclined) << result.message;
}

TEST(ExactVerify, APrimalFeasibleButDualInfeasibleBasisFails) {
  // x basic, row 1's slack basic, row 0 at its lower bound: x = 4, y = 0, a vertex of the
  // feasible region with objective 4 against the optimum 8/3. The exact dual check must say so.
  const Model model = tiny_lp();
  Solution solution;
  solution.allocate_for(model);
  solution.status = SolveStatus::kOptimal;
  solution.col_value = {4.0, 0.0};
  solution.row_activity = {4.0, 8.0};
  solution.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  solution.row_status = {BasisStatus::kAtLower, BasisStatus::kBasic};
  solution.objective = 4.0;
  const ExactResult result = verify_basis_exact(model, solution);
  EXPECT_EQ(result.verdict, ExactVerdict::kFailed) << result.message;
}

TEST(ExactVerify, AColumnLabelledFixedWithUnequalBoundsIsNotTakenAsFixed) {
  // min -x1 + y s.t. y >= 1, x1 in [0, 10] in no row (review of #622). Postsolve labels a
  // removed column kFixed whatever its bounds; taken at its lower bound with no sign check,
  // x1 = 0 gave objective 1 and a VERIFIED stamp where the optimum is -9.
  const Model model =
      make_lp({{0.0, 1.0}}, {1.0}, {kInfinity}, {-1.0, 1.0}, {0.0, 0.0}, {10.0, kInfinity});
  Solution solution;
  solution.allocate_for(model);
  solution.status = SolveStatus::kOptimal;
  solution.col_value = {0.0, 1.0};
  solution.row_activity = {1.0};
  solution.col_status = {BasisStatus::kFixed, BasisStatus::kBasic};
  solution.row_status = {BasisStatus::kAtLower};
  solution.objective = 1.0;
  const ExactResult result = verify_basis_exact(model, solution);
  EXPECT_EQ(result.verdict, ExactVerdict::kFailed) << result.message;
}

TEST(ExactVerify, AFractionWiderThan64BitsIsWrittenInFull) {
  // 0.1 and 0.3 are not dyadic, so their exact values carry large power-of-two denominators;
  // min 0.1 x s.t. 0.3 x >= 0.1 has an exact objective whose denominator passes 2^64, and a
  // cast to long long printed its low 64 bits (review of #622).
  const Model model = make_lp({{0.3}}, {0.1}, {kInfinity}, {0.1}, {0.0}, {kInfinity});
  const Solution solution = solve(model, quiet_exact());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal);
  const ExactResult result = verify_basis_exact(model, solution);
  if (result.verdict == ExactVerdict::kVerified) {
    const std::string& text = result.exact_objective;
    const auto slash = text.find('/');
    ASSERT_NE(slash, std::string::npos) << text;
    // The denominator is written in decimal digits, more than 19 of them (past 2^64).
    EXPECT_GT(text.size() - slash - 1, 19U) << text;
  }
}

TEST(ExactVerify, PastItsTimeBudgetTheVerdictIsDeclinedNotAHang) {
  // brandy (220 rows) ran for over forty minutes in the exact modules before #757's budget.
  const Model model = tiny_lp();
  const Solution solution = solve(model, quiet_exact());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal);
  const ExactResult result = verify_basis_exact(model, solution, 1e-9);
  EXPECT_EQ(result.verdict, ExactVerdict::kDeclined);
  EXPECT_NE(result.message.find("exact_seconds"), std::string::npos) << result.message;
  EXPECT_EQ(verify_basis_exact(model, solution).verdict, ExactVerdict::kVerified);
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
