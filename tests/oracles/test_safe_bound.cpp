// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the Neumaier-Shcherbina safe bound (#519) against the exact oracle.
//
// The claim under test is one-sided and absolute: for ANY multiplier vector, the safe bound
// never exceeds the exact optimum of the LP. So it is tested with duals that are good (the
// simplex's), slightly wrong (perturbed the way an inexact solve perturbs them), chosen to
// make the plain bound lie, and random, and every comparison with the optimum is made in
// exact rational arithmetic.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "core/safe_bound.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {
namespace {

/// A rational at least `v`: exact for every double of magnitude at least 2^-60, and the
/// next larger simple value below that, which keeps "bound <= optimum" a conservative test.
Rational rational_at_least(double v) {
  constexpr int kMinExponent = -60;
  if (v == 0.0) return Rational(0);
  if (std::fabs(v) < std::ldexp(1.0, kMinExponent)) {
    return v < 0.0 ? Rational(0) : Rational(1, static_cast<Rational::Int>(1) << 60);
  }
  int exponent = 0;
  const double mantissa = std::frexp(v, &exponent);  // v = mantissa * 2^exponent
  const auto scaled = static_cast<std::int64_t>(std::ldexp(mantissa, 53));
  const int shift = exponent - 53;
  if (shift >= 0) {
    return Rational(static_cast<Rational::Int>(scaled) *
                    (static_cast<Rational::Int>(1) << shift));
  }
  return Rational(static_cast<Rational::Int>(scaled), static_cast<Rational::Int>(1)
                                                          << (-shift));
}

Solution solve_float(const Model& model) {
  Options options;
  options.set_bool("log_to_console", false);
  return solve(model, options);
}

/// The naive bound a solver would report from y: the dual objective b.y, in floating point,
/// with the reduced costs assumed dual feasible. For A x >= b, x >= 0 with finite upper
/// bounds it is sum b_i y_i + sum_j min(r_j, 0) u_j - and the plain one drops the second
/// term, which is exactly the mistake an almost-feasible dual invites.
double plain_bound(const Model& model, const std::vector<double>& y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) sum += model.row_lower[i] * y[i];
  return sum;
}

TEST(SafeBound, AdversarialNearOptimalDualOverstatesThePlainBoundButNotTheSafeOne) {
  // min x1 + x2 + 3 x3  s.t.  x1 + x2 + x3 >= 4,  x1 - x2 >= -2,  x2 + 2 x3 >= 1,
  //                           0 <= x <= 5.
  GeneratedLp lp;
  lp.num_rows = 3;
  lp.num_cols = 3;
  lp.a = {{1, 1, 1}, {1, -1, 0}, {0, 1, 2}};
  lp.b = {4, -2, 1};
  lp.c = {1, 1, 3};
  lp.upper = {5, 5, 5};
  const OracleResult exact = solve_exact(lp);
  ASSERT_EQ(exact.status, OracleStatus::kOptimal);

  const Model model = to_model(lp);
  const Solution solved = solve_float(model);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal);
  ASSERT_EQ(solved.row_dual.size(), 3U);

  // The simplex's own dual gives a safe bound within rounding of the optimum.
  const SafeBound good = safe_dual_bound(model, solved.row_dual);
  ASSERT_TRUE(std::isfinite(good.value));
  EXPECT_LE(rational_at_least(good.value), exact.objective);
  EXPECT_NEAR(good.value, exact.objective.to_double(), 1e-9);

  // The adversary: raise the multiplier of the first row (b_1 = 4 > 0) by 1e-6. The dual is
  // now infeasible by 1e-6 on every column of that row, and its plain objective b.y claims
  // 4e-6 more than the LP can deliver.
  std::vector<double> y = solved.row_dual;
  y[0] += 1e-6;
  const double plain = plain_bound(model, y);
  EXPECT_GT(rational_at_least(plain), exact.objective)
      << "the plain bound " << plain << " should overstate the optimum "
      << exact.objective.to_double();
  const SafeBound safe = safe_dual_bound(model, y);
  ASSERT_TRUE(std::isfinite(safe.value));
  EXPECT_LE(rational_at_least(safe.value), exact.objective)
      << "safe " << safe.value << " optimum " << exact.objective.to_double();
  // Valid AND useful: it gives back only what the dual infeasibility costs over the box.
  EXPECT_GT(safe.value, exact.objective.to_double() - 1e-4);
}

TEST(SafeBound, RoundToNearestAloneWouldOverstateByAnUlp) {
  // min 0.1 x  s.t.  x >= 3,  0 <= x <= 10, with 0.1 the double nearest to one tenth. The
  // exact optimum is 3 * 0.1(double) = 0.3000000000000000166..., and y = 0.1 is an exactly
  // optimal dual (its reduced cost is exactly zero). Yet b.y rounded to nearest is
  // 0.30000000000000004, ABOVE the optimum: rounding alone makes the plain bound lie, so the
  // outward rounding is not decoration.
  Model model;
  model.resize_columns(1);
  model.col_cost = {0.1};
  model.col_upper = {10.0};
  model.resize_rows(1);
  model.row_lower = {3.0};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();
  const Rational optimum = rational_at_least(0.1) * Rational(3);  // exact: 0.1 is dyadic
  const std::vector<double> y = {0.1};
  const double plain = 3.0 * y[0];
  EXPECT_GT(rational_at_least(plain), optimum);
  const SafeBound safe = safe_dual_bound(model, y);
  ASSERT_TRUE(std::isfinite(safe.value));
  EXPECT_LE(rational_at_least(safe.value), optimum);
  EXPECT_EQ(safe.value, std::nextafter(plain, 0.0));
}

TEST(SafeBound, FreeColumnWithNonzeroReducedCostNeedsAnImpliedBound) {
  // min x1  s.t.  x1 - x2 >= 0,  x2 >= 1 (as a row),  x1 free above, x2 in [0, inf).
  Model model;
  model.resize_columns(2);
  model.col_cost = {1.0, 0.0};
  model.resize_rows(2);
  model.row_lower = {0.0, 1.0};
  model.row_upper = {kInfinity, kInfinity};
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.finalize();
  // y = (1 + 1e-9, 1): r_1 = -1e-9 needs an upper bound on x1, which no row implies (x1 is
  // only bounded below by the first row), so y itself proves nothing. x1's cost is
  // positive, so the retry with (1 - e) y does (see the next test), and what it proves is
  // still below the optimum, 1.
  const std::vector<double> y = {1.0 + 1e-9, 1.0};
  const SafeBound rescued = safe_dual_bound(model, y, false);
  ASSERT_TRUE(std::isfinite(rescued.value));
  EXPECT_GT(rescued.shrink, 0.0);
  EXPECT_LE(rescued.value, 1.0);
  // y = (1 - 1e-9, 1): r_1 = +1e-9 needs a LOWER bound on x1 (it has 0), and r_2 =
  // -1 + (1 - 1e-9) ... = -1e-9 needs an upper bound on x2, which again nothing implies.
  // Give x2 a row upper bound instead: x2 <= 7 through row 1's upper side.
  model.row_upper[1] = 7.0;
  const std::vector<double> y2 = {1.0 - 1e-9, 1.0};
  const SafeBound implied = safe_dual_bound(model, y2, true);
  ASSERT_TRUE(std::isfinite(implied.value));
  EXPECT_EQ(implied.implied_bounds, 1);
  EXPECT_LE(implied.value, 1.0);
  EXPECT_GT(implied.value, 1.0 - 1e-7);
  // A multiplier that prices a side the row does not have is dropped, not used.
  const std::vector<double> y3 = {-1.0, 1.0};  // row 0 has no upper side
  const SafeBound dropped = safe_dual_bound(model, y3, true);
  EXPECT_EQ(dropped.dropped_multipliers, 1);
}

TEST(SafeBound, ABasicColumnWithNoUpperBoundIsRescuedByScalingY) {
  // min x1 + x2  s.t.  x1 + x2 >= 1,  x >= 0 with no upper bounds and no row that implies
  // one. y = 1 + 1e-12 leaves both reduced costs at -1e-12: y itself proves nothing, but
  // (1 - e) y for a tiny e does, because both costs are positive.
  Model model;
  model.resize_columns(2);
  model.col_cost = {1.0, 1.0};
  model.resize_rows(1);
  model.row_lower = {1.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  const SafeBound bound = safe_dual_bound(model, {1.0 + 1e-12});
  ASSERT_TRUE(std::isfinite(bound.value));
  EXPECT_GT(bound.shrink, 0.0);
  EXPECT_LE(bound.shrink, 1e-6);
  EXPECT_LE(bound.value, 1.0);
  EXPECT_GT(bound.value, 1.0 - 1e-10);
  // A zero cost cannot be rescued that way: -inf, and no shrink reported.
  model.col_cost = {1.0, 0.0};
  const SafeBound none = safe_dual_bound(model, {1e-12});
  EXPECT_FALSE(std::isfinite(none.value));
  EXPECT_EQ(none.shrink, 0.0);
}

struct Tally {
  int optimal = 0;
  int finite_from_simplex = 0;
  int infeasible_proved = 0;
  int infeasible = 0;
  int skipped = 0;
  double worst_gap = 0.0;  ///< optimum - safe, over finite bounds from the simplex duals
};

/// Every y we can think of, against the exact optimum. Returns false on a violation.
bool check_instance(const GeneratedLp& lp, std::mt19937_64& rng, Tally* tally,
                    std::string* failure) {
  const OracleResult exact = solve_exact(lp);
  if (exact.status == OracleStatus::kOverflow ||
      exact.status == OracleStatus::kIterationLimit) {
    ++tally->skipped;
    return true;
  }
  const Model model = to_model(lp);
  const Solution solved = solve_float(model);
  std::normal_distribution<double> noise(0.0, 1.0);
  try {
    if (exact.status == OracleStatus::kOptimal) {
      ++tally->optimal;
      std::vector<std::vector<double>> candidates;
      if (solved.row_dual.size() == static_cast<std::size_t>(lp.num_rows)) {
        candidates.push_back(solved.row_dual);
        for (const double scale : {1e-9, 1e-6, 1e-3}) {
          std::vector<double> y = solved.row_dual;
          for (double& v : y) v += scale * noise(rng);
          candidates.push_back(std::move(y));
        }
      }
      std::vector<double> random(static_cast<std::size_t>(lp.num_rows));
      for (double& v : random) v = noise(rng);
      candidates.push_back(std::move(random));
      for (std::size_t k = 0; k < candidates.size(); ++k) {
        const SafeBound bound = safe_dual_bound(model, candidates[k]);
        if (!std::isfinite(bound.value)) continue;
        if (rational_at_least(bound.value) > exact.objective) {
          *failure = "safe bound " + std::to_string(bound.value) + " > exact optimum " +
                     std::to_string(exact.objective.to_double()) + " (candidate " +
                     std::to_string(k) + ")\n" + lp.to_text();
          return false;
        }
        if (k == 0) {
          ++tally->finite_from_simplex;
          tally->worst_gap =
              std::max(tally->worst_gap, exact.objective.to_double() - bound.value);
        }
      }
    } else if (exact.status == OracleStatus::kInfeasible) {
      ++tally->infeasible;
      // A zero objective turns the bound into a Farkas test: > 0 proves infeasibility.
      if (solved.farkas_dual.size() == static_cast<std::size_t>(lp.num_rows)) {
        SafeBoundProblem problem;
        problem.matrix = &model.matrix;
        problem.row_lower = model.row_lower;
        problem.row_upper = model.row_upper;
        problem.col_lower = model.col_lower;
        problem.col_upper = model.col_upper;
        std::vector<double> negated = solved.farkas_dual;
        for (double& v : negated) v = -v;
        if (safe_dual_bound(problem, solved.farkas_dual).value > 0.0 ||
            safe_dual_bound(problem, negated).value > 0.0) {
          ++tally->infeasible_proved;
        }
      }
    }
    // On a FEASIBLE model no y may prove infeasibility: min 0 over a nonempty set is 0.
    if (exact.status != OracleStatus::kInfeasible) {
      SafeBoundProblem problem;
      problem.matrix = &model.matrix;
      problem.row_lower = model.row_lower;
      problem.row_upper = model.row_upper;
      problem.col_lower = model.col_lower;
      problem.col_upper = model.col_upper;
      std::vector<double> random(static_cast<std::size_t>(lp.num_rows));
      for (double& v : random) v = noise(rng);
      const double value = safe_dual_bound(problem, random).value;
      if (value > 0.0) {
        *failure = "a random y proved a feasible model infeasible (" + std::to_string(value) +
                   ")\n" + lp.to_text();
        return false;
      }
    }
  } catch (const RationalOverflow&) {
    ++tally->skipped;
  }
  return true;
}

TEST(SafeBound, NeverExceedsTheExactOptimumOnRandomLps) {
  std::mt19937_64 rng(519);
  Tally tally;
  std::vector<std::string> failures;
  for (const double bounded : {1.0, 0.35}) {
    GeneratorConfig config;
    config.bounded_column_probability = bounded;
    for (int k = 0; k < 150; ++k) {
      std::string failure;
      GeneratedLp lp;
      switch (k % 3) {
        case 0: lp = random_lp(rng, config); break;
        case 1: lp = degenerate_lp(rng, config); break;
        default: lp = kkt_lp(rng, config).lp; break;
      }
      if (!check_instance(lp, rng, &tally, &failure)) failures.push_back(failure);
    }
  }
  std::cout << "[safe bound] optimal " << tally.optimal << ", finite from the simplex duals "
            << tally.finite_from_simplex << " (worst optimum - safe " << tally.worst_gap
            << "), infeasible " << tally.infeasible << " of which proved by the Farkas test "
            << tally.infeasible_proved << ", skipped " << tally.skipped << "\n";
  for (const std::string& f : failures) ADD_FAILURE() << f;
  EXPECT_TRUE(failures.empty());
  // The bound is not vacuous: the simplex duals give a finite one on most optimal instances,
  // and it is tight to rounding.
  EXPECT_GT(tally.finite_from_simplex, tally.optimal / 2);
  EXPECT_LT(tally.worst_gap, 1e-6);
}

TEST(SafeBound, BranchAndBoundWithSafeBoundsMatchesTheExactMilp) {
  std::mt19937_64 rng(5190);
  GeneratorConfig config;
  config.bounded_column_probability = 1.0;
  int compared = 0;
  for (int k = 0; k < 40; ++k) {
    GeneratedLp lp = kkt_lp(rng, config).lp;
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 1);
    const OracleResult exact = solve_exact_milp(lp, 20000);
    if (exact.status != OracleStatus::kOptimal) continue;
    Model model = to_model(lp);
    for (auto& t : model.col_type) t = VarType::kInteger;
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("safe_bounds", true);
    const Solution solved = solve(model, options);
    ASSERT_EQ(solved.status, SolveStatus::kOptimal) << lp.to_text();
    EXPECT_NEAR(solved.objective, exact.objective.to_double(), 1e-6) << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 10);
}

}  // namespace
}  // namespace sankhya::oracle
