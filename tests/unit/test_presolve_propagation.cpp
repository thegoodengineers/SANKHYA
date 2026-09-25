// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bound propagation from row activity in presolve (#485).
//
// A propagated bound is in the reduced model's box and not in the original's. When the
// engine leaves a column on it, the column is inside its original box and the reduced cost
// it carries has to move onto the row that implied the bound, or the answer handed back is
// dual infeasible on the model the caller asked about - with the right objective, which is
// why every test here judges the returned point AND duals against the ORIGINAL model with the
// in-process copy of the verifier's checks (core/kkt_check.hpp), basis included when the
// engine reports one. The hand cases put the propagated bound at the optimum on purpose,
// one of them through a chain of two propagations; random instances are then held to the
// exact rational oracle and to the same KKT check.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/kkt_check.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Options with_propagation(bool on, const std::string& algorithm = "auto") {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_bound_propagation", on);
  if (algorithm != "auto") options.set_string("algorithm", algorithm);
  return options;
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integer = {}) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (std::size_t j = 0; j < integer.size(); ++j) {
    if (integer[j]) model.col_type[j] = VarType::kInteger;
  }
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
  EXPECT_EQ(model.validate(), "");
  return model;
}

/// Solved with the reduction on by `algorithm`, optimal, propagated at least `propagated`
/// bounds, and the answer passes the verifier's checks on the ORIGINAL model.
Solution solve_and_verify(const Model& model, const std::string& algorithm, Count propagated) {
  const Solution s = solve(model, with_propagation(true, algorithm));
  EXPECT_EQ(s.status, SolveStatus::kOptimal) << algorithm << ": " << s.message;
  EXPECT_GE(s.presolve_report.propagated_bounds, propagated) << algorithm;
  const KktVerdict verdict = check_lp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << algorithm << ": " << verdict.check << ": " << verdict.detail;
  return s;
}

const char* const kEngines[] = {"dual-simplex", "simplex", "ipm"};

TEST(PresolvePropagation, OffByDefault) {
  EXPECT_FALSE(Options().get_bool("presolve_bound_propagation"));
  // x1 <= 3 puts x0 >= 2 through the row; with the option off nothing is propagated.
  const Model model = build({{1, 1}}, {5}, {kInfinity}, {2, -5}, {0, 0}, {10, 3});
  const Solution s = solve(model, with_propagation(false));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.presolve_report.propagated_bounds, 0);
}

TEST(PresolvePropagation, AnActivePropagatedBoundHandsItsPriceToTheRow) {
  // min 2 x0 - 5 x1  s.t.  x0 + x1 >= 5,  x0 in [0, 10],  x1 in [0, 3].
  // The row and x1 <= 3 give x0 >= 2, and the optimum is x0 = 2, x1 = 3, objective -11: x0
  // sits ON the propagated bound, strictly inside its own box [0, 10]. In the reduced model
  // the bound can carry x0's whole price (row dual 0, reduced cost 2 - a degenerate vertex,
  // which is what the simplex lands on); on the original the only admissible duals are
  // y = 2 on the row and d0 = 0, since x0 is interior. Without the transfer the answer is
  // right and its duals are not.
  const Model model = build({{1, 1}}, {5}, {kInfinity}, {2, -5}, {0, 0}, {10, 3});
  for (const char* algorithm : kEngines) {
    const Solution s = solve_and_verify(model, algorithm, 1);
    if (s.status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(s.objective, -11.0, 1e-7) << algorithm;
    EXPECT_NEAR(s.col_value[0], 2.0, 1e-6) << algorithm;
    EXPECT_NEAR(s.row_dual[0], 2.0, 1e-6) << algorithm;
    EXPECT_NEAR(s.col_dual[0], 0.0, 1e-6) << algorithm;
    EXPECT_NEAR(s.col_dual[1], -7.0, 1e-6) << algorithm;
  }
}

TEST(PresolvePropagation, AChainOfPropagatedBoundsIsPricedBackRowByRow) {
  // min 2 x0 - 5 x1 - x2  s.t.  x0 + x1 >= 5,  x0 + x2 <= 12,  x0 in [0, 10], x1 in [0, 3],
  // x2 in [0, 20]. Row 0 gives x0 >= 2, and row 1 then gives x2 <= 12 - 2 = 10 FROM that
  // propagated bound. Optimum x = (2, 3, 10), objective -21, both propagated bounds active
  // and both columns inside their own boxes, so both prices go back: y1 = -1 zeroes d2,
  // then y0 = 3 zeroes d0 = 2 - y0 - y1. Postsolve has to undo the second propagation
  // before the first.
  const Model model = build({{1, 1, 0}, {1, 0, 1}}, {5, -kInfinity}, {kInfinity, 12},
                            {2, -5, -1}, {0, 0, 0}, {10, 3, 20});
  for (const char* algorithm : kEngines) {
    const Solution s = solve_and_verify(model, algorithm, 2);
    if (s.status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(s.objective, -21.0, 1e-7) << algorithm;
    EXPECT_NEAR(s.row_dual[0], 3.0, 1e-6) << algorithm;
    EXPECT_NEAR(s.row_dual[1], -1.0, 1e-6) << algorithm;
  }
}

TEST(PresolvePropagation, AMaximizeModelIsPricedInItsOwnSense) {
  // The first case negated into a maximisation: max -2 x0 + 5 x1, the same point, objective
  // 11, and the row's dual reported in the model's own sense.
  Model model = build({{1, 1}}, {5}, {kInfinity}, {-2, 5}, {0, 0}, {10, 3});
  model.sense = ObjSense::kMaximize;
  for (const char* algorithm : kEngines) {
    const Solution s = solve_and_verify(model, algorithm, 1);
    if (s.status != SolveStatus::kOptimal) continue;
    EXPECT_NEAR(s.objective, 11.0, 1e-7) << algorithm;
    EXPECT_NEAR(s.col_dual[0], 0.0, 1e-6) << algorithm;
  }
}

TEST(PresolvePropagation, CrossedPropagatedBoundsProveInfeasibility) {
  // x0 + x1 >= 8 with x1 <= 3 gives x0 >= 5; x0 + x2 <= 4 with x2 >= 0 gives x0 <= 4.
  // Neither row is infeasible on its own box, so the activity test does not see it.
  const Model model = build({{1, 1, 0}, {1, 0, 1}}, {8, -kInfinity}, {kInfinity, 4}, {1, 1, 1},
                            {0, 0, 0}, {10, 3, 10});
  const Solution on = solve(model, with_propagation(true));
  const Solution off = solve(model, with_propagation(false));
  EXPECT_EQ(off.status, SolveStatus::kInfeasible) << off.message;
  EXPECT_EQ(on.status, SolveStatus::kInfeasible) << on.message;
}

TEST(PresolvePropagation, IntegerBoundsAreRoundedInwardOnlyOnIntegerColumns) {
  // 2 x0 + x1 >= 6, x1 in [0, 1.5]: x0 >= 2.25, which an integer x0 makes 3 and a
  // continuous one keeps. min x0 + 3 x1: the MILP optimum is x0 = 3, x1 = 0 (objective 3);
  // the LP's is x0 = 3 as well, but priced through the row.
  const Model milp =
      build({{2, 1}}, {6}, {kInfinity}, {1, 3}, {0, 0}, {10, 1.5}, {true, false});
  const Solution on = solve(milp, with_propagation(true));
  const Solution off = solve(milp, with_propagation(false));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(on.objective, off.objective, 1e-9);
  EXPECT_NEAR(on.objective, 3.0, 1e-9);

  // x0 continuous, cost favouring x1: x1 = 1.5, x0 = 2.25 - on the propagated bound, which
  // an inward integer rounding would have cut to 3.
  const Model lp = build({{2, 1}}, {6}, {kInfinity}, {1, 0.1}, {0, 0}, {10, 1.5});
  const Solution s = solve_and_verify(lp, "dual-simplex", 1);
  EXPECT_NEAR(s.col_value[0], 2.25, 1e-7);
}

/// The exact oracle's verdict against ours: same status, the optimum to 1e-6 relative.
bool agree(const oracle::OracleResult& exact, const Solution& got, std::string* why) {
  switch (exact.status) {
    case oracle::OracleStatus::kOptimal: {
      if (got.status != SolveStatus::kOptimal) {
        *why = "the exact optimum exists but the solver said " +
               std::string(to_string(got.status)) + ": " + got.message;
        return false;
      }
      const double expected = exact.objective.to_double();
      const double difference = std::fabs(got.objective - expected);
      if (difference > 1e-6 * std::max(1.0, std::fabs(expected))) {
        *why = "objectives differ: " + std::to_string(got.objective) + " against " +
               std::to_string(expected);
        return false;
      }
      return true;
    }
    case oracle::OracleStatus::kInfeasible:
      if (got.status == SolveStatus::kInfeasible) return true;
      *why = "exactly infeasible, the solver said " + std::string(to_string(got.status));
      return false;
    case oracle::OracleStatus::kUnbounded:
      if (got.status == SolveStatus::kUnbounded) return true;
      *why = "exactly unbounded, the solver said " + std::string(to_string(got.status));
      return false;
    case oracle::OracleStatus::kOverflow:
    case oracle::OracleStatus::kIterationLimit: return true;
  }
  return true;
}

/// Random instances, general and degenerate, with most columns boxed so rows imply bounds,
/// solved with the reduction on and off by each simplex and the interior point. Wherever the
/// answer without it agrees with the exact oracle, the answer with it must too; wherever the
/// answer without it passes the KKT check on the original model, the answer with it must
/// too. What the engines already get wrong on their own (counted and printed) is not this
/// reduction's to fix, and holding it to that would hide which change broke what.
TEST(PresolvePropagation, RandomLpsAgreeWithTheOracleAndVerifyOnTheOriginal) {
  std::mt19937_64 rng(485);
  oracle::GeneratorConfig config;
  config.bounded_column_probability = 0.9;
  int compared = 0;
  int verified = 0;
  int wrong_without = 0;
  Count propagated = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const oracle::GeneratedLp lp =
        trial % 2 == 0 ? oracle::random_lp(rng, config) : oracle::degenerate_lp(rng, config);
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kOverflow ||
        exact.status == oracle::OracleStatus::kIterationLimit) {
      continue;
    }
    const Model model = oracle::to_model(lp);
    for (const char* algorithm : kEngines) {
      const Solution off = solve(model, with_propagation(false, algorithm));
      const Solution on = solve(model, with_propagation(true, algorithm));
      propagated += on.presolve_report.propagated_bounds;
      std::string why;
      if (!agree(exact, off, &why)) {
        ++wrong_without;
        continue;
      }
      EXPECT_TRUE(agree(exact, on, &why))
          << "trial " << trial << " " << algorithm << ": " << why << '\n'
          << lp.to_text();
      ++compared;
      if (exact.status != oracle::OracleStatus::kOptimal) continue;
      if (on.status != SolveStatus::kOptimal) continue;
      if (!check_lp_optimality(model, off).passed) {
        ++wrong_without;
        continue;
      }
      const KktVerdict verdict = check_lp_optimality(model, on);
      EXPECT_TRUE(verdict.passed) << "trial " << trial << " " << algorithm << ": "
                                  << verdict.check << ": " << verdict.detail << '\n'
                                  << lp.to_text();
      ++verified;
    }
  }
  std::cout << "propagation fuzz: " << compared << " compared, " << verified
            << " verified on the original, " << wrong_without
            << " already wrong without the reduction, " << propagated << " bounds propagated\n";
  EXPECT_GT(compared, 600);
  EXPECT_GT(verified, 200);
  EXPECT_GT(propagated, 100) << "the generator should give the reduction something to do";
}

}  // namespace
}  // namespace sankhya
