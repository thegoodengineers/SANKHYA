// SPDX-License-Identifier: Apache-2.0
// SANKHYA - implied-free column substitution in presolve (#412).
//
// A column substituted away on a box that DOES bind comes back from postsolve outside its
// bounds, or the search never sees the bound it needed: a confident answer to a different
// problem. So the hand examples check that the substitution fires exactly where the row's
// activity range implies the box and stays out where the box is tighter than the row, that
// the recovered value lands inside the box, and random instances are then solved with the
// reduction on against the exact rational oracle, LP and MILP, judged the way the fuzz
// harness judges the simplex: same status, same optimum.

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options with_implied_free(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_implied_free", on);
  options.set_int("node_limit", 100000);
  return options;
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            bool maximize = false, const std::vector<bool>& integer = {}) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
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

Solution::PresolveReport report_of(const Model& model, bool on) {
  Logger logger(nullptr);
  return presolve::presolve(model, with_implied_free(on), logger).report;
}

TEST(PresolveImpliedFree, AColumnWhoseRowImpliesItsBoxIsSubstitutedAway) {
  // min x0 + 3 x1  s.t.  2 <= x0 + x1 <= 5,  0 <= x0 <= 10,  0 <= x1 <= 1. x0 is in one
  // row, and with x1 in [0, 1] the row puts x0 in [1, 5], inside its box [0, 10]: the box
  // never binds, so x0 goes with its row as a free column singleton would. The cost rate
  // pushes the row to its lower bound 2, x0 = 2 - x1, the objective 2 + 2 x1 is minimised
  // at x1 = 0, and x0 comes back as 2, inside its box. The answer is 2 either way.
  const Model model = build({{1, 1}}, {2.0}, {5.0}, {1, 3}, {0, 0}, {10, 1});
  const Solution::PresolveReport on_report = report_of(model, true);
  EXPECT_EQ(on_report.implied_free_column_singletons, 1);
  EXPECT_EQ(on_report.free_column_singletons, 1) << "counted under the kind it replays as";
  EXPECT_EQ(report_of(model, false).implied_free_column_singletons, 0) << "off means off";
  const Solution on = solve(model, with_implied_free(true));
  const Solution off = solve(model, with_implied_free(false));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(on.objective, 2.0, 1e-9);
  EXPECT_NEAR(off.objective, 2.0, 1e-9);
  ASSERT_EQ(on.col_value.size(), 2u);
  EXPECT_NEAR(on.col_value[0], 2.0, 1e-9);
  EXPECT_GE(on.col_value[0], 0.0);
  EXPECT_LE(on.col_value[0], 10.0);
  EXPECT_LE(on.primal_infeasibility, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-7);
}

TEST(PresolveImpliedFree, ANegativeCoefficientReversesTheImpliedInterval) {
  // max x0  s.t.  -4 <= x1 - x0 <= 1,  -20 <= x0 <= 20,  0 <= x1 <= 2. With a = -1 the row
  // puts x0 in [x1 - 1, x1 + 4] = [-1, 6], inside [-20, 20]. Maximising x0 pushes the row to
  // its lower bound -4: x0 = x1 + 4, at most 6 with x1 = 2.
  const Model model = build({{-1, 1}}, {-4.0}, {1.0}, {1, 0}, {-20, 0}, {20, 2},
                            /*maximize=*/true);
  EXPECT_EQ(report_of(model, true).implied_free_column_singletons, 1);
  const Solution on = solve(model, with_implied_free(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 6.0, 1e-9);
  EXPECT_NEAR(on.col_value[0], 6.0, 1e-9);
  EXPECT_LE(on.primal_infeasibility, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-7);
}

TEST(PresolveImpliedFree, StaysOutWhereTheBoxIsTighterThanTheRow) {
  // The same row as the first test with x0's box [0, 3]: the row allows x0 up to 5, so the
  // box binds and nothing is substituted; the search must see the bound.
  const Model tight = build({{1, 1}}, {2.0}, {5.0}, {1, 3}, {0, 0}, {3, 1});
  EXPECT_EQ(report_of(tight, true).implied_free_column_singletons, 0);
  // A row bounded above only implies no lower bound on x0, whose lower bound is finite.
  const Model open_below = build({{1, 1}}, {-kInf}, {5.0}, {1, 3}, {0, 0}, {10, 1});
  EXPECT_EQ(report_of(open_below, true).implied_free_column_singletons, 0);
  // The other column unbounded: the rest's range is infinite, so nothing is implied.
  const Model open_rest = build({{1, 1}}, {2.0}, {5.0}, {1, 3}, {0, 0}, {10, kInf});
  EXPECT_EQ(report_of(open_rest, true).implied_free_column_singletons, 0);
  // An integer column is never substituted: its recovered value need not be integral.
  const Model integer = build({{1, 1}}, {2.0}, {5.0}, {1, 3}, {0, 0}, {10, 1},
                              /*maximize=*/false, {true, false});
  EXPECT_EQ(report_of(integer, true).implied_free_column_singletons, 0);
  // A column in two rows is not a singleton; the second row shares x2 so that it is not a
  // singleton row the bound reduction would fold into x0's box first.
  const Model two_rows =
      build({{1, 1, 0}, {1, 0, 1}}, {2.0, -kInf}, {5.0, 8.0}, {1, 3, 0}, {0, 0, 0}, {10, 1, 5});
  EXPECT_EQ(report_of(two_rows, true).implied_free_column_singletons, 0);
  // A tie is declined: the box [1, 5] equals the implied interval, so the comparison
  // without slack admits it, and the box [1.0000001, 5] does not.
  const Model exact_fit = build({{1, 1}}, {2.0}, {5.0}, {1, 3}, {1, 0}, {5, 1});
  EXPECT_EQ(report_of(exact_fit, true).implied_free_column_singletons, 1);
  const Model hair_tighter = build({{1, 1}}, {2.0}, {5.0}, {1, 3}, {1.0000001, 0}, {5, 1});
  EXPECT_EQ(report_of(hair_tighter, true).implied_free_column_singletons, 0);
}

/// One random instance against the oracle, the fuzz harness's own comparison: the oracle's
/// status is the truth, its optimum to a relative 1e-6.
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
        *why = "objectives differ by " + std::to_string(difference) + " (" +
               std::to_string(got.objective) + " against " + std::to_string(expected) + ")";
        return false;
      }
      return true;
    }
    case oracle::OracleStatus::kInfeasible:
      if (got.status != SolveStatus::kInfeasible) {
        *why = "exactly infeasible, the solver said " + std::string(to_string(got.status));
        return false;
      }
      return true;
    case oracle::OracleStatus::kUnbounded:
      if (got.status != SolveStatus::kUnbounded) {
        *why = "exactly unbounded, the solver said " + std::string(to_string(got.status));
        return false;
      }
      return true;
    case oracle::OracleStatus::kOverflow:
    case oracle::OracleStatus::kIterationLimit: return true;
  }
  return true;
}

/// Append a column that sits in exactly one row with a coefficient of +-1 and a wide box,
/// so the row's range against the other columns' boxes often implies the box. The oracle
/// judges the augmented instance whole, so nothing about the column is assumed.
void add_a_singleton_column(oracle::GeneratedLp* lp, std::mt19937_64& rng) {
  if (lp->num_rows == 0) return;
  std::uniform_int_distribution<Index> pick_row(0, lp->num_rows - 1);
  std::uniform_int_distribution<int> coin(0, 1);
  std::uniform_int_distribution<std::int64_t> cost(-3, 3);
  const auto row = static_cast<std::size_t>(pick_row(rng));
  for (std::size_t i = 0; i < lp->a.size(); ++i) {
    lp->a[i].push_back(i == row ? (coin(rng) == 0 ? 1 : -1) : 0);
  }
  lp->c.push_back(cost(rng));
  lp->upper.push_back(coin(rng) == 0 ? oracle::kNoUpperBound : 40);
  if (!lp->integral.empty()) lp->integral.push_back(0);
  if (!lp->lower.empty()) lp->lower.push_back(0);
  ++lp->num_cols;
}

TEST(PresolveImpliedFree, RandomLpsAgreeWithTheExactOracle) {
  std::mt19937_64 rng(4124);
  oracle::GeneratorConfig config;
  int compared = 0;
  int skipped = 0;
  Count substituted = 0;
  for (int trial = 0; trial < 400; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    add_a_singleton_column(&lp, rng);
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kOverflow ||
        exact.status == oracle::OracleStatus::kIterationLimit) {
      ++skipped;
      continue;
    }
    const Model model = oracle::to_model(lp);
    const Solution got = solve(model, with_implied_free(true));
    substituted += got.presolve_report.implied_free_column_singletons;
    std::string why;
    EXPECT_TRUE(agree(exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                         << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 300) << skipped << " skipped";
  EXPECT_GT(substituted, 0) << "the generator should give the reduction something to do";
}

TEST(PresolveImpliedFree, RandomMilpsAgreeWithTheExactOracle) {
  // The substituted column is continuous by rule; the integer columns around it are what
  // the exact branch and bound judges.
  std::mt19937_64 rng(41240);
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  int compared = 0;
  Count substituted = 0;
  for (int trial = 0; trial < 200; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 2 == 0 ? 1 : 0;
    }
    add_a_singleton_column(&lp, rng);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;
    }
    Model model = oracle::to_model(lp);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.integral[static_cast<std::size_t>(j)] != 0) {
        model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      }
    }
    const Solution got = solve(model, with_implied_free(true));
    substituted += got.presolve_report.implied_free_column_singletons;
    std::string why;
    EXPECT_TRUE(agree(exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                         << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 100);
  EXPECT_GT(substituted, 0);
}

}  // namespace
}  // namespace sankhya
