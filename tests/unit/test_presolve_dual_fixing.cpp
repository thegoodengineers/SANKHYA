// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dual fixing in presolve (#412).
//
// A reduction that is slightly wrong does not crash; it returns a confident, feasible-looking
// answer to a different problem. So the hand examples check that the fixing fires exactly
// where its argument says it may and stays out where the argument does not hold, and random
// instances are then solved with the reduction on against the exact rational oracle, LP and
// MILP, judged the way the fuzz harness judges the simplex: same status, same optimum.

#include <cmath>
#include <limits>
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

Options with_dual_fixing(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", true);
  options.set_bool("presolve_dual_fixing", on);
  options.set_int("node_limit", 100000);
  return options;
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            bool maximize = false) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
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
  EXPECT_EQ(model.validate(), "");
  return model;
}

Count dual_fixed(const Model& model, bool on) {
  Logger logger(nullptr);
  return presolve::presolve(model, with_dual_fixing(on), logger).report.dual_fixed_columns;
}

TEST(PresolveDualFixing, ACostlyColumnThatOnlyHurtsARowIsFixedAtItsLowerBound) {
  // min x0 + x1 + x2  s.t.  x0 + x1 >= 1,  x0 + x2 <= 4,  0 <= x <= 10. x2 costs 1 and its
  // one entry is +1 in a row bounded above only: moving it up never helps a row and costs,
  // so it is fixed at 0 and comes back nonbasic at its lower bound. The answer is 1 either
  // way, and the point is checked against the original model by postsolve.
  const Model model = build({{1, 1, 0}, {1, 0, 1}}, {1.0, -kInf}, {kInf, 4.0}, {1, 1, 1},
                            {0, 0, 0}, {10, 10, 10});
  EXPECT_EQ(dual_fixed(model, true), 1);
  EXPECT_EQ(dual_fixed(model, false), 0) << "off means off";
  const Solution on = solve(model, with_dual_fixing(true));
  const Solution off = solve(model, with_dual_fixing(false));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(on.objective, 1.0, 1e-9);
  EXPECT_NEAR(off.objective, 1.0, 1e-9);
  ASSERT_EQ(on.col_value.size(), 3u);
  EXPECT_EQ(on.col_value[2], 0.0);
  EXPECT_EQ(on.col_status[2], BasisStatus::kAtLower);
  EXPECT_LE(on.primal_infeasibility, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-7) << "a column fixed at its lower bound needs a "
                                            "non-negative reduced cost, and this one has it";
}

TEST(PresolveDualFixing, AProfitableColumnThatOnlyHelpsIsFixedAtItsUpperBound) {
  // max x2  s.t.  -x0 + x2 >= -5,  0 <= x <= 3. In minimise space x2 costs -1, and moving it
  // down never helps the row (a positive entry in a row bounded below), so it is fixed at
  // 3. x0 has cost 0 and a negative entry in the same row: moving it up never helps either,
  // so it is fixed at its lower bound 0. x1 is empty and is the empty-column reduction's.
  const Model model = build({{-1, 0, 1}}, {-5.0}, {kInf}, {0, 0, 1}, {0, 0, 0}, {3, 3, 3},
                            /*maximize=*/true);
  EXPECT_EQ(dual_fixed(model, true), 2);
  const Solution on = solve(model, with_dual_fixing(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 3.0, 1e-9);
  EXPECT_EQ(on.col_value[2], 3.0);
  EXPECT_EQ(on.col_status[2], BasisStatus::kAtUpper);
  EXPECT_EQ(on.col_value[0], 0.0);
  EXPECT_LE(on.primal_infeasibility, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-7);
}

TEST(PresolveDualFixing, StaysOutWhereTheArgumentDoesNotHold) {
  // A row bounded on both sides: moving either way can help, nothing is fixed.
  const Model both = build({{1, 1}}, {1.0}, {3.0}, {1, 1}, {0, -kInf}, {kInf, 5});
  EXPECT_EQ(dual_fixed(both, true), 0);
  // The bound to land on is infinite: min x0 s.t. x0 - x1 <= 2 would fix x0 at its lower
  // bound, which does not exist, and x1 at its upper bound, which does not exist either.
  const Model open = build({{1, -1}}, {-kInf}, {2.0}, {1, 0}, {-kInf, 0}, {10, kInf});
  EXPECT_EQ(dual_fixed(open, true), 0);
}

/// One random instance against the oracle, the fuzz harness's own comparison: the oracle's
/// status is the truth, its optimum to a relative 1e-6.
bool agree(const oracle::GeneratedLp& lp, const oracle::OracleResult& exact,
           const Solution& got, std::string* why) {
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
  (void)lp;
  return true;
}

TEST(PresolveDualFixing, RandomLpsAgreeWithTheExactOracle) {
  std::mt19937_64 rng(412);
  oracle::GeneratorConfig config;
  int compared = 0;
  int skipped = 0;
  Count fixed_total = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kOverflow ||
        exact.status == oracle::OracleStatus::kIterationLimit) {
      ++skipped;
      continue;
    }
    const Model model = oracle::to_model(lp);
    const Solution got = solve(model, with_dual_fixing(true));
    fixed_total += got.presolve_report.dual_fixed_columns;
    std::string why;
    EXPECT_TRUE(agree(lp, exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                             << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 300) << skipped << " skipped";
  EXPECT_GT(fixed_total, 0) << "the generator should give the reduction something to fix";
}

TEST(PresolveDualFixing, RandomMilpsAgreeWithTheExactOracle) {
  // The argument is about any feasible point, so it holds for integer columns as it does
  // for continuous ones; the exact branch and bound is the judge here.
  std::mt19937_64 rng(4120);
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  int compared = 0;
  Count fixed_total = 0;
  for (int trial = 0; trial < 200; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 2 == 0 ? 1 : 0;
    }
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;  // unbounded or abstained: nothing this test can compare
    }
    // to_model() builds the LP; the integrality the exact branch and bound honoured is
    // put on the model here, so the two engines see one MILP.
    Model model = oracle::to_model(lp);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.integral[static_cast<std::size_t>(j)] != 0) {
        model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      }
    }
    const Solution got = solve(model, with_dual_fixing(true));
    fixed_total += got.presolve_report.dual_fixed_columns;
    std::string why;
    EXPECT_TRUE(agree(lp, exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                             << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 100);
  EXPECT_GT(fixed_total, 0);
}

}  // namespace
}  // namespace sankhya
