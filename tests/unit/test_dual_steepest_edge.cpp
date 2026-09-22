// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dual steepest-edge pricing (#411).
//
// The weight is a norm the update only reproduces in exact arithmetic; if it drifts in
// floating point the rule degrades into a bad Devex with nothing to say so. So the first
// test recomputes every row norm of B^-1 from the factors at the end of a dual solve and
// holds the maintained weights to it; the second solves random LPs under the rule against
// the exact rational oracle, the way the fuzz harness judges every pricing rule; the third
// checks two runs are the same run.

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "simplex/simplex_core.hpp"

namespace sankhya {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options dual_steepest_edge() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "dual-simplex");
  options.set_string("pricing", "dual-steepest-edge");
  return options;
}

/// A random covering LP, min c x s.t. A x >= b, 0 <= x <= 3, with every cost positive: the
/// slack basis is dual feasible, so the dual simplex runs it to the optimum on its own with
/// no hand-over to the primal, whose weights are not what is under test.
Model covering_lp(std::mt19937& rng, Index rows, Index cols) {
  std::uniform_int_distribution<int> coefficient(0, 4);
  std::uniform_int_distribution<int> cost(1, 9);
  Model m;
  const auto n = static_cast<std::size_t>(cols);
  m.col_cost.resize(n);
  for (double& c : m.col_cost) c = cost(rng);
  m.col_lower.assign(n, 0.0);
  m.col_upper.assign(n, 3.0);
  m.col_type.assign(n, VarType::kContinuous);
  m.matrix.reset(rows, cols);
  m.row_lower.resize(static_cast<std::size_t>(rows));
  m.row_upper.assign(static_cast<std::size_t>(rows), kInf);
  for (Index i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (Index j = 0; j < cols; ++j) {
      const double a = coefficient(rng);
      if (a != 0.0) {
        m.matrix.add_entry(i, j, a);
        sum += a;
      }
    }
    m.row_lower[static_cast<std::size_t>(i)] = std::floor(sum / 3.0) + 1.0;
  }
  m.matrix.finalize();
  m.hessian.reset(cols, cols);
  m.hessian.finalize();
  EXPECT_EQ(m.validate(), "");
  return m;
}

TEST(DualSteepestEdge, TheMaintainedWeightsMatchTheExactNormsAtTheEnd) {
  std::mt19937 rng(411);
  int checked = 0;
  Count pivots = 0;
  for (int trial = 0; trial < 12; ++trial) {
    const Model m = covering_lp(rng, 12 + trial, 18 + trial);
    Logger logger(nullptr);
    // The engine keeps a reference to its options, so they must outlive it: a temporary
    // here would leave it reading freed memory for its limits.
    const Options options = dual_steepest_edge();
    detail::Simplex simplex(m, options, logger, nullptr);
    const Solution solved = simplex.run_dual();
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    ASSERT_EQ(solved.algorithm, "simplex-dual")
        << "trial " << trial << ": the dual handed over to the primal";
    const std::vector<double>& kept = simplex.dual_weights_for_testing();
    const std::vector<double> exact = simplex.exact_dual_weights_for_testing();
    ASSERT_EQ(kept.size(), exact.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
      EXPECT_NEAR(kept[i], exact[i], 1e-9 * std::max(1.0, exact[i]))
          << "trial " << trial << ", row " << i << ": the maintained weight drifted from "
          << "the exact norm";
    }
    pivots += solved.iterations;
    ++checked;
  }
  EXPECT_EQ(checked, 12);
  EXPECT_GT(pivots, 50)
      << "the models should take some pivots, or the update was not exercised";
}

TEST(DualSteepestEdge, RandomLpsAgreeWithTheExactOracle) {
  std::mt19937_64 rng(4110);
  oracle::GeneratorConfig config;
  int compared = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    const oracle::OracleResult exact = oracle::solve_exact(lp);
    if (exact.status == oracle::OracleStatus::kOverflow ||
        exact.status == oracle::OracleStatus::kIterationLimit) {
      continue;
    }
    const Solution got = solve(oracle::to_model(lp), dual_steepest_edge());
    switch (exact.status) {
      case oracle::OracleStatus::kOptimal: {
        ASSERT_EQ(got.status, SolveStatus::kOptimal)
            << "trial " << trial << ": " << got.message << "\n"
            << lp.to_text();
        const double expected = exact.objective.to_double();
        EXPECT_NEAR(got.objective, expected, 1e-6 * std::max(1.0, std::fabs(expected)))
            << "trial " << trial << "\n"
            << lp.to_text();
        break;
      }
      case oracle::OracleStatus::kInfeasible:
        EXPECT_EQ(got.status, SolveStatus::kInfeasible) << "trial " << trial;
        break;
      case oracle::OracleStatus::kUnbounded:
        EXPECT_EQ(got.status, SolveStatus::kUnbounded) << "trial " << trial;
        break;
      case oracle::OracleStatus::kOverflow:
      case oracle::OracleStatus::kIterationLimit: break;
    }
    ++compared;
  }
  EXPECT_GT(compared, 250);
}

TEST(DualSteepestEdge, TwoRunsAreTheSameRun) {
  std::mt19937 rng(41100);
  const Model m = covering_lp(rng, 20, 30);
  const Solution first = solve(m, dual_steepest_edge());
  const Solution second = solve(m, dual_steepest_edge());
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  EXPECT_EQ(second.status, first.status);
  EXPECT_EQ(second.iterations, first.iterations);
  EXPECT_EQ(second.objective, first.objective);
  EXPECT_EQ(second.col_value, first.col_value);
}

TEST(DualSteepestEdge, TheChoiceIsRegisteredBesideTheOthers) {
  Options options;
  std::string error;
  for (const char* rule : {"devex", "dantzig", "dual-steepest-edge"}) {
    EXPECT_TRUE(options.set_from_string("pricing", rule, &error)) << rule << ": " << error;
  }
  EXPECT_FALSE(options.set_from_string("pricing", "steepest", &error));
}

}  // namespace
}  // namespace sankhya
