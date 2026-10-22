// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve_dual_reductions (src/presolve/dual_reduce.cpp) against the exact
// oracle and against optima known in closed form, and the two reductions on models built
// to fire them.

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "presolve/dual_reduce.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {
namespace {

Solution solve_with(const Model& model, bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve_dual_reductions", on);
  return solve(model, options);
}

bool basis_counts(const Solution& s, const Model& model) {
  if (s.col_status.empty()) return true;
  Index basic = 0;
  for (const BasisStatus b : s.col_status) basic += b == BasisStatus::kBasic ? 1 : 0;
  for (const BasisStatus b : s.row_status) basic += b == BasisStatus::kBasic ? 1 : 0;
  return basic == model.num_rows();
}

/// Rows `equal` as equalities; the oracle sees each as the row and its negation.
Model with_equalities(const GeneratedLp& lp, const std::vector<char>& equal,
                      GeneratedLp* both) {
  Model model = to_model(lp);
  *both = lp;
  for (Index i = 0; i < lp.num_rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (equal[u] == 0) continue;
    model.row_upper[u] = model.row_lower[u];
    std::vector<std::int64_t> negated = lp.a[u];
    for (auto& v : negated) v = -v;
    both->a.push_back(negated);
    both->b.push_back(-lp.b[u]);
    ++both->num_rows;
  }
  return model;
}

TEST(FuzzDualReduce, AForcingRowFixesItsColumnsAndPricesItself) {
  // x0 + x1 >= 2 with x0, x1 in [0, 1]: forcing, both at 1. min x0 + 3 x1 + x2 subject to
  // x2 - x0 >= 0: x2 = 1; objective 5. The forcing row's dual must make d_0 and d_1 of the
  // sign an upper bound needs, nonpositive: y_r = max(t_0, t_1) with t_1 = 3.
  Model model;
  model.resize_columns(3);
  model.resize_rows(2);
  model.col_cost = {1.0, 3.0, 1.0};
  model.col_upper = {1.0, 1.0, kInfinity};
  model.row_lower = {2.0, 0.0};
  model.matrix.reset(2, 3);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 2, 1.0);
  model.matrix.add_entry(1, 0, -1.0);
  model.matrix.finalize();
  Options options;
  options.set_bool("log_to_console", false);
  Logger logger(nullptr);
  const presolve::DualReduceResult reduced = presolve::dual_reduce(model, options, logger);
  EXPECT_EQ(reduced.forcing_rows, 1);
  const Solution on = solve_with(model, true);
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 5.0, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-9);
  EXPECT_NEAR(on.row_dual[0], 3.0, 1e-9);
  EXPECT_TRUE(basis_counts(on, model));
}

TEST(FuzzDualReduce, AMaximisationIsReducedInTheMinimisationSense) {
  // max 3 x0 + 2 x1 + 5 s.t. x0 + x1 <= 4, x0 + 3 x1 <= 6, 0 <= x0 <= 3, x1 >= 0: optimum
  // (3, 1), objective 16, both rows tight. The first CI run of the default flip (#1164)
  // reported this `feasible`: the stage read the costs in the model's own sense, so under
  // maximisation its dual box and the forcing row's dual had the wrong sign.
  Model model;
  model.sense = ObjSense::kMaximize;
  model.objective_offset = 5.0;
  model.resize_columns(2);
  model.col_cost = {3.0, 2.0};
  model.col_upper = {3.0, kInfinity};
  model.resize_rows(2);
  model.row_upper = {4.0, 6.0};
  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, 3.0);
  model.matrix.finalize();
  const Solution on = solve_with(model, true);
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 16.0, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-9);
  EXPECT_TRUE(basis_counts(on, model));
  // And the forcing case under maximisation: max -x0 - 3 x1 - x2, x0 + x1 >= 2 with
  // x0, x1 in [0, 1], x2 - x0 >= 0: the minimisation twin of the forcing test, objective -5.
  Model forcing;
  forcing.sense = ObjSense::kMaximize;
  forcing.resize_columns(3);
  forcing.resize_rows(2);
  forcing.col_cost = {-1.0, -3.0, -1.0};
  forcing.col_upper = {1.0, 1.0, kInfinity};
  forcing.row_lower = {2.0, 0.0};
  forcing.matrix.reset(2, 3);
  forcing.matrix.add_entry(0, 0, 1.0);
  forcing.matrix.add_entry(0, 1, 1.0);
  forcing.matrix.add_entry(1, 2, 1.0);
  forcing.matrix.add_entry(1, 0, -1.0);
  forcing.matrix.finalize();
  const Solution forced = solve_with(forcing, true);
  ASSERT_EQ(forced.status, SolveStatus::kOptimal) << forced.message;
  EXPECT_NEAR(forced.objective, -5.0, 1e-9);
  EXPECT_LE(forced.dual_infeasibility, 1e-9);
  EXPECT_TRUE(basis_counts(forced, forcing));
}

TEST(FuzzDualReduce, ADominatedColumnIsFixedAtItsLowerBound) {
  // Row 0: x0 + x1 = 1 (equality, y_0 free). x1 is a column singleton with cost 1 and no
  // upper bound: y_0 <= 1. x0 has cost 3 and the same entry: d_0 = 3 - y_0 >= 2 > 0, so
  // x0 = 0 at every optimum. min 3 x0 + x1: objective 1.
  Model model;
  model.resize_columns(2);
  model.resize_rows(1);
  model.col_cost = {3.0, 1.0};
  model.row_lower = {1.0};
  model.row_upper = {1.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  Options options;
  options.set_bool("log_to_console", false);
  Logger logger(nullptr);
  const presolve::DualReduceResult reduced = presolve::dual_reduce(model, options, logger);
  EXPECT_EQ(reduced.dominated_columns, 1);
  const Solution on = solve_with(model, true);
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 1.0, 1e-9);
  EXPECT_NEAR(on.col_value[0], 0.0, 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-9);
  EXPECT_TRUE(basis_counts(on, model));
}

TEST(FuzzDualReduce, KktInstancesKeepTheirOptimum) {
  std::mt19937_64 rng(1006300603);
  GeneratorConfig config;
  config.bounded_column_probability = 0.7;  // boxes make forcing rows
  int agreed = 0, wrong = 0, fired = 0;
  std::vector<std::string> failures;
  Options probe;
  probe.set_bool("log_to_console", false);
  Logger logger(nullptr);
  for (int trial = 0; trial < 1500; ++trial) {
    const KktInstance instance = kkt_lp(rng, config);
    const GeneratedLp& lp = instance.lp;
    std::vector<char> equal(static_cast<std::size_t>(lp.num_rows), 0);
    for (Index i = 0; i < lp.num_rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      std::int64_t activity = 0;
      for (Index j = 0; j < lp.num_cols; ++j) {
        activity += lp.a[u][static_cast<std::size_t>(j)] *
                    instance.optimal_x[static_cast<std::size_t>(j)];
      }
      equal[u] = (activity == lp.b[u] && (i % 2 == 0)) ? 1 : 0;
    }
    GeneratedLp both;
    const Model model = with_equalities(lp, equal, &both);
    fired += presolve::dual_reduce(model, probe, logger).reduced_anything() ? 1 : 0;
    const Solution s = solve_with(model, true);
    const auto expected = static_cast<double>(instance.optimal_objective);
    const bool ok =
        s.status == SolveStatus::kOptimal &&
        std::fabs(s.objective - expected) <= 1e-6 * std::max(1.0, std::fabs(expected)) &&
        s.dual_infeasibility <= 1e-6 && basis_counts(s, model);
    if (ok) {
      ++agreed;
      continue;
    }
    ++wrong;
    if (failures.size() < 5) {
      failures.push_back(std::string("status ") + to_string(s.status) + " objective " +
                         std::to_string(s.objective) + " expected " + std::to_string(expected) +
                         " dual infeasibility " + std::to_string(s.dual_infeasibility) +
                         (basis_counts(s, model) ? "" : " BASIS COUNT WRONG") + "\n" +
                         both.to_text());
    }
  }
  std::cout << "\n=== 1500 KKT instances, dual reductions on ===\n  agreed " << agreed
            << "\n  WRONG " << wrong << "\n  instances the stage reduced " << fired << "\n";
  for (const std::string& f : failures)
    std::cout << "\n--- failing instance ---\n" << f << "\n";
  EXPECT_EQ(wrong, 0);
  EXPECT_GT(agreed, 1400);
  EXPECT_GT(fired, 100) << "the generator stopped producing forcing rows and dominated columns";
}

TEST(FuzzDualReduce, RandomInstancesAgainstTheOracle) {
  std::mt19937_64 rng(1006400604);
  GeneratorConfig config;
  config.bounded_column_probability = 0.7;
  std::bernoulli_distribution third(1.0 / 3.0);
  int optimal = 0, other = 0, skipped = 0, mismatched = 0;
  std::vector<std::string> failures;
  for (int trial = 0; trial < 2000; ++trial) {
    const GeneratedLp lp = random_lp(rng, config);
    std::vector<char> equal(static_cast<std::size_t>(lp.num_rows), 0);
    for (auto& e : equal) e = third(rng) ? 1 : 0;
    GeneratedLp both;
    const Model model = with_equalities(lp, equal, &both);
    const OracleResult exact = solve_exact(both);
    if (exact.status == OracleStatus::kOverflow ||
        exact.status == OracleStatus::kIterationLimit) {
      ++skipped;
      continue;
    }
    const Solution s = solve_with(model, true);
    bool agree = false;
    switch (exact.status) {
      case OracleStatus::kOptimal: {
        const double expected = exact.objective.to_double();
        agree =
            s.status == SolveStatus::kOptimal &&
            std::fabs(s.objective - expected) <= 1e-6 * std::max(1.0, std::fabs(expected)) &&
            basis_counts(s, model);
        optimal += agree ? 1 : 0;
        break;
      }
      case OracleStatus::kInfeasible:
        agree = s.status == SolveStatus::kInfeasible;
        other += agree ? 1 : 0;
        break;
      case OracleStatus::kUnbounded:
        agree = s.status == SolveStatus::kUnbounded;
        other += agree ? 1 : 0;
        break;
      case OracleStatus::kOverflow:
      case OracleStatus::kIterationLimit: break;
    }
    if (agree) continue;
    ++mismatched;
    if (failures.size() < 5) {
      failures.push_back(std::string("oracle ") + to_string(exact.status) + " solver " +
                         to_string(s.status) + " objective " + std::to_string(s.objective) +
                         "\n" + both.to_text());
    }
  }
  std::cout << "\n=== 2000 random instances, dual reductions on ===\n  agreed optimal "
            << optimal << "\n  agreed infeasible/unbounded " << other << "\n  oracle abstained "
            << skipped << "\n  MISMATCHED " << mismatched << "\n";
  for (const std::string& f : failures)
    std::cout << "\n--- failing instance ---\n" << f << "\n";
  EXPECT_EQ(mismatched, 0);
  EXPECT_GT(optimal, 80);
  EXPECT_GT(other, 200);
}

}  // namespace
}  // namespace sankhya::oracle
