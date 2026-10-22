// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve_substitution (src/presolve/substitute.cpp) against the exact oracle and
// against optima known in closed form.
//
// The generators write A x >= b only, and the stage works on equality rows, so each test
// turns some rows into equalities. The oracle sees an equality as the row and its negation
// (a x >= b and -a x >= -b), the float solver as one row with equal bounds: the same set of
// points, so a disagreement cannot be blamed on the encoding.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "presolve/substitute.hpp"
#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {
namespace {

Solution solve_with_substitution(const Model& model, bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve_substitution", on);
  return solve(model, options);
}

/// The model with rows `equal` as equalities; and for the oracle, `lp` with their negations
/// appended.
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

/// Basic statuses must number exactly the rows: a postsolve that loses or invents one has
/// handed back something that is not a basis.
bool basis_counts(const Solution& s, const Model& model) {
  if (s.col_status.empty()) return true;
  Index basic = 0;
  for (const BasisStatus b : s.col_status) basic += b == BasisStatus::kBasic ? 1 : 0;
  for (const BasisStatus b : s.row_status) basic += b == BasisStatus::kBasic ? 1 : 0;
  return basic == model.num_rows();
}

GeneratorConfig sparse_config() {
  GeneratorConfig config;
  config.min_rows = 3;
  config.max_rows = 10;
  config.min_cols = 3;
  config.max_cols = 12;
  config.density = 0.3;  // two- and three-entry rows, where doubletons and chains appear
  return config;
}

TEST(FuzzSubstitution, AChainOfDoubletonsIsSubstitutedAndUndone) {
  // x0 - x1 = 0, x1 - x2 = 1, x2 - x3 = 0 (a chain presolve.cpp folds one link of), with
  // x0 + x3 >= 5 and boxes that the chain carries: min x0 + 2 x1 + x3.
  Model model;
  model.resize_columns(4);
  model.resize_rows(4);
  model.col_cost = {1.0, 2.0, 0.0, 1.0};
  model.col_upper = {10.0, 4.0, kInfinity, 10.0};
  model.row_lower = {0.0, 1.0, 0.0, 5.0};
  model.row_upper = {0.0, 1.0, 0.0, kInfinity};
  model.matrix.reset(4, 4);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, -1.0);
  model.matrix.add_entry(1, 1, 1.0);
  model.matrix.add_entry(1, 2, -1.0);
  model.matrix.add_entry(2, 2, 1.0);
  model.matrix.add_entry(2, 3, -1.0);
  model.matrix.add_entry(3, 0, 1.0);
  model.matrix.add_entry(3, 3, 1.0);
  model.matrix.finalize();

  Options options;
  options.set_bool("log_to_console", false);
  Logger logger(nullptr);
  const presolve::SubstitutionResult reduced = presolve::substitute(model, options, logger);
  EXPECT_EQ(reduced.records.size(), 3U);
  EXPECT_EQ(reduced.model.num_rows(), 1);

  const Solution off = solve_with_substitution(model, false);
  const Solution on = solve_with_substitution(model, true);
  ASSERT_EQ(off.status, SolveStatus::kOptimal);
  ASSERT_EQ(on.status, SolveStatus::kOptimal);
  // x1 = x0, x2 = x0 - 1, x3 = x0 - 1, 2 x0 - 1 >= 5: x0 = 3, objective 3 + 6 + 2 = 11.
  EXPECT_NEAR(on.objective, 11.0, 1e-9);
  EXPECT_NEAR(off.objective, 11.0, 1e-9);
  for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(on.row_dual[i], off.row_dual[i], 1e-9);
  EXPECT_LE(on.dual_infeasibility, 1e-9);
  EXPECT_TRUE(basis_counts(on, model));
}

TEST(FuzzSubstitution, OnlyTheSimplexEverSeesASubstitutedModel) {
  // #1125: under the interior point the crossover on cycle stopped short of a vertex and
  // capri's dense-column directions lost accuracy on a substituted model, so the stage runs
  // where a cold simplex solve starts and nowhere else: an explicit interior point or PDHG
  // never reaches it, and in the race (deterministic: every engine runs, in order) only the
  // dual simplex racer does. scorpion is a model the stage reduces by three quarters.
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib/scorpion.mps")
          .string();
  ASSERT_TRUE(io::read_model(path, &model).ok) << path;
  const auto run = [&](const std::vector<std::pair<std::string, std::string>>& extra) {
    Options options;
    options.set_bool("log_to_console", false);
    std::string error;
    for (const auto& [key, value] : extra) {
      EXPECT_TRUE(options.set_from_string(key, value, &error)) << error;
    }
    const std::int64_t before = presolve::substitutions_applied_for_testing();
    const Solution s = solve(model, options);
    EXPECT_EQ(s.status, SolveStatus::kOptimal) << s.message;
    return presolve::substitutions_applied_for_testing() - before;
  };
  EXPECT_GE(run({{"algorithm", "dual-simplex"}}), 1);
  EXPECT_EQ(run({{"algorithm", "ipm"}}), 0);
  EXPECT_EQ(run({{"algorithm", "pdhg"}}), 0);
  EXPECT_LE(run({{"engine_race", "true"}, {"deterministic", "true"}}), 1);
  EXPECT_LE(run({{"algorithm", "auto"}}), 1);
}

TEST(FuzzSubstitution, KktInstancesWithEqualitiesKeepTheirOptimum) {
  // Active rows of a KKT construction become equalities: x* stays feasible and y* >= 0 is
  // still a valid multiplier, so the analytic optimum is unchanged.
  std::mt19937_64 rng(1006100601);
  const GeneratorConfig config = sparse_config();
  int agreed = 0, wrong = 0;
  std::vector<std::string> failures;
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
      equal[u] = activity == lp.b[u] ? 1 : 0;
    }
    GeneratedLp both;
    const Model model = with_equalities(lp, equal, &both);
    const Solution s = solve_with_substitution(model, true);
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
  std::cout << "\n=== 1500 KKT instances, active rows as equalities, substitution on ===\n"
            << "  agreed " << agreed << "\n  WRONG " << wrong << "\n";
  for (const std::string& f : failures)
    std::cout << "\n--- failing instance ---\n" << f << "\n";
  EXPECT_EQ(wrong, 0);
  EXPECT_GT(agreed, 1400);
}

TEST(FuzzSubstitution, RandomInstancesWithEqualitiesAgainstTheOracle) {
  // Unstructured instances, a third of the rows equalities: every status, against the exact
  // oracle, with the stage on.
  std::mt19937_64 rng(1006200602);
  const GeneratorConfig config = sparse_config();
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
    const Solution s = solve_with_substitution(model, true);
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
  std::cout << "\n=== 2000 random instances with equalities, substitution on ===\n"
            << "  agreed optimal " << optimal << "\n  agreed infeasible/unbounded " << other
            << "\n  oracle abstained " << skipped << "\n  MISMATCHED " << mismatched << "\n";
  for (const std::string& f : failures)
    std::cout << "\n--- failing instance ---\n" << f << "\n";
  EXPECT_EQ(mismatched, 0);
  // Equalities make most unstructured instances infeasible: 113 of 2000 came out optimal on
  // the first run, and the optimal ones are what the KKT test above covers in bulk.
  EXPECT_GT(optimal, 80);
  EXPECT_GT(other, 200);
}

}  // namespace
}  // namespace sankhya::oracle
