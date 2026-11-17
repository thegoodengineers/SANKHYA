// SPDX-License-Identifier: Apache-2.0
// SANKHYA - #1312: a node whose relaxation is integral only to the tolerance, and whose
// rounded point the incumbent test refuses, is not closed on that point.
//
// presolve_coefficient_tightening=true reported egout optimal at 612.29 (published 568.101)
// on seeds 0 and 2 and modglob above its optimum on every seed. The presolved model still
// held the optimum (the debug-solution check passed it); the search lost it. At the node
// holding it the LP put a binary at 3.3e-7 and a flow at 7.2e-7 on the row x - 3.06 y <= 0:
// integral to the 1e-6 tolerance, 7.2e-7 infeasible once y is rounded to 0 (#1187), so the
// incumbent test refused the point, and the node was closed anyway, subtree and optimum with
// it. The same can happen with the reduction off; the reduction's tighter rows made it
// happen on these two models.
//
// Three checks: a two-column model where the root LP is exactly that point; egout itself
// (tests/data/miplib3/egout.mps is MIPLIB 3's file, sha256 351ed046eb56e07662b3c2dba35800f5
// d2cee3fac18c1bd7fbc8dea4a0dcd8db, the one data/miplib3/manifest.json names) with the
// reduction on; and random fixed-charge MILPs with a big-M far above the flow's bound,
// solved with and without the reduction against the exact rational branch and bound.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options tree_only() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("mip_threads", 1);
  options.set_int("node_limit", 20000);
  // The tree is what is tested: no heuristic may find the optimum for it.
  options.set_bool("mip_heuristics", false);
  options.set_string("mip_heur_fj", "off");
  return options;
}

TEST(RefusedIntegralNode, IsBranchedOnNotClosed) {
  // min 2 y - x  s.t.  x - 1e7 y <= 0,  0 <= x <= 1,  y binary. The optimum is 0 (y = 0,
  // x = 0; y = 1, x = 1 costs 1). The root LP is x = 1, y = 1e-7, worth -1 + 2e-7: integral
  // to the tolerance, and with y rounded to 0 the row is violated by 1.
  Model model;
  model.col_cost = {2.0, -1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {1.0, 1.0};
  model.col_type = {VarType::kInteger, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {0.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, -1e7);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  ASSERT_EQ(model.validate(), "");
  Options options = tree_only();
  options.set_bool("presolve", false);  // the big-M row reaches the tree as written
  options.set_bool("enable_root_cuts", false);
  const Solution solved = solve(model, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, 0.0, 1e-9) << solved.message;
}

TEST(RefusedIntegralNode, EgoutWithCoefficientTighteningProvesThePublishedOptimum) {
  Model model;
  const std::string path = (std::filesystem::path(__FILE__).parent_path().parent_path() /
                            "data" / "miplib3" / "egout.mps")
                               .string();
  const io::ReadResult read = io::read_model(path, &model);
  ASSERT_TRUE(read.ok) << path << ": " << read.error;
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("mip_threads", 1);
  options.set_double("time_limit", 120.0);
  options.set_bool("presolve_coefficient_tightening", true);
  const Solution solved = solve(model, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, 568.101, 1e-3) << solved.message;  // the manifest's
}

/// Add x_c <= M y_b with M far above x_c's bound, x_c paid for and y_b costed: the LP then
/// wants y_b = x_c / M, below the integrality tolerance.
void add_a_huge_big_m_link(oracle::GeneratedLp* lp, std::mt19937_64& rng) {
  std::uniform_int_distribution<Index> pick(0, lp->num_cols - 1);
  std::uniform_int_distribution<std::int64_t> big(2'000'000, 20'000'000);
  std::uniform_int_distribution<std::int64_t> fixed_cost(1, 6);
  std::uniform_int_distribution<std::int64_t> reward(1, 5);
  const auto c = static_cast<std::size_t>(pick(rng));
  if (lp->upper[c] == oracle::kNoUpperBound) lp->upper[c] = 3;
  lp->c[c] = -reward(rng);
  if (!lp->integral.empty()) lp->integral[c] = 0;
  for (auto& row : lp->a) row.push_back(0);
  std::vector<std::int64_t> link(static_cast<std::size_t>(lp->num_cols) + 1, 0);
  link[c] = -1;  // A x >= b form: M y - x >= 0
  link.back() = big(rng);
  lp->a.push_back(link);
  lp->b.push_back(0);
  lp->c.push_back(fixed_cost(rng));
  lp->upper.push_back(1);
  if (!lp->integral.empty()) lp->integral.push_back(1);
  if (!lp->lower.empty()) lp->lower.push_back(0);
  ++lp->num_cols;
  ++lp->num_rows;
}

TEST(RefusedIntegralNode, HugeBigMMilpsAgreeWithTheExactOracleWithAndWithoutTightening) {
  std::mt19937_64 rng(1312);
  oracle::GeneratorConfig config;
  config.max_rows = 4;
  config.max_cols = 5;
  config.bounded_column_probability = 0.7;
  int compared = 0;
  // About 40% of the draws have an exact optimum (85 of the first 200 with libstdc++); the
  // rest are infeasible or unbounded. Draw 287 is #1319's: 4459767 x3 >= 4 once x0 = 4.
  for (int trial = 0; trial < 400; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 3 == 0 ? 1 : 0;
    }
    add_a_huge_big_m_link(&lp, rng);
    add_a_huge_big_m_link(&lp, rng);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal) continue;
    Model model = oracle::to_model(lp);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.integral[static_cast<std::size_t>(j)] != 0) {
        model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      }
    }
    const double expected = exact.objective.to_double();
    for (const bool tightening : {false, true}) {
      Options options = tree_only();
      options.set_bool("presolve_coefficient_tightening", tightening);
      const Solution got = solve(model, options);
      EXPECT_EQ(got.status, SolveStatus::kOptimal)
          << "trial " << trial << " tightening " << tightening << ": " << got.message << "\n"
          << lp.to_text();
      if (got.status != SolveStatus::kOptimal) continue;
      EXPECT_NEAR(got.objective, expected, 1e-6 * std::max(1.0, std::fabs(expected)))
          << "trial " << trial << " tightening " << tightening << "\n"
          << lp.to_text();
    }
    ++compared;
    if (::testing::Test::HasFailure()) return;
  }
  EXPECT_GE(compared, 120);  // the distributions differ by library
}

}  // namespace
}  // namespace sankhya
