// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a node LP that fails numerically (#733).
//
// gen-ip016 at mip_threads=8 under load once ended numerical_error: one node LP's phase 1
// stalled at 4.1e-7 against a 1e-7 tolerance, and the tree gave up the whole solve. Now the
// node keeps the bound it inherited from its parent: pruned when that meets the incumbent,
// otherwise set aside and counted in every bound reported. These tests inject the failure
// (fail_next_node_lps_for_testing) and hold the answer to the one rule that matters: never a
// wrong `optimal`, never a bound that cuts off the true optimum, never `infeasible`.

#include <cmath>
#include <cstdint>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "mip/node_lp_fault.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

/// 0-1 knapsack with a side row and fractional values, so the root does not close the tree.
Model knapsack(int items, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> weight(10, 60);
  Model model;
  model.sense = ObjSense::kMaximize;
  model.matrix.reset(2, items);
  double total = 0.0;
  double second = 0.0;
  for (int j = 0; j < items; ++j) {
    const int w = weight(rng);
    const int v = w + std::uniform_int_distribution<int>(-5, 12)(rng);
    const int s = std::uniform_int_distribution<int>(1, 9)(rng);
    model.col_cost.push_back(static_cast<double>(v) + 0.25 +
                             0.125 * static_cast<double>(j % 3));
    model.col_lower.push_back(0.0);
    model.col_upper.push_back(1.0);
    model.col_type.push_back(VarType::kInteger);
    model.matrix.add_entry(0, j, static_cast<double>(w));
    model.matrix.add_entry(1, j, static_cast<double>(s));
    total += w;
    second += s;
  }
  model.matrix.finalize();
  model.row_lower = {-kInfinity, -kInfinity};
  model.row_upper = {std::floor(total / 2.0), std::floor(second / 2.0)};
  model.hessian.reset(items, items);
  model.hessian.finalize();
  return model;
}

Options quiet(int threads) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("mip_threads", threads);
  return options;
}

/// Resets the injection however the test ends.
struct InjectFailures {
  explicit InjectFailures(long long count) { mip::fail_next_node_lps_for_testing = count; }
  ~InjectFailures() { mip::fail_next_node_lps_for_testing = 0; }
  InjectFailures(const InjectFailures&) = delete;
  InjectFailures& operator=(const InjectFailures&) = delete;
};

/// The answer under injected failures against the true optimum of a maximisation.
void expect_honest(const Solution& solved, double truth, const std::string& what) {
  ASSERT_NE(solved.status, SolveStatus::kInfeasible) << what << ": " << solved.message;
  if (solved.status == SolveStatus::kNumericalError) return;  // no point found: no claim
  ASSERT_TRUE(solved.status == SolveStatus::kOptimal || solved.status == SolveStatus::kFeasible)
      << what << ": " << solved.message;
  EXPECT_LE(solved.objective, truth + 1e-6) << what;   // a real point
  EXPECT_GE(solved.dual_bound, truth - 1e-6) << what;  // the bound still covers the optimum
  if (solved.status == SolveStatus::kOptimal) {
    EXPECT_NEAR(solved.objective, truth, 1e-6) << what << ": a wrong optimal";
  }
}

TEST(NodeLpFailure, EveryNodeLpFailingLeavesAnHonestFeasibleAnswer) {
  for (const std::uint32_t seed : {1u, 2u, 3u}) {
    const Model model = knapsack(24, seed);
    const Solution truth = solve(model, quiet(1));
    ASSERT_EQ(truth.status, SolveStatus::kOptimal) << truth.message;
    Solution solved;
    {
      const InjectFailures inject(1'000'000);
      solved = solve(model, quiet(1));
    }
    const std::string what = "seed " + std::to_string(seed);
    expect_honest(solved, truth.objective, what);
    // Before #733 this was numerical_error. With every child unsolved, the root's bound
    // stays the reported bound, so nothing below it is proved.
    EXPECT_NE(solved.status, SolveStatus::kNumericalError) << what << ": " << solved.message;
    if (solved.status == SolveStatus::kFeasible) {
      EXPECT_NE(solved.message.find("#733"), std::string::npos) << solved.message;
      EXPECT_EQ(solved.stopped_by, LimitReason::kNone);
    }
  }
}

TEST(NodeLpFailure, AFewFailuresNeverGiveAWrongOptimal) {
  for (const long long failures : {1LL, 3LL, 10LL}) {
    for (const std::uint32_t seed : {4u, 5u}) {
      const Model model = knapsack(24, seed);
      const Solution truth = solve(model, quiet(1));
      ASSERT_EQ(truth.status, SolveStatus::kOptimal) << truth.message;
      Solution solved;
      {
        const InjectFailures inject(failures);
        solved = solve(model, quiet(1));
      }
      expect_honest(solved, truth.objective,
                    "seed " + std::to_string(seed) + " failures " + std::to_string(failures));
    }
  }
}

TEST(NodeLpFailure, TheParallelTreeKeepsTheFailedNodesBound) {
  for (const std::uint32_t seed : {6u, 7u}) {
    const Model model = knapsack(30, seed);
    const Solution truth = solve(model, quiet(1));
    ASSERT_EQ(truth.status, SolveStatus::kOptimal) << truth.message;
    for (const long long failures : {5LL, 1'000'000LL}) {
      Solution solved;
      {
        const InjectFailures inject(failures);
        solved = solve(model, quiet(4));
      }
      expect_honest(
          solved, truth.objective,
          "4 threads seed " + std::to_string(seed) + " failures " + std::to_string(failures));
    }
  }
}

}  // namespace
}  // namespace sankhya
