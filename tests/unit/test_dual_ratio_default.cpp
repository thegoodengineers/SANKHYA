// SPDX-License-Identifier: Apache-2.0
// SANKHYA - which dual ratio test a solve runs under by default (#712).
//
// dual_ratio_test=auto is Harris for an LP solve and the textbook rule for every LP inside
// branch and bound: Harris became the LP default on its Netlib A/B, and the MIP trees keep
// the rule they were measured with. An explicit harris or textbook applies to both. The rule
// each dual solve actually ran under is read from the profiler's counters (profile_out), so
// these tests see what the engine did rather than what the option says.

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

constexpr const char* kHarris = "dual solves, harris ratio test";
constexpr const char* kTextbook = "dual solves, textbook ratio test";

/// max 5 x1 + 4 x2 + 3 x3 over three knapsack rows, x in [0, 10]; integer when `integer`.
/// The relaxation's optimum is fractional, so the MIP has to branch.
Model knapsack(bool integer) {
  Model m;
  m.sense = ObjSense::kMaximize;
  m.col_cost = {5.0, 4.0, 3.0};
  m.col_lower.assign(3, 0.0);
  m.col_upper.assign(3, 10.0);
  m.col_type.assign(3, integer ? VarType::kInteger : VarType::kContinuous);
  const double a[3][3] = {{2.0, 3.0, 1.0}, {4.0, 1.0, 2.0}, {3.0, 4.0, 2.0}};
  m.matrix.reset(3, 3);
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) m.matrix.add_entry(i, j, a[i][j]);
  }
  m.row_lower.assign(3, -kInfinity);
  m.row_upper = {5.5, 11.0, 8.5};
  m.matrix.finalize();
  m.hessian.reset(3, 3);
  m.hessian.finalize();
  EXPECT_EQ(m.validate(), "");
  return m;
}

struct Profiled {
  Solution solution;
  nlohmann::json counters;
};

Profiled solve_profiled(const Model& model, Options options) {
  testing::TempFile file("", ".json");
  options.set_bool("log_to_console", false);
  // Presolve would solve models this small outright, and no simplex would run at all.
  options.set_bool("presolve", false);
  options.set_string("profile", "basic");
  options.set_string("profile_out", file.path());
  Profiled out;
  out.solution = solve(model, options);
  std::ifstream in(file.path());
  std::stringstream text;
  text << in.rdbuf();
  EXPECT_FALSE(text.str().empty()) << "profile_out was not written";
  if (!text.str().empty()) out.counters = nlohmann::json::parse(text.str())["counters"];
  return out;
}

std::int64_t counter(const Profiled& p, const char* name) {
  return p.counters.contains(name) ? p.counters[name].get<std::int64_t>() : 0;
}

TEST(DualRatioDefault, AnLpSolveRunsUnderHarris) {
  const Profiled lp = solve_profiled(knapsack(false), Options());
  ASSERT_EQ(lp.solution.status, SolveStatus::kOptimal) << lp.solution.message;
  EXPECT_GT(counter(lp, kHarris), 0) << lp.counters.dump(2);
  EXPECT_EQ(counter(lp, kTextbook), 0) << lp.counters.dump(2);
}

TEST(DualRatioDefault, BranchAndBoundLpsKeepTheTextbookRule) {
  const Profiled mip = solve_profiled(knapsack(true), Options());
  ASSERT_EQ(mip.solution.status, SolveStatus::kOptimal) << mip.solution.message;
  EXPECT_GT(counter(mip, kTextbook), 0) << mip.counters.dump(2);
  EXPECT_EQ(counter(mip, kHarris), 0) << mip.counters.dump(2);
}

TEST(DualRatioDefault, AnExplicitRuleAppliesToBoth) {
  Options harris;
  harris.set_string("dual_ratio_test", "harris");
  const Profiled mip = solve_profiled(knapsack(true), harris);
  ASSERT_EQ(mip.solution.status, SolveStatus::kOptimal) << mip.solution.message;
  EXPECT_GT(counter(mip, kHarris), 0) << mip.counters.dump(2);
  EXPECT_EQ(counter(mip, kTextbook), 0) << mip.counters.dump(2);

  Options textbook;
  textbook.set_string("dual_ratio_test", "textbook");
  const Profiled lp = solve_profiled(knapsack(false), textbook);
  ASSERT_EQ(lp.solution.status, SolveStatus::kOptimal) << lp.solution.message;
  EXPECT_GT(counter(lp, kTextbook), 0) << lp.counters.dump(2);
  EXPECT_EQ(counter(lp, kHarris), 0) << lp.counters.dump(2);

  // And the two defaults reach the same optimum as the explicit rules.
  EXPECT_NEAR(solve_profiled(knapsack(false), Options()).solution.objective,
              lp.solution.objective, 1e-9);
  EXPECT_NEAR(solve_profiled(knapsack(true), Options()).solution.objective,
              mip.solution.objective, 1e-9);
}

}  // namespace
}  // namespace sankhya
