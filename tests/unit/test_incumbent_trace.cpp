// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the incumbent trace a MILP solve reports (#504).
//
// bench/runners/miplib.py reads the time to first feasible and Berthold's primal integral
// ("Measuring the impact of primal heuristics", Oper. Res. Lett. 41, 2013) off
// Solution::incumbent_trace, through the stats JSON. A trace that disagrees with the answer
// would make both numbers fiction, so these tests hold it to the answer: every event an
// improvement in the model's own sense, on a clock that does not run backwards, and the last
// one the objective the solve reports.

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

/// A 0-1 knapsack, maximised: the search has room to improve on its first point.
Model knapsack(ObjSense sense) {
  const double value[] = {10, 13, 7, 8, 9, 11, 6, 12, 5, 14, 4, 9};
  const double weight[] = {5, 7, 4, 4, 5, 6, 3, 7, 3, 8, 2, 5};
  constexpr Index n = 12;
  Model model;
  model.name = "knapsack";
  model.sense = sense;
  model.matrix.reset(1, n);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_names.push_back("x" + std::to_string(j));
    // Minimising the negated values is the same knapsack, so both senses have one optimum.
    model.col_cost.push_back(sense == ObjSense::kMaximize ? value[u] : -value[u]);
    model.col_lower.push_back(0.0);
    model.col_upper.push_back(1.0);
    model.col_type.push_back(VarType::kInteger);
    model.matrix.add_entry(0, j, weight[u]);
  }
  model.matrix.finalize();
  model.row_names = {"capacity"};
  model.row_lower = {-kInfinity};
  model.row_upper = {23.0};
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

void expect_consistent(const Solution& s, ObjSense sense) {
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  ASSERT_FALSE(s.incumbent_trace.empty()) << "a solved MILP must have found its incumbent";
  double previous_time = 0.0;
  for (std::size_t k = 0; k < s.incumbent_trace.size(); ++k) {
    const Solution::IncumbentEvent& event = s.incumbent_trace[k];
    EXPECT_GE(event.seconds, previous_time) << "event " << k << " is earlier than the last";
    previous_time = event.seconds;
    if (k > 0) {
      const double before = s.incumbent_trace[k - 1].objective;
      if (sense == ObjSense::kMaximize) {
        EXPECT_GT(event.objective, before) << "event " << k << " is not an improvement";
      } else {
        EXPECT_LT(event.objective, before) << "event " << k << " is not an improvement";
      }
    }
  }
  EXPECT_NEAR(s.incumbent_trace.back().objective, s.objective,
              1e-9 * std::max(1.0, std::fabs(s.objective)))
      << "the last incumbent is not the answer";
}

TEST(IncumbentTrace, MaximiseImprovesUpwardsAndEndsAtTheAnswer) {
  const Solution s = solve(knapsack(ObjSense::kMaximize), quiet());
  expect_consistent(s, ObjSense::kMaximize);
}

TEST(IncumbentTrace, MinimiseImprovesDownwardsAndEndsAtTheAnswer) {
  const Solution s = solve(knapsack(ObjSense::kMinimize), quiet());
  expect_consistent(s, ObjSense::kMinimize);
}

TEST(IncumbentTrace, WithoutPresolveTheSameContractHolds) {
  Options options = quiet();
  options.set_bool("presolve", false);
  const Solution s = solve(knapsack(ObjSense::kMaximize), options);
  expect_consistent(s, ObjSense::kMaximize);
}

TEST(IncumbentTrace, AnLpHasNone) {
  Model model = knapsack(ObjSense::kMaximize);
  for (VarType& type : model.col_type) type = VarType::kContinuous;
  const Solution s = solve(model, quiet());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_TRUE(s.incumbent_trace.empty());
}

TEST(IncumbentTrace, TheStatsJsonCarriesItAsSecondsObjectivePairs) {
  const Model model = knapsack(ObjSense::kMaximize);
  const Solution s = solve(model, quiet());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  testing::TempFile file("", ".json");
  std::string error;
  ASSERT_TRUE(io::write_stats_json(file.path(), model, s, &error)) << error;
  std::ifstream in(file.path());
  std::stringstream text;
  text << in.rdbuf();
  const nlohmann::json blob = nlohmann::json::parse(text.str());
  const nlohmann::json& trace = blob.at("effort").at("incumbent_trace");
  ASSERT_TRUE(trace.is_array());
  ASSERT_EQ(trace.size(), s.incumbent_trace.size());
  for (std::size_t k = 0; k < trace.size(); ++k) {
    ASSERT_EQ(trace[k].size(), 2U);
    EXPECT_DOUBLE_EQ(trace[k][0].get<double>(), s.incumbent_trace[k].seconds);
    EXPECT_DOUBLE_EQ(trace[k][1].get<double>(), s.incumbent_trace[k].objective);
  }
}

}  // namespace
}  // namespace sankhya
