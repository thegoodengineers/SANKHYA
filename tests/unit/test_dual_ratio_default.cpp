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

/// A model whose search must branch below the root, so that node LPs run: integer a in
/// [0, 5] and continuous b in [-10, 10] with a + b >= 2.5 and a - b >= 0.5 (so a >= 1.5,
/// which no single row shows), minimising a; and a c-block of three integer c in [0, 1]
/// and a continuous s >= 0 with 2 c1 + 2 c2 + 2 c3 - s <= 3, maximising the c at a cost on
/// s, fractional wherever it is branched. The same model #502's tests branch on.
/// Continuous throughout when `integer` is false.
Model branching_model(bool integer) {
  Model m;
  m.col_cost = {1.0, 0.0, -10.0, -10.0, -10.0, 7.0};
  m.col_lower = {0.0, -10.0, 0.0, 0.0, 0.0, 0.0};
  m.col_upper = {5.0, 10.0, 1.0, 1.0, 1.0, 100.0};
  m.col_type = {VarType::kInteger, VarType::kContinuous, VarType::kInteger,
                VarType::kInteger, VarType::kInteger,    VarType::kContinuous};
  if (!integer) m.col_type.assign(6, VarType::kContinuous);
  m.matrix.reset(3, 6);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(1, 0, 1.0);
  m.matrix.add_entry(1, 1, -1.0);
  for (Index j = 2; j <= 4; ++j) m.matrix.add_entry(2, j, 2.0);
  m.matrix.add_entry(2, 5, -1.0);
  m.row_lower = {2.5, 0.5, -kInfinity};
  m.row_upper = {kInfinity, kInfinity, 3.0};
  m.matrix.finalize();
  m.hessian.reset(6, 6);
  m.hessian.finalize();
  EXPECT_EQ(m.validate(), "");
  return m;
}

/// Presolve, root cuts and objective integrality off: each would close these small models
/// before a node LP ran, and a test of the node LPs' rule needs node LPs.
Options bare() {
  Options o;
  o.set_bool("presolve", false);
  o.set_bool("enable_root_cuts", false);
  o.set_bool("mip_objective_integrality", false);
  return o;
}

struct Profiled {
  Solution solution;
  nlohmann::json counters;
};

Profiled solve_profiled(const Model& model, Options options) {
  testing::TempFile file("", ".json");
  options.set_bool("log_to_console", false);
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
  const Profiled lp = solve_profiled(branching_model(false), bare());
  ASSERT_EQ(lp.solution.status, SolveStatus::kOptimal) << lp.solution.message;
  EXPECT_GT(counter(lp, kHarris), 0) << lp.counters.dump(2);
  EXPECT_EQ(counter(lp, kTextbook), 0) << lp.counters.dump(2);
}

TEST(DualRatioDefault, BranchAndBoundLpsKeepTheTextbookRule) {
  const Profiled mip = solve_profiled(branching_model(true), bare());
  ASSERT_EQ(mip.solution.status, SolveStatus::kOptimal) << mip.solution.message;
  ASSERT_GT(mip.solution.nodes, 1) << "the test needs the search to branch";
  EXPECT_GT(counter(mip, kTextbook), 0) << mip.counters.dump(2);
  EXPECT_EQ(counter(mip, kHarris), 0) << mip.counters.dump(2);
}

TEST(DualRatioDefault, AnExplicitRuleAppliesToBoth) {
  Options harris = bare();
  harris.set_string("dual_ratio_test", "harris");
  const Profiled mip = solve_profiled(branching_model(true), harris);
  ASSERT_EQ(mip.solution.status, SolveStatus::kOptimal) << mip.solution.message;
  EXPECT_GT(counter(mip, kHarris), 0) << mip.counters.dump(2);
  EXPECT_EQ(counter(mip, kTextbook), 0) << mip.counters.dump(2);

  Options textbook = bare();
  textbook.set_string("dual_ratio_test", "textbook");
  const Profiled lp = solve_profiled(branching_model(false), textbook);
  ASSERT_EQ(lp.solution.status, SolveStatus::kOptimal) << lp.solution.message;
  EXPECT_GT(counter(lp, kTextbook), 0) << lp.counters.dump(2);
  EXPECT_EQ(counter(lp, kHarris), 0) << lp.counters.dump(2);

  // And the two defaults reach the same optimum as the explicit rules.
  EXPECT_NEAR(solve_profiled(branching_model(false), bare()).solution.objective,
              lp.solution.objective, 1e-9);
  EXPECT_NEAR(solve_profiled(branching_model(true), bare()).solution.objective,
              mip.solution.objective, 1e-9);
}

}  // namespace
}  // namespace sankhya
