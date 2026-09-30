// SPDX-License-Identifier: Apache-2.0
// Scenarios (#752): every scenario of a warm-started batch agrees with a separate cold solve
// of the same edited model to tol::kScenarioAgreement, and nothing unverified is called
// optimal.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/scenarios.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

namespace sc = scenarios;

Model read(const std::string& relative) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
          .string();
  const io::ReadResult result = io::read_model(path, &model);
  EXPECT_TRUE(result.ok) << path << ": " << result.error;
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// Cost scenarios, right-hand-side scenarios and mixed ones, spread over the model: every
/// fifth column's cost scaled, every row with a finite side moved by a few per cent.
std::vector<sc::Scenario> spread(const Model& model) {
  std::vector<sc::Scenario> out;
  for (int k = 0; k < 6; ++k) {
    sc::Scenario s;
    s.name = "s" + std::to_string(k);
    const double factor = 1.0 + 0.03 * (k - 2);
    if (k % 3 != 1) {
      for (Index j = k; j < model.num_cols(); j += 5) {
        s.overrides.push_back(
            {sc::Target::kCost, j, model.col_cost[static_cast<std::size_t>(j)] * factor});
      }
    }
    if (k % 3 != 0) {
      for (Index i = k; i < model.num_rows(); i += 4) {
        const auto u = static_cast<std::size_t>(i);
        const double side =
            is_finite_bound(model.row_upper[u]) ? model.row_upper[u] : model.row_lower[u];
        if (!is_finite_bound(side) || side == 0.0) continue;
        s.overrides.push_back({sc::Target::kRhs, i, side * factor});
      }
    }
    out.push_back(s);
  }
  return out;
}

void expect_agreement(const std::string& relative) {
  const Model model = read(relative);
  const std::vector<sc::Scenario> scenarios = spread(model);
  const sc::BatchReport report = sc::solve_all(model, scenarios, quiet());
  ASSERT_EQ(report.base.status, SolveStatus::kOptimal) << relative;
  ASSERT_EQ(report.results.size(), scenarios.size());
  for (std::size_t k = 0; k < scenarios.size(); ++k) {
    const sc::ScenarioResult& r = report.results[k];
    const Solution single = solve(sc::apply(model, scenarios[k]), quiet());
    EXPECT_TRUE(r.verified) << relative << " " << scenarios[k].name << ": " << r.verdict;
    ASSERT_EQ(r.solution.status, single.status) << relative << " " << scenarios[k].name;
    if (single.status != SolveStatus::kOptimal) continue;
    const double rel = std::fabs(r.solution.objective - single.objective) /
                       std::max(1.0, std::fabs(single.objective));
    EXPECT_LE(rel, tol::kScenarioAgreement) << relative << " " << scenarios[k].name;
    // The safe bound, when the duals prove a finite one, is on the right side of the
    // optimum. It may be absent: a reduced cost a rounding error the wrong side of zero on a
    // column with no finite bound on that side proves nothing (safe_bound.hpp), and NaN is
    // how that is reported, never a number.
    if (std::isnan(r.safe_bound)) continue;
    const double slack = 1e-9 * std::max(1.0, std::fabs(single.objective));
    if (model.sense == ObjSense::kMaximize) {
      EXPECT_GE(r.safe_bound, single.objective - slack);
    } else {
      EXPECT_LE(r.safe_bound, single.objective + slack);
    }
  }
}

TEST(Scenarios, AgreeWithSeparateSolvesOnTheDemoBlend) {
  expect_agreement("demo/crude_blend.mps");
}

TEST(Scenarios, AgreeWithSeparateSolvesOnNetlib) {
  for (const char* name : {"afiro", "adlittle", "share2b", "sc105"}) {
    expect_agreement(std::string("data/netlib/") + name + ".mps");
  }
}

TEST(Scenarios, AnInfeasibleScenarioIsProvedAndNeverOptimal) {
  // The blend's diesel pool lands on exactly 40 from at most 120 of crude whose best diesel
  // yield is 0.45: 60 is out of reach.
  const Model model = read("demo/crude_blend.mps");
  const auto diesel =
      static_cast<Index>(std::find(model.row_names.begin(), model.row_names.end(), "DIESEL") -
                         model.row_names.begin());
  ASSERT_LT(diesel, model.num_rows());
  const sc::BatchReport report =
      sc::solve_all(model, {{"short", {{sc::Target::kRhs, diesel, 60.0}}}}, quiet());
  const sc::ScenarioResult& r = report.results[0];
  EXPECT_NE(r.solution.status, SolveStatus::kOptimal);
  if (r.verified) {
    EXPECT_EQ(r.solution.status, SolveStatus::kInfeasible);
    EXPECT_EQ(r.verdict, "farkas");
  }
}

TEST(Scenarios, AMilpIsRefusedNotAnsweredWithItsRelaxation) {
  const Model model = read("demo/blend_milp.mps");
  const sc::BatchReport report = sc::solve_all(model, {{"a", {}}}, quiet());
  EXPECT_FALSE(report.results[0].verified);
  EXPECT_NE(report.results[0].verdict.find("LP only"), std::string::npos);
}

}  // namespace
}  // namespace sankhya
