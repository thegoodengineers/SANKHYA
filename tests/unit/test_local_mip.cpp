// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for the Local-MIP improvement heuristic (#507).
//
// Lin, Zou and Cai, "Local-MIP: efficient local search for mixed integer programming",
// CP 2024, LIPIcs 307.
//
// Each test builds a tiny MILP whose optimal value is provably better than the initial
// feasible point, hands that point to local_mip_improve(), and asserts that the incumbent
// strictly improved. The models are chosen so that a sequence of lift moves alone is
// sufficient to reach the optimum from the starting point.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <regex>
#include <string>
#include <vector>

#include "mip/heuristics.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Build a dense all-integer model with all columns in [lower, upper].
Model make_model(const std::vector<std::vector<double>>& rows,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                 const std::vector<double>& cost, double lower, double upper) {
  Model m;
  const auto n = cost.size();
  m.col_cost = cost;
  m.col_lower.assign(n, lower);
  m.col_upper.assign(n, upper);
  m.col_type.assign(n, VarType::kInteger);
  m.matrix.reset(static_cast<Index>(rows.size()), static_cast<Index>(n));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i][j] != 0.0) {
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), rows[i][j]);
      }
    }
  }
  m.matrix.finalize();
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

// =============================================================================
// Test: single-variable lift
// =============================================================================
// min -x,  x integer in [0, 5],  no constraints.
// Initial incumbent: x = 0 (objective 0). Optimal: x = 5 (objective -5).
// The lift move increases x by 1 each iteration (c = -1 < 0 → increase).
TEST(LocalMip, SingleVariableLift) {
  // No rows needed.
  Model m = make_model({}, {}, {}, {-1.0}, 0.0, 5.0);

  Options opts;
  Logger quiet(nullptr);

  Solution incumbent;
  incumbent.col_value = {0.0};
  incumbent.objective = 0.0;  // -1 * 0 = 0

  const bool got_better = local_mip_improve(m, opts, incumbent, quiet);

  EXPECT_TRUE(got_better);
  EXPECT_LT(incumbent.objective, 0.0 - 1e-9);
  // Best reachable: x = 5, obj = -5.
  EXPECT_NEAR(incumbent.col_value[0], 5.0, 1e-9);
  EXPECT_NEAR(incumbent.objective, -5.0, 1e-9);
}

// =============================================================================
// Test: two-variable lift with a binding upper bound on a row
// =============================================================================
// min -x1 - x2,  x1 + x2 <= 4,  0 <= x1, x2 <= 5 integer.
// Initial incumbent: (0, 0), objective 0. Optimal: any (x1,x2) with x1+x2 = 4, e.g. (2,2).
// Lift moves increase x1 and x2 alternately until x1 + x2 = 4.
TEST(LocalMip, TwoVariableLiftWithRowBound) {
  // One row: x1 + x2 <= 4.
  Model m = make_model({{1.0, 1.0}}, {-kInf}, {4.0}, {-1.0, -1.0}, 0.0, 5.0);

  Options opts;
  Logger quiet(nullptr);

  Solution incumbent;
  incumbent.col_value = {0.0, 0.0};
  incumbent.objective = 0.0;

  const bool got_better = local_mip_improve(m, opts, incumbent, quiet);

  EXPECT_TRUE(got_better);
  EXPECT_LT(incumbent.objective, 0.0 - 1e-9);
  // The sum x1 + x2 must be <= 4 at optimum.
  const double sum = incumbent.col_value[0] + incumbent.col_value[1];
  EXPECT_LE(sum, 4.0 + 1e-9);
  // And the objective must be at most -3 (at least as good as (1,2) or (2,1)).
  EXPECT_LE(incumbent.objective, -3.0 + 1e-9);
}

// =============================================================================
// Test: minimisation with a positive cost — decrease move
// =============================================================================
// min x,  x integer in [0, 5],  no constraints.
// Initial incumbent: x = 5 (objective 5). Optimal: x = 0 (objective 0).
// c = 1 > 0 so the lift move decreases x.
TEST(LocalMip, DecreaseMove) {
  Model m = make_model({}, {}, {}, {1.0}, 0.0, 5.0);

  Options opts;
  Logger quiet(nullptr);

  Solution incumbent;
  incumbent.col_value = {5.0};
  incumbent.objective = 5.0;

  const bool got_better = local_mip_improve(m, opts, incumbent, quiet);

  EXPECT_TRUE(got_better);
  EXPECT_LT(incumbent.objective, 5.0 - 1e-9);
  EXPECT_NEAR(incumbent.col_value[0], 0.0, 1e-9);
  EXPECT_NEAR(incumbent.objective, 0.0, 1e-9);
}

// =============================================================================
// Test: already optimal — no improvement
// =============================================================================
// min x,  x integer in [0, 5].  Incumbent at x = 0 (optimal): must NOT be changed.
TEST(LocalMip, AlreadyOptimal) {
  Model m = make_model({}, {}, {}, {1.0}, 0.0, 5.0);

  Options opts;
  Logger quiet(nullptr);

  Solution incumbent;
  incumbent.col_value = {0.0};
  incumbent.objective = 0.0;

  const bool got_better = local_mip_improve(m, opts, incumbent, quiet);

  EXPECT_FALSE(got_better);
  EXPECT_NEAR(incumbent.objective, 0.0, 1e-9);
}

/// End to end through the search (review of #632): a maximize 0-1 knapsack, strongly
/// correlated. With mip_local_mip on the optimum is the same as with it off, and the
/// heuristic runs at most once per incumbent, not at every node.
TEST(LocalMip, InTheSearchItRunsPerIncumbentAndKeepsTheOptimum) {
  constexpr int kItems = 24;
  Model model;
  model.sense = ObjSense::kMaximize;
  model.resize_columns(kItems);
  model.resize_rows(1);
  model.row_lower = {-kInfinity};
  model.matrix.reset(1, kItems);
  double capacity = 0.0;
  for (int j = 0; j < kItems; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double weight = 10.0 + static_cast<double>((j * 37) % 23);
    model.col_cost[u] = weight + 5.0;
    model.col_upper[u] = 1.0;
    model.col_type[u] = VarType::kInteger;
    model.matrix.add_entry(0, j, weight);
    capacity += weight;
  }
  model.row_upper = {std::floor(capacity / 2.0) + 0.5};
  model.matrix.finalize();

  const auto run = [&model](bool local_mip, std::string* log) {
    Options options;
    options.set_bool("presolve", false);
    options.set_bool("log_to_console", true);
    options.set_bool("mip_local_mip", local_mip);
    options.set_int("mip_threads", 1);  // the incumbent trace is sequential-only
    ::testing::internal::CaptureStdout();
    const Solution s = solve(model, options);
    std::fflush(stdout);
    *log = ::testing::internal::GetCapturedStdout();
    return s;
  };
  std::string off_log;
  std::string on_log;
  const Solution off = run(false, &off_log);
  const Solution on = run(true, &on_log);
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, off.objective, 1e-6);
  std::smatch calls;
  ASSERT_TRUE(std::regex_search(on_log, calls, std::regex(R"(local MIP\s+calls\s+(\d+))")))
      << "the search never ran Local-MIP";
  const long long ran = std::stoll(calls[1].str());
  EXPECT_GE(ran, 1);
  // It starts once from each incumbent, so it cannot run more often than the incumbent
  // changed; the trace records every change (#504).
  EXPECT_LE(ran, static_cast<long long>(on.incumbent_trace.size()))
      << "Local-MIP ran " << ran << " times for " << on.incumbent_trace.size()
      << " incumbents: it should only run when the incumbent changes";
}

}  // namespace
}  // namespace sankhya::mip
