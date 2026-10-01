// SPDX-License-Identifier: Apache-2.0
// SANKHYA - objective branching (#418).
//
// Splitting a node on the objective row instead of a column changes which tree the search
// explores and must not change its answer: a row bound applied to the wrong side, or one
// left behind when the search moved to another node, would prove the second-best answer
// optimal and nothing downstream could tell. So every model here is also solved by
// enumeration and the search must report exactly what enumeration says, with the option on,
// with cuts and conflicts on beside it, twice in deterministic mode, across a checkpoint and
// a resume, and in the parallel tree where the row is declined.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "support/temp_file.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Rows given densely, all columns binary.
Model binary_model(const std::vector<std::vector<double>>& rows,
                   const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                   const std::vector<double>& cost, bool maximize) {
  Model m;
  const auto n = cost.size();
  m.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  m.col_cost = cost;
  m.col_lower.assign(n, 0.0);
  m.col_upper.assign(n, 1.0);
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

/// Independent of everything under test: integral, inside the box, every row satisfied.
bool feasible(const Model& m, const std::vector<double>& x) {
  if (x.size() != static_cast<std::size_t>(m.num_cols())) return false;
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (std::fabs(x[j] - std::round(x[j])) > 1e-9) return false;
    if (x[j] < m.col_lower[j] - 1e-9 || x[j] > m.col_upper[j] + 1e-9) return false;
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    double activity = 0.0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      activity += m.matrix.at(i, j) * x[static_cast<std::size_t>(j)];
    }
    const auto u = static_cast<std::size_t>(i);
    if (activity < m.row_lower[u] - 1e-9 || activity > m.row_upper[u] + 1e-9) return false;
  }
  return true;
}

/// Every 0/1 assignment: whether any is feasible, and the best objective among them.
std::pair<bool, double> enumerate(const Model& m) {
  const auto n = static_cast<std::size_t>(m.num_cols());
  bool any = false;
  double best = 0.0;
  for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
    std::vector<double> x(n);
    for (std::size_t j = 0; j < n; ++j) x[j] = (mask >> j) & 1u ? 1.0 : 0.0;
    if (!feasible(m, x)) continue;
    double value = 0.0;
    for (std::size_t j = 0; j < n; ++j) value += m.col_cost[j] * x[j];
    if (!any || (m.sense == ObjSense::kMaximize ? value > best : value < best)) best = value;
    any = true;
  }
  return {any, best};
}

/// A random covering / packing model with integer costs of a common step, so the objective
/// is integral (#398) and the relaxation's value lands between two attainable values often.
Model random_model(std::mt19937& rng, int trial) {
  std::uniform_int_distribution<int> coefficient(0, 4);
  std::uniform_int_distribution<int> cost_value(1, 12);
  const std::size_t n = 8 + static_cast<std::size_t>(trial % 5);
  const double step = 1.0 + static_cast<double>(trial % 3);  // 1, 2 or 3
  std::vector<std::vector<double>> rows(4, std::vector<double>(n));
  std::vector<double> lower(4, -kInf);
  std::vector<double> upper(4, kInf);
  for (std::size_t i = 0; i < 4; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      rows[i][j] = coefficient(rng);
      sum += rows[i][j];
    }
    if (i % 2 == 0) {
      upper[i] = std::floor(sum / 2.0);
    } else {
      lower[i] = std::floor(sum / 4.0);
    }
  }
  std::vector<double> cost(n);
  for (double& c : cost) c = step * cost_value(rng);
  return binary_model(rows, lower, upper, cost, trial % 2 == 1);
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_bool("mip_objective_branching", true);
  return options;
}

void expect_enumerated(const Model& m, const Solution& solved, int trial, const char* what) {
  const auto [any, best] = enumerate(m);
  if (!any) {
    EXPECT_EQ(solved.status, SolveStatus::kInfeasible)
        << what << " trial " << trial << ": " << solved.message;
    return;
  }
  ASSERT_EQ(solved.status, SolveStatus::kOptimal)
      << what << " trial " << trial << ": " << solved.message;
  EXPECT_NEAR(solved.objective, best, 1e-6) << what << " trial " << trial;
  EXPECT_TRUE(feasible(m, solved.col_value)) << what << " trial " << trial;
}

TEST(ObjectiveBranching, TheOptionIsRegisteredAndOffByDefault) {
  const Options options;
  EXPECT_FALSE(options.get_bool("mip_objective_branching"));
}

TEST(ObjectiveBranching, SplittingTheObjectiveRowKeepsTheEnumeratedOptimum) {
  std::mt19937 rng(4181);
  Count splits = 0;
  for (int trial = 0; trial < 150; ++trial) {
    const Model m = random_model(rng, trial);
    const Solution solved = solve(m, quiet());
    expect_enumerated(m, solved, trial, "objective");
    splits += solved.objective_branches;
  }
  EXPECT_GT(splits, 0) << "no node was split on the objective row, so the path was not "
                          "exercised";
}

TEST(ObjectiveBranching, NothingHappensWithoutADetectedStep) {
  // The row is appended only when #398 found the objective's step; with the detection off
  // the option is inert, and the answer is still the enumerated one.
  std::mt19937 rng(41810);
  for (int trial = 0; trial < 40; ++trial) {
    const Model m = random_model(rng, trial);
    Options options = quiet();
    options.set_bool("mip_objective_integrality", false);
    const Solution solved = solve(m, options);
    expect_enumerated(m, solved, trial, "no step");
    EXPECT_EQ(solved.objective_branches, 0) << "trial " << trial;
  }
}

TEST(ObjectiveBranching, CutRowsConflictsAndHeuristicsSitBesideTheObjectiveRow) {
  // Cut rows are appended after the objective row and the pool grows and shrinks below it;
  // a conflict is never learnt from a chain that holds a row decision; the dives unwind the
  // row bound like any other. None of that may move the answer.
  std::mt19937 rng(418100);
  Count splits = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const Model m = random_model(rng, trial);
    Options options = quiet();
    options.set_bool("enable_root_cuts", true);
    options.set_int("tree_cut_depth", 3);
    options.set_bool("conflict_analysis", true);
    options.set_bool("mip_heuristics", true);
    const Solution solved = solve(m, options);
    expect_enumerated(m, solved, trial, "cuts+conflicts");
    splits += solved.objective_branches;
  }
  EXPECT_GT(splits, 0) << "no node was split on the objective row with cuts on";
}

TEST(ObjectiveBranching, DeterministicRunsRepeat) {
  std::mt19937 rng(4181000);
  const Model m = random_model(rng, 4);
  Options options = quiet();
  options.set_bool("deterministic", true);
  options.set_bool("mip_heuristics", true);
  const Solution first = solve(m, options);
  const Solution second = solve(m, options);
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  EXPECT_EQ(second.status, first.status);
  EXPECT_EQ(second.objective, first.objective);
  EXPECT_EQ(second.nodes, first.nodes);
  EXPECT_EQ(second.objective_branches, first.objective_branches);
  EXPECT_EQ(second.col_value, first.col_value);
}

TEST(ObjectiveBranching, AResumedSearchReachesTheEnumeratedOptimum) {
  // A row decision in an open node's chain is written to the checkpoint as a domain change
  // on the row's logical index and must be admitted and re-applied by the resume.
  std::mt19937 rng(41810000);
  int resumed_short = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Model m = random_model(rng, trial);
    const Solution whole = solve(m, quiet());
    if (whole.status != SolveStatus::kOptimal || whole.nodes < 12) continue;

    testing::TempFile file("", ".chk");
    Options first = quiet();
    first.set_string("checkpoint", file.path());
    first.set_int("node_limit", std::max<Count>(2, whole.nodes / 4));
    const Solution stopped = solve(m, first);
    if (stopped.status == SolveStatus::kOptimal) continue;  // closed early: nothing to resume
    ++resumed_short;

    Options second = quiet();
    second.set_string("resume", file.path());
    const Solution resumed = solve(m, second);
    ASSERT_EQ(resumed.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << resumed.message;
    expect_enumerated(m, resumed, trial, "resume");
    EXPECT_GE(resumed.nodes, stopped.nodes) << "the node count continues from the checkpoint";
  }
  EXPECT_GT(resumed_short, 2) << "too few trials left the resumed search anything to do";
}

TEST(ObjectiveBranching, TheParallelTreeDeclinesTheRow) {
  // A worker shares the driver's model and scaling (#222), whose row count the appended row
  // would not match, so a parallel search never splits on the objective; the answer must
  // still be the enumerated one.
  std::mt19937 rng(418100000);
  for (int trial = 0; trial < 20; ++trial) {
    const Model m = random_model(rng, trial);
    Options options = quiet();
    options.set_int("mip_threads", 2);
    const Solution solved = solve(m, options);
    expect_enumerated(m, solved, trial, "parallel");
    EXPECT_EQ(solved.objective_branches, 0) << "trial " << trial;
  }
}

}  // namespace
}  // namespace sankhya::mip
