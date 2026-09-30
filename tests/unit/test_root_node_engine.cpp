// SPDX-License-Identifier: Apache-2.0
// SANKHYA - which engine solves the root node LP, and on what budget (#803).
//
// The root is the one node LP with no basis to start from, and it used to be the cold
// primal simplex's alone. On MIPLIB tier 2 that primal failed both cvs16r128-89 (a stall of
// 1,001 degenerate iterations under Bland's rule) and ran14x18-disj-8 (phase 1 diverging
// through basis repairs), and the tree stopped at node 1 with numerical_error, while the dual
// simplex solves both relaxations. Those files are fetched, not tracked, so the tests here
// hold the routing and the budget on generated models:
//
//   - the root LP is the dual simplex's, and the primal runs only under
//     mip_node_engine=primal, where the two report the same root bound;
//   - a root LP the clock stops is reported as stopped, not handed to the primal simplex,
//     which under the old per-node budget got the whole time_limit again.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

/// Maximise the number of chosen binaries under packing rows with mixed signs, the shape of
/// cvs16r128-89 (every coefficient +1 or -1, rows of two to four columns, `<=` rows only):
/// dual degenerate throughout, the case the cold primal stalled on.
Model packing(Index columns, Index rows, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<Index> pick(0, columns - 1);
  std::uniform_int_distribution<int> width(2, 4);
  std::uniform_int_distribution<int> percent(0, 99);
  Model model;
  model.name = "packing";
  const auto n = static_cast<std::size_t>(columns);
  model.col_cost.assign(n, -1.0);
  model.col_lower.assign(n, 0.0);
  model.col_upper.assign(n, 1.0);
  model.col_type.assign(n, VarType::kInteger);
  model.matrix.reset(rows, columns);
  for (Index i = 0; i < rows; ++i) {
    const int k = width(rng);
    std::vector<Index> used;
    int positive = 0;
    while (static_cast<int>(used.size()) < k) {
      const Index j = pick(rng);
      bool seen = false;
      for (const Index u : used) seen = seen || u == j;
      if (seen) continue;
      used.push_back(j);
      const bool plus = percent(rng) < 55;
      positive += plus ? 1 : 0;
      model.matrix.add_entry(i, j, plus ? 1.0 : -1.0);
    }
    model.row_lower.push_back(-kInfinity);
    model.row_upper.push_back(static_cast<double>(positive > 0 ? positive - 1 : 0));
  }
  model.matrix.finalize();
  model.hessian.reset(columns, columns);
  model.hessian.finalize();
  return model;
}

Options root_only() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);  // the root the tree sees is the model's own
  options.set_bool("mip_symmetry", false);
  options.set_bool("enable_root_cuts", false);
  options.set_int("node_limit", 1);
  return options;
}

/// The search's log, captured.
std::string solve_logged(const Model& model, const Options& options, Solution* out,
                         LogLevel level = LogLevel::kInfo) {
  std::FILE* stream = std::tmpfile();
  EXPECT_NE(stream, nullptr);
  Logger logger(stream, level);
  *out = mip::solve_branch_and_bound(model, options, logger, nullptr);
  std::fflush(stream);
  std::rewind(stream);
  std::string text;
  char buffer[4096];
  while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) text += buffer;
  std::fclose(stream);
  return text;
}

/// The "Node LPs:" summary line's counts.
struct NodeLpCounts {
  long cold = -1;
  long cold_primal = -1;
  long fallbacks = -1;
};

NodeLpCounts node_lp_counts(const std::string& log) {
  static const std::regex line(
      R"(Node LPs: \d+ warm-started dual \(\d+ iterations\), (\d+) cold \(\d+ iterations, (\d+) )"
      R"(by the primal simplex\), (\d+) primal fallback)");
  std::smatch match;
  NodeLpCounts counts;
  if (std::regex_search(log, match, line)) {
    counts.cold = std::stol(match[1].str());
    counts.cold_primal = std::stol(match[2].str());
    counts.fallbacks = std::stol(match[3].str());
  }
  return counts;
}

TEST(RootNodeEngine, TheRootLpIsTheDualSimplexsAndThePrimalAgreesWithIt) {
  const Model model = packing(120, 160, 20260930);

  Solution dual;
  const std::string dual_log = solve_logged(model, root_only(), &dual);
  const NodeLpCounts by_dual = node_lp_counts(dual_log);
  ASSERT_GE(by_dual.cold, 1) << dual_log;
  EXPECT_EQ(by_dual.cold_primal, 0) << "the root LP went to the primal simplex\n" << dual_log;
  EXPECT_EQ(by_dual.fallbacks, 0) << dual_log;
  EXPECT_NE(dual.status, SolveStatus::kNumericalError) << dual.message;
  ASSERT_FALSE(std::isnan(dual.root_bound)) << dual.message;

  Options primal_options = root_only();
  primal_options.set_string("mip_node_engine", "primal");
  Solution primal;
  const std::string primal_log = solve_logged(model, primal_options, &primal);
  const NodeLpCounts by_primal = node_lp_counts(primal_log);
  ASSERT_GE(by_primal.cold, 1) << primal_log;
  EXPECT_EQ(by_primal.cold_primal, by_primal.cold) << primal_log;
  ASSERT_FALSE(std::isnan(primal.root_bound)) << primal.message;

  // Two engines, one LP: the same optimal value to the feasibility tolerances' order.
  EXPECT_NEAR(dual.root_bound, primal.root_bound,
              1e-6 * std::max(1.0, std::fabs(dual.root_bound)));
}

TEST(RootNodeEngine, ARootLpStoppedByTheClockIsNotHandedToThePrimal) {
  // Big enough that its root LP cannot finish in the budget on any machine this runs on;
  // if one ever does, the test says so rather than passing on nothing. A search stopped
  // with no incumbent returns before the "Node LPs:" summary, so the fallback is read from
  // the verbose line the node solve writes when it hands a node to the primal.
  const Model model = packing(12000, 16000, 803);
  Options options = root_only();
  options.set_int("node_limit", -1);
  options.set_double("time_limit", 0.2);
  Solution stopped;
  const std::string log = solve_logged(model, options, &stopped, LogLevel::kVerbose);
  // One node entered and the clock stopped it: the root LP was attempted and did not finish.
  if (stopped.status != SolveStatus::kTimeLimit || stopped.nodes != 1) {
    GTEST_SKIP() << "the root LP was not stopped by the clock here (status "
                 << to_string(stopped.status) << ", " << stopped.nodes << " nodes)";
  }
  EXPECT_EQ(log.find("re-solving cold with the primal simplex"), std::string::npos)
      << "a clock stop was retried with the primal\n"
      << log;
}

}  // namespace
}  // namespace sankhya
