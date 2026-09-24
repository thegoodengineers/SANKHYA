// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the root separation loop (#495).
//
// Three things: off is the single round exactly as before (the same bound after cuts and
// the same answer); on, the loop logs its rounds and the root bound after cuts is never
// below the single round's (every round only appends rows); and a search with the loop on,
// every family on, still reaches the exact optimum of the rational branch and bound.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "mip/cut_selection.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

/// Binary knapsack and covering rows, `n` columns: the structure covers, cliques and
/// zero-half cuts are built for.
oracle::GeneratedLp binary_instance(std::mt19937_64& rng, Index n) {
  std::uniform_int_distribution<Index> height(2, 4);
  std::uniform_int_distribution<std::int64_t> weight(1, 9);
  std::uniform_int_distribution<std::int64_t> profit(1, 40);
  std::uniform_int_distribution<int> percent(0, 99);
  oracle::GeneratedLp lp;
  lp.num_cols = n;
  const Index knapsacks = height(rng);
  const Index covers = height(rng) - 1;
  lp.num_rows = knapsacks + covers;
  const auto u = static_cast<std::size_t>(n);
  lp.integral.assign(u, 1);
  lp.upper.assign(u, 1);
  lp.c.resize(u);
  for (std::size_t j = 0; j < u; ++j) lp.c[j] = -profit(rng);  // maximise profit
  for (Index i = 0; i < knapsacks; ++i) {
    std::vector<std::int64_t> row(u, 0);
    std::int64_t sum = 0;
    for (std::size_t j = 0; j < u; ++j) {
      if (percent(rng) < 70) row[j] = weight(rng);
      sum += row[j];
    }
    for (std::int64_t& a : row) a = -a;  // sum a x <= b, as -a x >= -b
    lp.a.push_back(row);
    lp.b.push_back(-std::max<std::int64_t>(1, sum / 2));
  }
  for (Index i = 0; i < covers; ++i) {
    std::vector<std::int64_t> row(u, 0);
    for (std::size_t j = 0; j < u; ++j) row[j] = percent(rng) < 30 ? 1 : 0;
    lp.a.push_back(row);
    lp.b.push_back(1);
  }
  return lp;
}

Model integer_model(const oracle::GeneratedLp& lp) {
  Model model = oracle::to_model(lp);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (lp.integral[u] != 0) model.col_type[u] = VarType::kInteger;
    if (lp.upper[u] != oracle::kNoUpperBound) {
      model.col_upper[u] = static_cast<double>(lp.upper[u]);
    }
  }
  return model;
}

Options root_cuts(bool loop) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);  // the root the loop sees is the model's own
  for (const char* name :
       {"enable_root_cuts", "enable_mir_cuts", "enable_clique_cuts", "enable_zero_half_cuts"}) {
    options.set_bool(name, true);
  }
  // On models this small 0.2 n is one or two nonzeros; the floor (#608) lets cuts through.
  options.set_int("cut_support_floor", 100);
  options.set_bool("root_cut_loop", loop);
  return options;
}

/// The search's log, captured.
std::string solve_logged(const Model& model, const Options& options, Solution* out) {
  std::FILE* stream = std::tmpfile();
  EXPECT_NE(stream, nullptr);
  Logger logger(stream, LogLevel::kInfo);
  *out = mip::solve_branch_and_bound(model, options, logger, nullptr);
  std::fflush(stream);
  std::rewind(stream);
  std::string text;
  char buffer[4096];
  while (std::fgets(buffer, sizeof(buffer), stream) != nullptr) text += buffer;
  std::fclose(stream);
  return text;
}

TEST(RootCutLoop, OffIsTheSingleRoundAndOnNeverEndsBelowIt) {
  std::mt19937_64 rng(20260927);
  int looped = 0;
  int higher = 0;
  int instances = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const oracle::GeneratedLp lp = binary_instance(rng, 12);
    const Model model = integer_model(lp);
    Solution off;
    const std::string off_log = solve_logged(model, root_cuts(false), &off);
    Solution on;
    const std::string on_log = solve_logged(model, root_cuts(true), &on);
    if (off.status != SolveStatus::kOptimal || std::isnan(off.root_bound_after_cuts)) continue;
    ++instances;
    EXPECT_EQ(off_log.find("Root cut round"), std::string::npos) << "the loop ran while off";
    ASSERT_EQ(on.status, SolveStatus::kOptimal);
    EXPECT_NEAR(on.objective, off.objective, 1e-6 * std::max(1.0, std::fabs(off.objective)));
    // Minimisation (the profits enter as negative costs): a tighter bound is a LARGER one.
    // Round 1 is the same in both runs and later rounds only add rows, so the loop's bound
    // is never looser.
    const double scale = std::max(1.0, std::fabs(off.objective));
    EXPECT_GE(on.root_bound_after_cuts, off.root_bound_after_cuts - 1e-7 * scale);
    if (on_log.find("Root cut round 2:") != std::string::npos) ++looped;
    if (on.root_bound_after_cuts > off.root_bound_after_cuts + 1e-6 * scale) ++higher;
    // Every loop says how it stopped.
    if (on_log.find("Root cut round 1:") != std::string::npos) {
      EXPECT_NE(on_log.find("Root cut loop: "), std::string::npos) << on_log;
      EXPECT_NE(on_log.find("stopped on "), std::string::npos) << on_log;
    }
  }
  EXPECT_GE(instances, 40);
  EXPECT_GT(looped, 0) << "no instance ran a second round";
  EXPECT_GT(higher, 0) << "no instance's root bound moved past the first round";
  std::printf(
      "[  INFO    ] root cut loop: %d instances, %d ran a second round, %d closed more of the "
      "root gap than one round\n",
      instances, looped, higher);
}

/// The loop's closing line, "Root cut loop: R round(s), N row(s) of a budget of B, ...":
/// N and B, or {-1, -1} when the log has none.
std::pair<long, long> rows_and_budget(const std::string& log) {
  const std::size_t at = log.find("Root cut loop: ");
  if (at == std::string::npos) return {-1, -1};
  long rounds = 0;
  long rows = 0;
  long budget = 0;
  if (std::sscanf(log.c_str() + at,
                  "Root cut loop: %ld round(s), %ld row(s) of a budget of %ld", &rounds, &rows,
                  &budget) != 3) {
    return {-1, -1};
  }
  return {rows, budget};
}

TEST(RootCutLoop, NeverAddsMoreRowsThanItsBudget) {
  // Wider binary models than above, and rounds of up to 60 cuts: every loop reports
  // max(kRootCutRowFloor, share * m) as its budget and never passes it, and a loop that
  // reaches it says that is why it stopped. (These loops stall well below the floor; the
  // cut-short round itself is take_within_budget's test below.)
  std::mt19937_64 rng(20260929);
  Options options = root_cuts(true);
  options.set_int("cut_max_per_round", 60);
  int loops = 0;
  int reached = 0;
  long most = 0;
  for (int trial = 0; trial < 30; ++trial) {
    const oracle::GeneratedLp lp = binary_instance(rng, 40);
    const Model model = integer_model(lp);
    Solution on;
    const std::string log = solve_logged(model, options, &on);
    const auto [rows, budget] = rows_and_budget(log);
    if (rows < 0) continue;
    ++loops;
    most = std::max(most, rows);
    const double share = tol::kRootCutRowShare * static_cast<double>(model.num_rows());
    EXPECT_EQ(budget, static_cast<long>(std::max(static_cast<double>(tol::kRootCutRowFloor),
                                                 std::floor(share))));
    EXPECT_LE(rows, budget) << log;
    EXPECT_GE(on.cuts_applied, rows);  // the tree may add more; the root adds no more
    if (rows == budget) {
      ++reached;
      EXPECT_NE(log.find("stopped on the row budget"), std::string::npos) << log;
    }
  }
  EXPECT_GE(loops, 10);
  std::printf("[  INFO    ] root row budget: %d loops, %d reached it, the most rows %ld\n",
              loops, reached, most);
}

TEST(RootCutLoop, TheBudgetIsTheFloorOrTheShareOfTheRows) {
  EXPECT_EQ(mip::root_cut_row_budget(0), tol::kRootCutRowFloor);
  EXPECT_EQ(mip::root_cut_row_budget(29), tol::kRootCutRowFloor);  // gt2: the floor
  const auto timtab1 = static_cast<Index>(std::floor(tol::kRootCutRowShare * 169.0));
  EXPECT_EQ(mip::root_cut_row_budget(169), std::max<Index>(tol::kRootCutRowFloor, timtab1));
}

TEST(RootCutLoop, ARoundCutShortKeepsItsBestAndTheRestWaitFirst) {
  // Five selected cuts, best first, told apart by their rhs; two already waiting.
  auto cut = [](double rhs) {
    mip::Cut c;
    c.coeff = {1.0};
    c.rhs = rhs;
    return c;
  };
  std::vector<mip::Cut> selected{cut(1), cut(2), cut(3), cut(4), cut(5)};
  std::vector<mip::Cut> waiting{cut(10), cut(11)};
  mip::take_within_budget(&selected, &waiting, 2);
  ASSERT_EQ(selected.size(), 2U);
  EXPECT_EQ(selected[0].rhs, 1.0);
  EXPECT_EQ(selected[1].rhs, 2.0);
  ASSERT_EQ(waiting.size(), 5U);
  const std::vector<double> order{3, 4, 5, 10, 11};
  for (std::size_t k = 0; k < order.size(); ++k) EXPECT_EQ(waiting[k].rhs, order[k]) << k;
  // Room enough: nothing moves. No room: everything waits.
  mip::take_within_budget(&selected, &waiting, 7);
  EXPECT_EQ(selected.size(), 2U);
  EXPECT_EQ(waiting.size(), 5U);
  mip::take_within_budget(&selected, &waiting, 0);
  EXPECT_TRUE(selected.empty());
  EXPECT_EQ(waiting.size(), 7U);
  EXPECT_EQ(waiting.front().rhs, 1.0);
  // The waiting list keeps its limit, dropping from the back.
  std::vector<mip::Cut> many(static_cast<std::size_t>(tol::kCutWaitingLimit), cut(0));
  std::vector<mip::Cut> top{cut(-1), cut(-2)};
  mip::take_within_budget(&top, &many, 1);
  EXPECT_EQ(many.size(), static_cast<std::size_t>(tol::kCutWaitingLimit));
  EXPECT_EQ(many.front().rhs, -2.0);
}

TEST(RootCutLoop, ASearchWithTheLoopOnReachesTheExactOptimum) {
  // Pure-binary instances of 8 to 12 columns and random mixed-integer ones of 3 to 6: the
  // later rounds' Gomory cuts come from tableau rows that contain earlier cut rows, which is
  // the case a single round never exercised, and every answer is held to the exact rational
  // branch and bound.
  std::mt19937_64 rng(20260928);
  std::uniform_int_distribution<Index> width(8, 12);
  oracle::GeneratorConfig config;
  config.min_rows = 2;
  config.max_rows = 5;
  config.min_cols = 3;
  config.max_cols = 6;
  config.magnitude = 5;
  Options options = root_cuts(true);
  options.set_bool("presolve", true);
  options.set_int("tree_cut_depth", 3);
  int solved = 0;
  std::int64_t cuts = 0;
  for (int trial = 0; trial < 240; ++trial) {
    oracle::GeneratedLp lp;
    if (trial % 2 == 0) {
      lp = binary_instance(rng, width(rng));
    } else {
      lp = oracle::random_lp(rng, config);
      lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 1);
      for (Index j = 0; j < lp.num_cols; ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (j % 2 == 1) lp.integral[u] = 0;
        if (lp.upper[u] == oracle::kNoUpperBound) lp.upper[u] = 6;
      }
    }
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal) continue;
    const Solution s = solve(integer_model(lp), options);
    const double expected = exact.objective.to_double();
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << lp.to_text();
    EXPECT_NEAR(s.objective, expected, 1e-6 * std::max(1.0, std::fabs(expected)))
        << lp.to_text();
    cuts += s.cuts_applied;
    ++solved;
  }
  EXPECT_GE(solved, 120);
  EXPECT_GT(cuts, 0);
  std::printf("[  INFO    ] root cut loop: %d instances at the exact optimum, %lld cut rows\n",
              solved, static_cast<long long>(cuts));
}

}  // namespace
}  // namespace sankhya
