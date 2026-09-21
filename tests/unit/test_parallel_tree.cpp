// SPDX-License-Identifier: Apache-2.0
// SANKHYA - parallel tree search (#222).
//
// What must not change with the number of threads is the ANSWER: the status, the objective,
// and that the point verifies. What may change is the tree. So the tests here compare
// against brute-force enumeration at several thread counts, run an instance with many
// optimal points ten times and demand one objective, and check that every limit - nodes,
// time, an interrupt - still leaves a bound that is a bound. CI runs this file under
// ThreadSanitizer, because a race here shows up as a wrong bound once a month, not as a crash.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya {
namespace {

Options on_threads(int threads) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("mip_threads", threads);
  return options;
}

/// How many subtrees the message says were given away; -1 when it is not a parallel answer.
long long given_away(const Solution& solution) {
  static const std::regex pattern(R"((\d+) given away)");
  std::smatch match;
  if (!std::regex_search(solution.message, match, pattern)) return -1;
  return std::stoll(match[1].str());
}

/// A small pure-integer program built around a random point, with equality rows now and then.
Model random_integer_program(std::mt19937& rng, bool maximize) {
  std::uniform_int_distribution<int> columns(5, 8);
  std::uniform_int_distribution<int> rows(2, 4);
  std::uniform_int_distribution<int> upper(1, 3);
  std::uniform_int_distribution<int> coefficient(-3, 7);
  std::uniform_int_distribution<int> cost(-9, 9);
  std::uniform_int_distribution<int> kind(0, 3);
  const int n = columns(rng);
  const int m = rows(rng);
  Model model;
  model.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  std::vector<double> point;
  for (int j = 0; j < n; ++j) {
    model.col_upper.push_back(static_cast<double>(upper(rng)));
    model.col_cost.push_back(static_cast<double>(cost(rng)));
    point.push_back(static_cast<double>(
        std::uniform_int_distribution<int>(0, static_cast<int>(model.col_upper.back()))(rng)));
  }
  model.matrix.reset(m, n);
  for (int i = 0; i < m; ++i) {
    double activity = 0.0;
    for (int j = 0; j < n; ++j) {
      const int a = coefficient(rng);
      if (a == 0 || a == 1) continue;
      model.matrix.add_entry(i, j, static_cast<double>(a));
      activity += a * point[static_cast<std::size_t>(j)];
    }
    const double shift = std::uniform_int_distribution<int>(0, 4)(rng) == 0 ? 1.0 : 0.0;
    if (kind(rng) == 0) {
      model.row_lower.push_back(activity + shift);
      model.row_upper.push_back(activity + shift);
    } else {
      model.row_lower.push_back(-kInfinity);
      model.row_upper.push_back(activity + 2.0 + shift);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

struct Enumerated {
  bool feasible = false;
  double best = 0.0;
};

Enumerated enumerate(const Model& model) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<std::vector<double>> dense(m, std::vector<double>(n, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      dense[static_cast<std::size_t>(column.rows[k])][j] += column.values[k];
    }
  }
  Enumerated result;
  std::vector<double> x(model.col_lower);
  const bool maximize = model.sense == ObjSense::kMaximize;
  while (true) {
    bool ok = true;
    for (std::size_t i = 0; i < m && ok; ++i) {
      double activity = 0.0;
      for (std::size_t j = 0; j < n; ++j) activity += dense[i][j] * x[j];
      ok = activity >= model.row_lower[i] - 1e-9 && activity <= model.row_upper[i] + 1e-9;
    }
    if (ok) {
      double value = 0.0;
      for (std::size_t j = 0; j < n; ++j) value += model.col_cost[j] * x[j];
      if (!result.feasible || (maximize ? value > result.best : value < result.best)) {
        result.best = value;
      }
      result.feasible = true;
    }
    std::size_t j = 0;
    while (j < n && x[j] >= model.col_upper[j]) {
      x[j] = model.col_lower[j];
      ++j;
    }
    if (j == n) break;
    x[j] += 1.0;
  }
  return result;
}

/// 0-1 knapsack with a side constraint: enough nodes that every worker gets some.
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
    // A fractional part on every value, so the objective is not integral and #221's
    // bound rounding cannot close the tree at the root: these models exist to have a tree.
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

/// A market split instance (Cornuejols & Dawande, IPCO 1998): rows of random weights over
/// binaries, each row to hit half its total exactly, the misses penalised. Small and famously
/// hard for branch and bound, which is what the limit tests need: a tree that outlasts them.
Model market_split(int rows, int columns, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> weight(0, 99);
  Model model;
  const int n = columns + 2 * rows;
  model.matrix.reset(rows, n);
  for (int j = 0; j < columns; ++j) {
    model.col_cost.push_back(0.0);
    model.col_lower.push_back(0.0);
    model.col_upper.push_back(1.0);
    model.col_type.push_back(VarType::kInteger);
  }
  for (int i = 0; i < rows; ++i) {
    double total = 0.0;
    for (int j = 0; j < columns; ++j) {
      const int a = weight(rng);
      model.matrix.add_entry(i, j, static_cast<double>(a));
      total += a;
    }
    for (int side = 0; side < 2; ++side) {  // the miss above and below the target
      const int column = columns + 2 * i + side;
      model.col_cost.push_back(1.0);
      model.col_lower.push_back(0.0);
      model.col_upper.push_back(kInfinity);
      model.col_type.push_back(VarType::kContinuous);
      model.matrix.add_entry(i, column, side == 0 ? 1.0 : -1.0);
    }
    model.row_lower.push_back(std::floor(total / 2.0));
    model.row_upper.push_back(std::floor(total / 2.0));
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

TEST(ParallelTree, EveryThreadCountAgreesWithEnumeration) {
  std::mt19937 rng(222);
  long long donated = 0;
  int infeasible = 0;
  for (int trial = 0; trial < 120; ++trial) {
    const Model model = random_integer_program(rng, trial % 2 == 1);
    const Enumerated truth = enumerate(model);
    if (!truth.feasible) ++infeasible;
    for (const int threads : {2, 4, 8}) {
      const Solution solved = solve(model, on_threads(threads));
      if (!truth.feasible) {
        EXPECT_EQ(solved.status, SolveStatus::kInfeasible)
            << "trial " << trial << " threads " << threads << ": " << solved.message;
        continue;
      }
      ASSERT_EQ(solved.status, SolveStatus::kOptimal)
          << "trial " << trial << " threads " << threads << ": " << solved.message;
      EXPECT_NEAR(solved.objective, truth.best, 1e-7)
          << "trial " << trial << " threads " << threads;
      EXPECT_LE(solved.primal_infeasibility, 1e-6);
      EXPECT_LE(solved.integrality_violation, 1e-6);
      donated += std::max(0LL, given_away(solved));
    }
  }
  EXPECT_GT(infeasible, 3);
  EXPECT_GT(donated, 50) << "the workers must actually have shared the trees";
}

TEST(ParallelTree, AKnapsackIsSplitAcrossTheWorkersAndSolvedToTheSameOptimum) {
  for (const std::uint32_t seed : {1u, 2u, 3u}) {
    const Model model = knapsack(30, seed);
    const Solution sequential = solve(model, on_threads(1));
    ASSERT_EQ(sequential.status, SolveStatus::kOptimal) << sequential.message;
    EXPECT_EQ(given_away(sequential), -1) << "one thread is the sequential search";
    const Solution parallel = solve(model, on_threads(4));
    ASSERT_EQ(parallel.status, SolveStatus::kOptimal) << parallel.message;
    EXPECT_NEAR(parallel.objective, sequential.objective, 1e-6) << "seed " << seed;
    EXPECT_LE(parallel.primal_infeasibility, 1e-6);
    EXPECT_GT(given_away(parallel), 0) << parallel.message;
  }
}

TEST(ParallelTree, TenRunsOfAModelWithManyOptimaReportOneObjective) {
  // Twelve identical items and room for exactly six: C(12, 6) = 924 optimal selections, so
  // which one a run returns is up to the timing. The objective and the status are not.
  Model model;
  model.sense = ObjSense::kMaximize;
  const int items = 12;
  model.matrix.reset(1, items);
  for (int j = 0; j < items; ++j) {
    model.col_cost.push_back(5.0);
    model.col_lower.push_back(0.0);
    model.col_upper.push_back(1.0);
    model.col_type.push_back(VarType::kInteger);
    model.matrix.add_entry(0, j, 7.0);
  }
  model.matrix.finalize();
  model.row_lower = {-kInfinity};
  model.row_upper = {6.0 * 7.0 + 3.0};
  model.hessian.reset(items, items);
  model.hessian.finalize();
  for (int run = 0; run < 10; ++run) {
    const Solution solved = solve(model, on_threads(4));
    ASSERT_EQ(solved.status, SolveStatus::kOptimal) << "run " << run << ": " << solved.message;
    EXPECT_NEAR(solved.objective, 30.0, 1e-9) << "run " << run;
    EXPECT_LE(solved.primal_infeasibility, 1e-9) << "run " << run;
  }
}

TEST(ParallelTree, ANodeLimitCountsTheWholeSearchAndLeavesAValidBound) {
  const Model model = market_split(2, 16, 7);
  const Solution full = solve(model, on_threads(1));
  ASSERT_EQ(full.status, SolveStatus::kOptimal);
  Options limited = on_threads(4);
  limited.set_int("node_limit", 60);
  const Solution stopped = solve(model, limited);
  ASSERT_TRUE(stopped.status == SolveStatus::kFeasible ||
              stopped.status == SolveStatus::kNodeLimit)
      << to_string(stopped.status) << " after " << stopped.nodes << " nodes (" << full.nodes
      << " for the proof): " << stopped.message;
  EXPECT_EQ(stopped.stopped_by, LimitReason::kNodes);
  // Each worker counts its node before it checks, so the overshoot is at most a node each.
  EXPECT_LE(stopped.nodes, 60 + 4);
  EXPECT_GT(full.nodes, 200) << "the model should need a real tree";
  // A minimisation: the bound must be no larger than the optimum, the incumbent no smaller.
  EXPECT_LE(stopped.dual_bound, full.objective + 1e-6) << "a bound past the optimum";
  if (stopped.status == SolveStatus::kFeasible) {
    EXPECT_GE(stopped.objective, full.objective - 1e-6);
  }
}

TEST(ParallelTree, ALimitedSearchReportsTheSameRoundedBoundAsTheSequentialOne) {
  // With integer costs on integer columns every integer optimum is an integer, and the
  // sequential search reports its bound rounded up to one (#221). The parallel search must
  // too, or a 4-thread run reports 13.28 where the 1-thread run on the same tree says 14.
  std::mt19937 rng(2221);
  int limited = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Model model = random_integer_program(rng, trial % 2 == 0);
    const Enumerated truth = enumerate(model);
    if (!truth.feasible) continue;
    Options options = on_threads(4);
    options.set_int("node_limit", 4);
    const Solution stopped = solve(model, options);
    if (stopped.status == SolveStatus::kOptimal || !std::isfinite(stopped.dual_bound)) continue;
    ++limited;
    EXPECT_EQ(stopped.dual_bound, std::round(stopped.dual_bound))
        << "trial " << trial << ": an unrounded bound " << stopped.dual_bound;
    const bool maximize = model.sense == ObjSense::kMaximize;
    EXPECT_TRUE(maximize ? stopped.dual_bound >= truth.best - 1e-9
                         : stopped.dual_bound <= truth.best + 1e-9)
        << "trial " << trial << ": the bound " << stopped.dual_bound << " passes the optimum "
        << truth.best;
  }
  EXPECT_GT(limited, 5);
}

TEST(ParallelTree, ATimeLimitStopsEveryWorkerPromptly) {
  const Model model = market_split(4, 40, 11);
  Options limited = on_threads(4);
  limited.set_double("time_limit", 0.3);
  const auto start = std::chrono::steady_clock::now();
  const Solution stopped = solve(model, limited);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  EXPECT_LT(seconds, 5.0) << "the workers must see the limit, not run the tree out";
  ASSERT_NE(stopped.status, SolveStatus::kOptimal) << "the model must outlast the limit";
  EXPECT_EQ(stopped.stopped_by, LimitReason::kTime) << stopped.message;
}

TEST(ParallelTree, AnInterruptFromAnotherThreadStopsTheSearch) {
  const Model model = market_split(4, 40, 13);
  SolveControl control;
  std::thread stopper([&control] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    control.interrupt();
  });
  const Solution stopped = solve(model, on_threads(4), &control);
  stopper.join();
  ASSERT_NE(stopped.status, SolveStatus::kOptimal) << "the model must outlast the interrupt";
  EXPECT_EQ(stopped.stopped_by, LimitReason::kInterrupt) << stopped.message;
}

TEST(ParallelTree, AProgressCallbackIsCalledFromOneThreadAndCanStopTheSearch) {
  const Model model = market_split(4, 40, 17);
  SolveControl control;
  const std::thread::id caller = std::this_thread::get_id();
  std::atomic<int> calls{0};
  std::atomic<bool> foreign{false};
  control.progress_callback = [&](const Progress&) {
    if (std::this_thread::get_id() != caller) foreign = true;
    return ++calls >= 3 ? 1 : 0;
  };
  const Solution stopped = solve(model, on_threads(4), &control);
  EXPECT_FALSE(foreign) << "the callback was never promised to be thread-safe";
  EXPECT_GE(calls.load(), 3);
  ASSERT_NE(stopped.status, SolveStatus::kOptimal) << "the model must outlast three callbacks";
  EXPECT_EQ(stopped.stopped_by, LimitReason::kInterrupt);
}

TEST(ParallelTree, ThePoolIsSharedAndEveryMemberIsFeasible) {
  const Model model = knapsack(24, 5);
  Options options = on_threads(4);
  options.set_int("pool_size", 5);
  const Solution solved = solve(model, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  ASSERT_GE(solved.pool.size(), 2u);
  EXPECT_NEAR(solved.pool.front().objective, solved.objective, 1e-9);
  for (std::size_t k = 0; k < solved.pool.size(); ++k) {
    const std::vector<double>& x = solved.pool[k].col_value;
    ASSERT_EQ(x.size(), static_cast<std::size_t>(model.num_cols()));
    std::vector<double> activity(2, 0.0);
    model.matrix.multiply(x.data(), activity.data());
    EXPECT_LE(activity[0], model.row_upper[0] + 1e-9) << "member " << k;
    EXPECT_LE(activity[1], model.row_upper[1] + 1e-9) << "member " << k;
    if (k > 0) {
      EXPECT_LE(solved.pool[k].objective, solved.pool[k - 1].objective + 1e-9);
    }
  }
}

TEST(ParallelTree, TheModesItDoesNotTakeRunSequentially) {
  const Model model = knapsack(16, 3);
  Options deterministic = on_threads(4);
  deterministic.set_bool("deterministic", true);
  EXPECT_EQ(given_away(solve(model, deterministic)), -1);
  Options complete = on_threads(4);
  complete.set_bool("pool_complete", true);
  complete.set_int("pool_size", 3);
  EXPECT_EQ(given_away(solve(model, complete)), -1);
}

TEST(ParallelTree, ANodeLpStoppedByALimitKeepsItsBoundInTheAnswer) {
  // Before #222 the node whose LP hit the iteration limit was dropped from the open list,
  // and with nothing else open the search reported a bound of +infinity on a minimisation:
  // a claim that no point is better than infinity, which is true, but a bound it had no
  // business printing. The root's bound, minus infinity, is what the search knows.
  Model model = knapsack(20, 9);
  model.sense = ObjSense::kMinimize;
  for (double& c : model.col_cost) c = -c;
  for (const int threads : {1, 4}) {
    Options options = on_threads(threads);
    options.set_int("iteration_limit", 1);
    options.set_bool("presolve", false);
    const Solution stopped = solve(model, options);
    ASSERT_EQ(stopped.stopped_by, LimitReason::kIterations)
        << threads << ": " << to_string(stopped.status) << " " << stopped.message;
    EXPECT_FALSE(stopped.dual_bound > 0.0) << threads << ": the bound " << stopped.dual_bound
                                           << " claims more than the search proved";
  }
}

}  // namespace
}  // namespace sankhya
