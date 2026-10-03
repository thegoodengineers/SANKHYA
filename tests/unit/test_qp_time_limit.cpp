// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a QP or MIQP solve ends at its time limit, wherever the time goes (#835).
//
// QPLIB_9030 (10,000 integer columns, a quadratic objective, linear rows) under a 60 s limit
// ended at 170.8 s. Each node QP honoured the clock it was handed, but the tree handed every
// one of them the whole time_limit: the root QP took 48 s, the re-solve after the root cut
// round got another 60 s, the solve after that another 60 s. Then, with the time spent, 21
// more node QPs each paid for the convexity test's LDL^T before looking at the clock. The
// instance is fetched, not tracked, so the tests here hold both halves on generated models:
//
//   - a node QP runs on the time that is LEFT, as a node LP has since #803;
//   - the convexity test answers to the deadline, and a stopped test decides nothing: the
//     solve ends with time_limit, never optimal and never a refusal.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

#include "qp/convexity.hpp"

namespace sankhya {
namespace {

using Clock = std::chrono::steady_clock;

/// What a solve may overrun its limit by here: one step that cannot be interrupted, plus a
/// loaded CI machine's scheduling. The old code overran by seconds in every case below.
constexpr double kMargin = 0.5;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// A symmetric, strictly diagonally dominant (so positive definite) Hessian on a random
/// sparse graph, `degree` off-diagonal entries per column on average. A random graph has no
/// small separators, so its LDL^T fills in to nearly dense whatever the ordering: about
/// n^3 / 3 operations, seconds at n = 4,000, which is what the deadline has to reach. With
/// `negative_last`, the last diagonal entry is negative and Q is indefinite instead.
Model random_hessian_qp(Index n, int degree, std::uint64_t seed, bool negative_last = false) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<Index> pick(0, n - 1);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  const auto un = static_cast<std::size_t>(n);
  std::vector<std::tuple<Index, Index, double>> entries;
  std::vector<double> row_sum(un, 0.0);
  for (Index j = 0; j < n; ++j) {
    for (int k = 0; k < degree / 2; ++k) {
      const Index i = pick(rng);
      if (i == j) continue;
      const double v = value(rng);
      entries.emplace_back(std::max(i, j), std::min(i, j), v);
      row_sum[static_cast<std::size_t>(i)] += std::fabs(v);
      row_sum[static_cast<std::size_t>(j)] += std::fabs(v);
    }
  }
  Model model;
  model.name = "random_hessian";
  model.col_cost.assign(un, 1.0);
  model.col_lower.assign(un, -10.0);
  model.col_upper.assign(un, 10.0);
  model.col_type.assign(un, VarType::kContinuous);
  model.matrix.reset(0, n);
  model.matrix.finalize();
  model.hessian.reset(n, n);
  for (const auto& [i, j, v] : entries) model.hessian.add_entry(i, j, v);
  for (Index j = 0; j < n; ++j) {
    const double d = row_sum[static_cast<std::size_t>(j)] + 1.0;
    model.hessian.add_entry(j, j, negative_last && j == n - 1 ? -d : d);
  }
  model.hessian.finalize();
  return model;
}

/// The shape of QPLIB_9030: integer columns in [0, 10], `>=` rows of two columns, and a
/// banded positive definite Hessian whose diagonal spans four orders of magnitude, so a
/// first-order method needs far more than a second to reach the node tolerance.
Model banded_miqp(Index n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> exponent(0.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_int_distribution<Index> pick(0, n - 1);
  const auto un = static_cast<std::size_t>(n);
  Model model;
  model.name = "banded_miqp";
  model.col_cost.resize(un);
  for (auto& c : model.col_cost) c = -50.0 * unit(rng);
  model.col_lower.assign(un, 0.0);
  model.col_upper.assign(un, 10.0);
  model.col_type.assign(un, VarType::kInteger);
  const Index m = n / 2;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    const Index a = pick(rng);
    Index b = pick(rng);
    if (b == a) b = (a + 1) % n;
    model.matrix.add_entry(i, a, 1.0);
    model.matrix.add_entry(i, b, 1.0);
    model.row_lower.push_back(std::floor(10.0 * unit(rng)) + 0.5);
    model.row_upper.push_back(kInfinity);
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  std::vector<double> diagonal(un);
  for (auto& d : diagonal) d = std::pow(10.0, exponent(rng));
  for (Index j = 0; j + 1 < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double coupling = 0.4 * std::min(diagonal[u], diagonal[u + 1]);
    model.hessian.add_entry(j + 1, j, coupling);
  }
  for (Index j = 0; j < n; ++j)
    model.hessian.add_entry(j, j, diagonal[static_cast<std::size_t>(j)]);
  model.hessian.finalize();
  return model;
}

// =========================================================================================
// The convexity test and its deadline
// =========================================================================================

TEST(QpTimeLimit, AConvexityTestStoppedByItsDeadlineDecidesNothing) {
  const std::function<bool()> always = [] { return true; };
  const std::function<bool()> never = [] { return false; };
  for (const bool indefinite : {false, true}) {
    const Model model = random_hessian_qp(60, 6, 835, indefinite);
    // Unstopped, the test decides, and the generator means what it says.
    const qp::ConvexityResult decided = qp::check_convexity(model, never);
    EXPECT_FALSE(decided.stopped);
    EXPECT_EQ(decided.verdict, indefinite ? qp::Convexity::kIndefinite : qp::Convexity::kConvex)
        << decided.detail;
    // Stopped, it decides nothing either way: never convex, never a refusal with a witness.
    const qp::ConvexityResult stopped = qp::check_convexity(model, always);
    EXPECT_TRUE(stopped.stopped) << stopped.detail;
    EXPECT_EQ(stopped.verdict, qp::Convexity::kUnverified) << stopped.detail;
    EXPECT_TRUE(stopped.witness.empty());
  }
}

TEST(QpTimeLimit, TheConvexityTestEndsAtTheTimeLimitInBothQpEngines) {
  // Its LDL^T fills in to nearly dense, seconds of work with no deadline: the old test ran
  // it to the end and only then did the engine look at the clock, and under this 0.05 s
  // limit the solve returned after 2.85 s (Condat-Vu) and 3.91 s (interior point) on the
  // Windows laptop it was measured on.
  const Model model = random_hessian_qp(4000, 12, 9030);
  for (const bool ipm : {false, true}) {
    Options options;
    options.set_bool("log_to_console", false);
    options.set_double("time_limit", 0.05);
    Logger logger(nullptr);
    const auto start = Clock::now();
    const Solution solution = ipm ? qp::solve_convex_qp_ipm(model, options, logger)
                                  : qp::solve_convex_qp(model, options, logger);
    const double elapsed = seconds_since(start);
    SCOPED_TRACE(ipm ? "interior point" : "Condat-Vu");
    EXPECT_LT(elapsed, 0.05 + kMargin) << solution.message;
    EXPECT_EQ(solution.status, SolveStatus::kTimeLimit) << solution.message;
    EXPECT_NE(solution.message.find("convexity not established within the time limit"),
              std::string::npos)
        << solution.message;
  }
}

// =========================================================================================
// Node QPs on the time that is left
// =========================================================================================

TEST(QpTimeLimit, AnMiqpNodeQpRunsOnTheTimeThatIsLeft) {
  // The first time the tree looks at the clock, before the root QP, the progress callback
  // spends most of the budget. A root QP handed the whole time_limit afresh then runs a full
  // limit past that point; on the time that is left it stops where the limit is. Measured
  // with the old code: 2.12 s and 2.14 s against the 1.2 s limit, the two engines alike.
  const Model model = banded_miqp(20000, 835);
  constexpr double kLimit = 1.2;
  constexpr auto kSpent = std::chrono::milliseconds(900);

  for (const bool ipm : {false, true}) {
    SCOPED_TRACE(ipm ? "miqp_node_ipm=true" : "Condat-Vu nodes");
    bool slept = false;
    bool qp_before_sleep = false;
    SolveControl control;
    control.progress_callback = [&](const Progress& progress) {
      if (progress.phase == Progress::Phase::kTree && !slept) {
        slept = true;
        std::this_thread::sleep_for(kSpent);
      } else if (!slept && progress.phase == Progress::Phase::kLp) {
        qp_before_sleep = true;
      }
      return 0;
    };
    Options options;
    options.set_bool("log_to_console", false);
    options.set_bool("presolve", false);
    options.set_bool("mip_symmetry", false);
    options.set_bool("miqp_node_ipm", ipm);
    options.set_double("time_limit", kLimit);
    Logger logger(nullptr);
    const auto start = Clock::now();
    const Solution solution = mip::solve_branch_and_bound(model, options, logger, &control);
    const double elapsed = seconds_since(start);
    if (!slept || qp_before_sleep) {
      GTEST_SKIP() << "the tree's first clock check did not come before the root QP here";
    }
    EXPECT_LT(elapsed, kLimit + kMargin) << solution.message;
    EXPECT_TRUE(solution.status == SolveStatus::kTimeLimit ||
                solution.status == SolveStatus::kFeasible)
        << to_string(solution.status) << ": " << solution.message;
  }
}

}  // namespace
}  // namespace sankhya
