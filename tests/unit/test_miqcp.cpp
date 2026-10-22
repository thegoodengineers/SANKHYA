// SPDX-License-Identifier: Apache-2.0
// MIQCP through solve_conic() (#755, item 4): small random instances checked against brute
// force enumeration of every integer point, and every returned point re-verified here on the
// model's own rows, bounds and integrality.
//
// The family: integers x_1..x_3 in [-3, 3] and one free continuous y,
//     min  c'x + d y
//     s.t. sum_j a_j (x_j - p_j)^2 + y^2 <= r      (convex, a_j > 0)
//          x_1 + x_2 + x_3 >= k
// With x fixed the best y is closed form, -|d| sqrt(r - sum_j a_j (x_j - p_j)^2), so the
// enumeration is exact and does not lean on the engine under test.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qcqp.hpp"

namespace sankhya {
namespace {

struct Instance {
  double a[3], p[3], c[3], d, r, k;
};

QcqpModel build(const Instance& in) {
  QcqpModel m;
  Model& lin = m.linear;
  lin.resize_columns(4);
  for (std::size_t j = 0; j < 3; ++j) {
    lin.col_lower[j] = -3.0;
    lin.col_upper[j] = 3.0;
    lin.col_type[j] = VarType::kInteger;
    lin.col_cost[j] = in.c[j];
  }
  lin.col_lower[3] = -kInfinity;
  lin.col_cost[3] = in.d;
  lin.resize_rows(2);
  SparseMatrix a(2, 4);
  double constant = 0.0;
  for (Index j = 0; j < 3; ++j) {
    const auto u = static_cast<std::size_t>(j);
    a.add_entry(0, j, -2.0 * in.a[u] * in.p[u]);
    a.add_entry(1, j, 1.0);
    m.quadratic.push_back({0, j, j, in.a[u]});
    constant += in.a[u] * in.p[u] * in.p[u];
  }
  m.quadratic.push_back({0, 3, 3, 1.0});
  a.finalize();
  lin.matrix = std::move(a);
  lin.row_lower = {-kInfinity, in.k};
  lin.row_upper = {in.r - constant, kInfinity};
  return m;
}

/// The optimum by enumeration, +inf when no integer point is feasible.
double brute_force(const Instance& in) {
  double best = kInfinity;
  for (int x0 = -3; x0 <= 3; ++x0) {
    for (int x1 = -3; x1 <= 3; ++x1) {
      for (int x2 = -3; x2 <= 3; ++x2) {
        const double x[3] = {double(x0), double(x1), double(x2)};
        if (x[0] + x[1] + x[2] < in.k) continue;
        double s = 0.0;
        double cx = 0.0;
        for (int j = 0; j < 3; ++j) {
          s += in.a[j] * (x[j] - in.p[j]) * (x[j] - in.p[j]);
          cx += in.c[j] * x[j];
        }
        if (s > in.r) continue;
        best = std::min(best, cx - std::fabs(in.d) * std::sqrt(in.r - s));
      }
    }
  }
  return best;
}

void verify(const QcqpModel& m, const Solution& s) {
  const Model& lin = m.linear;
  ASSERT_EQ(s.col_value.size(), 4U);
  for (std::size_t j = 0; j < 4; ++j) {
    EXPECT_GE(s.col_value[j], lin.col_lower[j] - 1e-7);
    EXPECT_LE(s.col_value[j], lin.col_upper[j] + 1e-7);
    if (lin.col_type[j] == VarType::kInteger) {
      EXPECT_NEAR(s.col_value[j], std::round(s.col_value[j]), 1e-6);
    }
  }
  const std::vector<double> act = m.row_activity(s.col_value);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_GE(act[i], lin.row_lower[i] - 1e-7);
    EXPECT_LE(act[i], lin.row_upper[i] + 1e-7);
  }
  EXPECT_NEAR(s.objective, m.objective(s.col_value), 1e-9);
}

TEST(Miqcp, MatchesBruteForceOnRandomInstances) {
  std::mt19937 rng(755);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  Options options;
  options.set_bool("log_to_console", false);
  int feasible = 0;
  int infeasible = 0;
  for (int trial = 0; trial < 40; ++trial) {
    Instance in{};
    for (int j = 0; j < 3; ++j) {
      in.a[j] = 0.5 + 2.0 * u01(rng);
      in.p[j] = -2.0 + 4.0 * u01(rng);
      in.c[j] = -2.0 + 4.0 * u01(rng);
    }
    in.d = -1.0 + 2.0 * u01(rng);
    in.r = 0.3 + 6.0 * u01(rng);
    in.k = std::floor(-1.0 + 4.0 * u01(rng));
    const QcqpModel m = build(in);
    const double expected = brute_force(in);
    const Solution s = solve_conic(m, options);
    SCOPED_TRACE(::testing::Message() << "trial " << trial << ": " << s.message);
    if (expected == kInfinity) {
      ++infeasible;
      EXPECT_EQ(s.status, SolveStatus::kInfeasible);
      continue;
    }
    ++feasible;
    ASSERT_EQ(s.status, SolveStatus::kOptimal);
    verify(m, s);
    // The optimum to the MIP gap the search closes: absolute 1e-6 or relative 1e-4.
    EXPECT_NEAR(s.objective, expected, std::max(1e-5, 1e-4 * std::fabs(expected)));
    EXPECT_LE(s.dual_bound, s.objective + 1e-9);
  }
  // The family must exercise both outcomes, or the test proves less than it says.
  EXPECT_GT(feasible, 10);
  EXPECT_GT(infeasible, 0);
}

// The search ends within time_limit: each node is given only what is left of it, so neither
// the tree nor its last node runs past the limit. 16 integers in [-5, 5] under one ball, far
// more tree than 50 ms allows; the slack is for one interior point iteration and the clock.
TEST(Miqcp, TimeLimitBoundsTheWholeSearch) {
  constexpr Index n = 16;
  QcqpModel m;
  Model& lin = m.linear;
  lin.resize_columns(n);
  std::mt19937 rng(1146);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  SparseMatrix a(1, n);
  double constant = 0.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    lin.col_lower[u] = -5.0;
    lin.col_upper[u] = 5.0;
    lin.col_type[u] = VarType::kInteger;
    lin.col_cost[u] = -1.0 + 2.0 * u01(rng);
    const double p = -0.5 + u01(rng);
    a.add_entry(0, j, -2.0 * p);
    m.quadratic.push_back({0, j, j, 1.0});
    constant += p * p;
  }
  a.finalize();
  lin.resize_rows(1);
  lin.matrix = std::move(a);
  lin.row_lower = {-kInfinity};
  lin.row_upper = {7.3 - constant};
  Options options;
  options.set_bool("log_to_console", false);
  constexpr double kLimit = 0.05;
  options.set_double("time_limit", kLimit);
  const Solution s = solve_conic(m, options);
  EXPECT_LE(s.solve_seconds, kLimit + 0.25) << s.message;
  if (s.status != SolveStatus::kOptimal) {
    EXPECT_TRUE(s.status == SolveStatus::kFeasible || s.status == SolveStatus::kTimeLimit)
        << s.message;
  }
  if (!s.col_value.empty()) {
    for (const double v : s.col_value) EXPECT_NEAR(v, std::round(v), 1e-6);
    EXPECT_LE(m.row_activity(s.col_value)[0], lin.row_upper[0] + 1e-7);
  }
}

}  // namespace
}  // namespace sankhya
