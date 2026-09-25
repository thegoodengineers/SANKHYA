// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted Halpern iteration and the PID primal weight in the first-order QP engine
// (#493, CPU).
//
// Three things are pinned. Off, the engine is the one it was: the defaults are off and an
// explicit off (with the controller's gains moved, which must then be ignored) reproduces the
// default run bit for bit. On, each switch alone and both together reach the optimum the
// interior point (#490) reaches, on hand-derived QPs and on seeded random ones, and every
// answer passes the in-process KKT check (the same conditions tools/verify_solution.py
// re-derives). And the step-size algebra the controller relies on - that the default steps are
// the weight formula at the default weight, that Condat's condition holds at every weight, and
// that kp = 0.5 alone is PDLP's smoothing - is checked directly.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/kkt_check.hpp"
#include "qp/qp_first_order_accel.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Options quiet(double tolerance = 1e-9) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_double("qp_tolerance", tolerance);
  options.set_int("iteration_limit", 400000);
  return options;
}

Options with(bool halpern, bool pid, double tolerance = 1e-9) {
  Options options = quiet(tolerance);
  options.set_bool("qp_halpern", halpern);
  options.set_bool("qp_primal_weight_pid", pid);
  return options;
}

Model make_qp(const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper,
              const std::vector<std::tuple<Index, Index, double>>& hessian_lower,
              const std::vector<std::tuple<Index, Index, double>>& entries = {},
              const std::vector<double>& row_lower = {},
              const std::vector<double>& row_upper = {}) {
  Model model;
  model.name = "halperntest";
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(row_lower.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(cost.size(), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (const auto& [i, j, v] : entries) model.matrix.add_entry(i, j, v);
  model.matrix.finalize();
  model.hessian.reset(n, n);
  for (const auto& [i, j, v] : hessian_lower) model.hessian.add_entry(i, j, v);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

/// min x1^2 + x2^2 - 2 x1 - 4 x2  s.t.  x1 + x2 <= 2,  x >= 0: optimum (0.5, 1.5), -4.5.
Model inequality_qp() {
  return make_qp({-2.0, -4.0}, {0.0, 0.0}, {kInfinity, kInfinity}, {{0, 0, 2.0}, {1, 1, 2.0}},
                 {{0, 0, 1.0}, {0, 1, 1.0}}, {-kInfinity}, {2.0});
}

/// min (x1^2 + x2^2 + x3^2)/2  s.t.  x1 + x2 + x3 = 3,  1 <= x1 - x2 <= 2,  x1 free, x2 >= 0,
/// x3 fixed at 0.5: optimum (1.75, 0.75, 0.5), objective 1.9375 (derived in test_qp_ipm.cpp).
Model equality_ranged_qp() {
  return make_qp({0.0, 0.0, 0.0}, {-kInfinity, 0.0, 0.5}, {kInfinity, kInfinity, 0.5},
                 {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}},
                 {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {1, 1, -1.0}}, {3.0, 1.0},
                 {3.0, 2.0});
}

/// Q = [[2, 1], [1, 2]] (the off-diagonal entry stored once), c = (-4, -4), x <= 1 on the
/// first column. Unconstrained the minimiser is (4/3, 4/3), so the bound binds: x1 = 1, and
/// stationarity on x2, 1 + 2 x2 = 4, gives x2 = 1.5; the first gradient, 2 + 1.5 - 4 < 0,
/// presses against the upper bound as it must.
Model coupled_bound_qp() {
  return make_qp({-4.0, -4.0}, {0.0, 0.0}, {1.0, kInfinity},
                 {{0, 0, 2.0}, {1, 0, 1.0}, {1, 1, 2.0}});
}

/// A seeded convex QP with rows of every kind: Q = B'B / k + 0.1 I (positive definite, so the
/// optimum is unique and the points can be compared, not only the objectives), a sparse A,
/// and row activities built around a point of the box so the model is feasible.
Model random_qp(unsigned seed, Index n, Index m) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::uniform_real_distribution<double> share(0.0, 1.0);

  const Index k = n / 2;
  std::vector<std::vector<double>> b(static_cast<std::size_t>(k),
                                     std::vector<double>(static_cast<std::size_t>(n), 0.0));
  for (auto& row : b) {
    for (double& v : row) v = share(rng) < 0.3 ? unit(rng) : 0.0;
  }
  std::vector<std::tuple<Index, Index, double>> hessian;
  for (Index j = 0; j < n; ++j) {
    for (Index i = j; i < n; ++i) {
      double v = 0.0;
      for (const auto& row : b)
        v += row[static_cast<std::size_t>(i)] * row[static_cast<std::size_t>(j)];
      v /= static_cast<double>(k);
      if (i == j) v += 0.1;
      if (v != 0.0) hessian.emplace_back(i, j, v);
    }
  }

  std::vector<double> cost(static_cast<std::size_t>(n));
  std::vector<double> lower(static_cast<std::size_t>(n));
  std::vector<double> upper(static_cast<std::size_t>(n));
  std::vector<double> point(static_cast<std::size_t>(n));
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    cost[u] = 3.0 * unit(rng);
    const int kind = static_cast<int>(j % 3);
    lower[u] = kind == 2 ? -kInfinity : 0.0;
    upper[u] = kind == 0 ? 2.0 : kInfinity;
    point[u] = kind == 2 ? unit(rng) : 1.0 + 0.5 * unit(rng);
  }

  std::vector<std::tuple<Index, Index, double>> entries;
  std::vector<double> activity(static_cast<std::size_t>(m), 0.0);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      if (share(rng) < 0.25 || j == (i % n)) {
        const double v = unit(rng);
        entries.emplace_back(i, j, v);
        activity[static_cast<std::size_t>(i)] += v * point[static_cast<std::size_t>(j)];
      }
    }
  }
  std::vector<double> row_lower(static_cast<std::size_t>(m));
  std::vector<double> row_upper(static_cast<std::size_t>(m));
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    switch (i % 4) {
      case 0:  // equality
        row_lower[u] = row_upper[u] = activity[u];
        break;
      case 1:  // <=, tight-ish so it can bind
        row_lower[u] = -kInfinity;
        row_upper[u] = activity[u] + 0.1;
        break;
      case 2:  // >=
        row_lower[u] = activity[u] - 0.1;
        row_upper[u] = kInfinity;
        break;
      default:  // ranged
        row_lower[u] = activity[u] - 0.2;
        row_upper[u] = activity[u] + 0.2;
        break;
    }
  }
  return make_qp(cost, lower, upper, hessian, entries, row_lower, row_upper);
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

/// Solve with the given switches and hold the answer to the interior point's and to the KKT
/// conditions.
void expect_matches_ipm(const Model& model, bool halpern, bool pid, const std::string& label) {
  SCOPED_TRACE(label + (halpern ? " halpern" : "") + (pid ? " pid" : ""));
  Logger logger(nullptr);
  const Solution reference = qp::solve_convex_qp_ipm(model, quiet(), logger);
  ASSERT_EQ(reference.status, SolveStatus::kOptimal) << reference.message;

  const Solution s = qp::solve_convex_qp(model, with(halpern, pid), logger);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, reference.objective,
              1e-6 * std::max(1.0, std::fabs(reference.objective)));
  for (std::size_t j = 0; j < s.col_value.size(); ++j) {
    EXPECT_NEAR(s.col_value[j], reference.col_value[j], 1e-5) << "column " << j;
    EXPECT_GE(s.col_value[j], model.col_lower[j]) << "column " << j << " left its box";
    EXPECT_LE(s.col_value[j], model.col_upper[j]) << "column " << j << " left its box";
  }
  const KktVerdict verdict = check_qp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
}

// =========================================================================================
// Off is what it was
// =========================================================================================

TEST(QpHalpern, BothSwitchesAreOffByDefault) {
  const Options options;
  EXPECT_FALSE(options.get_bool("qp_halpern"));
  EXPECT_FALSE(options.get_bool("qp_primal_weight_pid"));
  EXPECT_EQ(options.get_string("qp_algorithm"), "condat-vu");
}

TEST(QpHalpern, AnExplicitOffIsTheDefaultRunBitForBit) {
  for (const unsigned seed : {1u, 2u}) {
    const Model model = random_qp(seed, 16, 8);
    Logger logger(nullptr);
    const Solution base = qp::solve_convex_qp(model, quiet(), logger);
    Options off = with(false, false);
    off.set_double("qp_pid_kp", 1.3);  // gains of a switched-off controller change nothing
    off.set_double("qp_pid_ki", 0.4);
    off.set_double("qp_pid_kd", 0.2);
    const Solution again = qp::solve_convex_qp(model, off, logger);
    EXPECT_EQ(base.status, again.status);
    EXPECT_EQ(base.iterations, again.iterations);
    EXPECT_TRUE(bitwise_equal(base.col_value, again.col_value));
    EXPECT_TRUE(bitwise_equal(base.row_dual, again.row_dual));
    EXPECT_TRUE(bitwise_equal(base.col_dual, again.col_dual));
  }
}

TEST(QpHalpern, ThePrimalWeightIsInertWithoutRows) {
  // No rows, no dual, nothing for a weight to balance: the switch is ignored, bit for bit.
  const Model model = coupled_bound_qp();
  Logger logger(nullptr);
  const Solution base = qp::solve_convex_qp(model, quiet(), logger);
  const Solution pid = qp::solve_convex_qp(model, with(false, true), logger);
  EXPECT_EQ(base.iterations, pid.iterations);
  EXPECT_TRUE(bitwise_equal(base.col_value, pid.col_value));
}

// =========================================================================================
// On reaches the interior point's optimum
// =========================================================================================

TEST(QpHalpern, EachSwitchReachesTheInteriorPointsOptimumOnHandDerivedQps) {
  for (const auto& [halpern, pid] : {std::pair{true, false}, {false, true}, {true, true}}) {
    expect_matches_ipm(inequality_qp(), halpern, pid, "inequality");
    expect_matches_ipm(equality_ranged_qp(), halpern, pid, "equality-ranged-free-fixed");
    expect_matches_ipm(coupled_bound_qp(), halpern, pid, "coupled-bound");
  }
}

TEST(QpHalpern, TheHandDerivedOptimaThemselves) {
  Logger logger(nullptr);
  const Solution a = qp::solve_convex_qp(inequality_qp(), with(true, true), logger);
  ASSERT_EQ(a.status, SolveStatus::kOptimal) << a.message;
  EXPECT_NEAR(a.objective, -4.5, 1e-7);
  EXPECT_NEAR(a.col_value[0], 0.5, 1e-6);
  EXPECT_NEAR(a.col_value[1], 1.5, 1e-6);
  EXPECT_NEAR(a.row_dual[0], -1.0, 1e-6);

  const Solution b = qp::solve_convex_qp(equality_ranged_qp(), with(true, true), logger);
  ASSERT_EQ(b.status, SolveStatus::kOptimal) << b.message;
  EXPECT_NEAR(b.objective, 1.9375, 1e-7);
  EXPECT_NEAR(b.col_value[0], 1.75, 1e-6);
  EXPECT_NEAR(b.col_value[1], 0.75, 1e-6);
  EXPECT_DOUBLE_EQ(b.col_value[2], 0.5);
}

TEST(QpHalpern, EachSwitchReachesTheInteriorPointsOptimumOnSeededRandomQps) {
  for (const unsigned seed : {11u, 12u, 13u, 14u}) {
    const Model model = random_qp(seed, 24, 12);
    for (const auto& [halpern, pid] : {std::pair{true, false}, {false, true}, {true, true}}) {
      expect_matches_ipm(model, halpern, pid, "seed " + std::to_string(seed));
    }
  }
}

TEST(QpHalpern, AMaximizationIsReportedInItsOwnSense) {
  Model model = inequality_qp();
  model.sense = ObjSense::kMaximize;
  model.col_cost = {2.0, 4.0};
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, -2.0);
  model.hessian.add_entry(1, 1, -2.0);
  model.hessian.finalize();
  Logger logger(nullptr);
  const Solution s = qp::solve_convex_qp(model, with(true, true), logger);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, 4.5, 1e-7);
  EXPECT_NEAR(s.col_value[0], 0.5, 1e-6);
  EXPECT_NEAR(s.row_dual[0], 1.0, 1e-6);
  const KktVerdict verdict = check_qp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
}

TEST(QpHalpern, TheSwitchIsLiveThroughSolve) {
  // Through the dispatcher and presolve, on: the same answer, reached by a different path.
  const Model model = random_qp(21, 24, 12);
  const Solution off = solve(model, quiet());
  const Solution on = solve(model, with(true, false));
  ASSERT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NE(off.iterations, on.iterations);
  EXPECT_NEAR(on.objective, off.objective, 1e-6 * std::max(1.0, std::fabs(off.objective)));
}

// =========================================================================================
// The step-size algebra
// =========================================================================================

TEST(QpHalpernSteps, TheDefaultStepsAreTheWeightFormulaAtTheDefaultWeight) {
  for (const double l : {0.0, 0.3, 2.0, 50.0}) {
    for (const double a2 : {1e-4, 1.0, 9.0, 1e4}) {
      // The engine's default: tau = 1 / (L/2 + ||A||), sigma = (1/tau - L/2) / (2 ||A||^2).
      const double tau = 1.0 / (l / 2.0 + std::sqrt(a2));
      const double sigma = (1.0 / tau - l / 2.0) / (2.0 * a2);
      const qp::CondatVuSteps steps =
          qp::condat_vu_steps_at_weight(l, a2, qp::condat_vu_weight_of(tau, sigma));
      EXPECT_NEAR(steps.tau, tau, 1e-13 * tau) << "L " << l << " a2 " << a2;
      EXPECT_NEAR(steps.sigma, sigma, 1e-12 * sigma) << "L " << l << " a2 " << a2;
    }
  }
}

TEST(QpHalpernSteps, CondatsConditionHoldsAtEveryWeight) {
  for (const double l : {0.0, 1.0, 40.0}) {
    for (const double omega : {1e-6, 1e-3, 0.7, 1.0, 30.0, 1e6}) {
      const double a2 = 4.0;
      const qp::CondatVuSteps s = qp::condat_vu_steps_at_weight(l, a2, omega);
      ASSERT_GT(s.tau, 0.0);
      ASSERT_GT(s.sigma, 0.0);
      EXPECT_NEAR(s.sigma / s.tau, omega * omega, 1e-9 * omega * omega);
      // 1/tau - sigma ||A||^2 >= L/2, with the slack the engine's rule builds in.
      EXPECT_GE(1.0 / s.tau - s.sigma * a2, l / 2.0);
      EXPECT_GE(s.reflection_max, 0.0);
      EXPECT_LT(s.reflection_max, l > 0.0 ? 1.0 : 1.0 + 1e-15);
    }
  }
  // An LP's step is firmly nonexpansive: the full reflection 2T - I is allowed.
  EXPECT_DOUBLE_EQ(qp::condat_vu_reflection_max(0.0, 0.25), 1.0);
}

TEST(QpHalpernSteps, ProportionalOnlyIsPdlpsSmoothing) {
  const qp::PidGains gains{0.5, 0.0, 0.0};
  qp::PidState state;
  const double omega = 2.0;
  const double dx = 3.0;
  const double dy = 0.5;
  const double pdlp = std::exp(0.5 * std::log(dy / dx) + 0.5 * std::log(omega));
  EXPECT_NEAR(qp::pid_primal_weight(omega, dx, dy, gains, &state), pdlp, 1e-14);
}

TEST(QpHalpernSteps, TheControllerSettlesOnABalancedWeightAndIsClamped) {
  const qp::PidGains gains{0.5, 0.1, 0.1};
  qp::PidState state;
  double omega = 1.0;
  // A fixed movement ratio: the balanced weight is dy / dx = 4.
  for (int i = 0; i < 60; ++i) omega = qp::pid_primal_weight(omega, 1.0, 4.0, gains, &state);
  EXPECT_NEAR(omega, 4.0, 1e-6);
  EXPECT_LE(std::fabs(state.integral), tol::kQpPidIntegralLimit);

  qp::PidState fresh;
  EXPECT_EQ(qp::pid_primal_weight(3.0, 0.0, 1.0, gains, &fresh), 3.0);  // no movement
  EXPECT_FALSE(fresh.has_previous);
  EXPECT_EQ(qp::pid_primal_weight(1.0, 1.0, 1e30, qp::PidGains{2.0, 0.0, 0.0}, &fresh),
            tol::kQpPrimalWeightMax);
}

TEST(QpHalpernSteps, TheHalpernBlendIsTheAnchoredReflection) {
  std::vector<double> z = {1.0, -2.0};
  const std::vector<double> tz = {3.0, 0.0};
  const std::vector<double> anchor = {0.0, 4.0};
  // k = 1: w = 2/3; rho = 0.5: reflected = 1.5 tz - 0.5 z = (4, 1).
  qp::halpern_blend(&z, tz, anchor, 1.0, 0.5);
  EXPECT_NEAR(z[0], 2.0 / 3.0 * 4.0 + 1.0 / 3.0 * 0.0, 1e-15);
  EXPECT_NEAR(z[1], 2.0 / 3.0 * 1.0 + 1.0 / 3.0 * 4.0, 1e-15);
}

}  // namespace
}  // namespace sankhya
