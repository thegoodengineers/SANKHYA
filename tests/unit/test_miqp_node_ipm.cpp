// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MIQP with the QP interior point as the node solver (#494), against enumeration.
//
// Small convex pure-integer QPs, Q = B'B + D with integer B and D >= 0, so every one is convex
// and every integer point can be enumerated. The search with miqp_node_ipm on must reach the
// enumerated optimum, report an infeasible model as infeasible (the interior point has no
// infeasibility detection; the node LP decides those nodes), and never report a dual bound
// past the optimum when a node limit stops it - the bound it prunes on is the linearised one,
// valid for any iterate.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

struct Instance {
  Model model;
  std::vector<std::vector<double>> q;  // the full symmetric Hessian, for enumeration
};

Instance random_convex_miqp(std::mt19937& rng, bool maximize) {
  const int n = std::uniform_int_distribution<int>(3, 5)(rng);
  const int m = std::uniform_int_distribution<int>(1, 3)(rng);
  const int factors = std::uniform_int_distribution<int>(1, 3)(rng);
  Instance instance;
  Model& model = instance.model;
  model.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 3.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  std::uniform_int_distribution<int> cost(-12, 12);
  for (int j = 0; j < n; ++j) model.col_cost.push_back(static_cast<double>(cost(rng)));

  // Q = B'B + D in the minimise sense; a maximised model carries -Q (concave).
  std::vector<std::vector<double>> b(static_cast<std::size_t>(factors),
                                     std::vector<double>(static_cast<std::size_t>(n)));
  for (auto& row : b) {
    for (double& v : row) v = std::uniform_int_distribution<int>(-2, 2)(rng);
  }
  auto& q = instance.q;
  q.assign(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
  for (std::size_t i = 0; i < q.size(); ++i) {
    for (std::size_t j = 0; j < q.size(); ++j) {
      for (const auto& row : b) q[i][j] += row[i] * row[j];
    }
    q[i][i] += std::uniform_int_distribution<int>(0, 2)(rng);
  }
  const double sign = maximize ? -1.0 : 1.0;
  model.hessian.reset(n, n);
  for (int j = 0; j < n; ++j) {
    for (int i = j; i < n; ++i) {
      const double v = q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.hessian.add_entry(i, j, sign * v);
    }
  }
  model.hessian.finalize();
  for (auto& row : q) {
    for (double& v : row) v *= sign;
  }

  std::vector<double> point(static_cast<std::size_t>(n));
  for (double& v : point) v = std::uniform_int_distribution<int>(0, 3)(rng);
  model.matrix.reset(m, n);
  for (int i = 0; i < m; ++i) {
    double activity = 0.0;
    for (int j = 0; j < n; ++j) {
      const int a = std::uniform_int_distribution<int>(-2, 5)(rng);
      if (a == 0 || a == 1) continue;
      model.matrix.add_entry(i, j, static_cast<double>(a));
      activity += a * point[static_cast<std::size_t>(j)];
    }
    // Equalities make infeasible nodes, and one row in six is nudged off the point so some
    // models are infeasible outright.
    const double shift = std::uniform_int_distribution<int>(0, 5)(rng) == 0 ? 1.0 : 0.0;
    switch (std::uniform_int_distribution<int>(0, 2)(rng)) {
      case 0:
        model.row_lower.push_back(activity + shift);
        model.row_upper.push_back(activity + shift);
        break;
      case 1:
        model.row_lower.push_back(-kInfinity);
        model.row_upper.push_back(activity + 2.0 - 3.0 * shift);
        break;
      default:
        model.row_lower.push_back(activity - 2.0 + 3.0 * shift);
        model.row_upper.push_back(kInfinity);
        break;
    }
  }
  model.matrix.finalize();
  return instance;
}

double objective(const Instance& instance, const std::vector<double>& x) {
  double value = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    value += instance.model.col_cost[j] * x[j];
    for (std::size_t k = 0; k < x.size(); ++k) value += 0.5 * x[j] * instance.q[j][k] * x[k];
  }
  return value;
}

/// The best objective over every integer point meeting every row; NaN when there is none.
double enumerate(const Instance& instance) {
  const Model& model = instance.model;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  const bool maximize = model.sense == ObjSense::kMaximize;
  double best = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> x(n, 0.0);
  std::vector<double> activity(m);
  while (true) {
    std::fill(activity.begin(), activity.end(), 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      const ColumnView column = model.matrix.column(static_cast<Index>(j));
      for (Index k = 0; k < column.size; ++k) {
        activity[static_cast<std::size_t>(column.rows[k])] += column.values[k] * x[j];
      }
    }
    bool ok = true;
    for (std::size_t i = 0; i < m && ok; ++i) {
      ok = activity[i] >= model.row_lower[i] - 1e-9 && activity[i] <= model.row_upper[i] + 1e-9;
    }
    if (ok) {
      const double value = objective(instance, x);
      if (std::isnan(best) || (maximize ? value > best : value < best)) best = value;
    }
    std::size_t j = 0;
    while (j < n && x[j] >= model.col_upper[j]) {
      x[j] = model.col_lower[j];
      ++j;
    }
    if (j == n) break;
    x[j] += 1.0;
  }
  return best;
}

Options ipm_nodes() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("miqp_node_ipm", true);
  options.set_double("mip_relative_gap", 1e-9);
  options.set_double("mip_absolute_gap", 1e-9);
  return options;
}

TEST(MiqpNodeIpm, RandomConvexMiqpsReachTheEnumeratedOptimumOrAreInfeasible) {
  std::mt19937 rng(494);
  int optimal = 0;
  int infeasible = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const bool maximize = trial % 3 == 2;
    const Instance instance = random_convex_miqp(rng, maximize);
    const double best = enumerate(instance);
    const Solution solved = solve(instance.model, ipm_nodes());
    if (std::isnan(best)) {
      ++infeasible;
      EXPECT_EQ(solved.status, SolveStatus::kInfeasible)
          << "trial " << trial << ": " << solved.message;
      continue;
    }
    ++optimal;
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    EXPECT_NEAR(solved.objective, best, 1e-6 * std::max(1.0, std::fabs(best)))
        << "trial " << trial;
    EXPECT_NEAR(objective(instance, solved.col_value), best,
                1e-6 * std::max(1.0, std::fabs(best)))
        << "trial " << trial << ": the reported point does not have the reported objective";
  }
  // Neither half may pass vacuously.
  EXPECT_GT(optimal, 40);
  EXPECT_GT(infeasible, 3);
}

TEST(MiqpNodeIpm, ANodeLimitNeverReportsABoundPastTheOptimum) {
  std::mt19937 rng(4941);
  int limited = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Instance instance = random_convex_miqp(rng, false);
    const double best = enumerate(instance);
    if (std::isnan(best)) continue;
    Options options = ipm_nodes();
    options.set_int("node_limit", 3);
    const Solution solved = solve(instance.model, options);
    if (solved.status == SolveStatus::kOptimal) continue;
    ++limited;
    if (std::isfinite(solved.dual_bound)) {
      EXPECT_LE(solved.dual_bound, best + 1e-6 * std::max(1.0, std::fabs(best)))
          << "trial " << trial << ": a bound past the optimum";
    }
  }
  EXPECT_GT(limited, 3);
}

}  // namespace
}  // namespace sankhya
