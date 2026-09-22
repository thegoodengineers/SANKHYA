// SPDX-License-Identifier: Apache-2.0
// SANKHYA - reduced-cost fixing and restarts (#418).
//
// Both change the tree the search explores and neither may change its answer: a bound
// tightened by a reduced cost the root did not prove, or a restart that dropped the wrong
// thing, would make the search prove the second-best answer optimal, and nothing downstream
// could tell. So every model here is also solved by enumeration, and the search must report
// exactly what enumeration says - status and optimum - with fixing on, with restarts on, in
// deterministic mode twice, and in the one place restarts are declined (a parallel tree).

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

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

/// A random covering / packing model with spread-out costs, so the root relaxation leaves
/// some columns nonbasic with a reduced cost worth fixing on once an incumbent exists.
Model random_model(std::mt19937& rng, int trial) {
  std::uniform_int_distribution<int> coefficient(0, 4);
  std::uniform_int_distribution<int> cost_value(1, 40);
  const std::size_t n = 8 + static_cast<std::size_t>(trial % 5);
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
  for (double& c : cost) c = cost_value(rng);
  return binary_model(rows, lower, upper, cost, trial % 2 == 1);
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
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

TEST(Restarts, ReducedCostFixingKeepsTheEnumeratedOptimum) {
  std::mt19937 rng(418);
  Count fixings = 0;
  for (int trial = 0; trial < 150; ++trial) {
    const Model m = random_model(rng, trial);
    Options options = quiet();
    options.set_bool("mip_reduced_cost_fixing", true);
    const Solution solved = solve(m, options);
    expect_enumerated(m, solved, trial, "fixing");
    fixings += solved.reduced_cost_fixings;
  }
  EXPECT_GT(fixings, 0) << "the generator should give the root something to fix";
}

TEST(Restarts, ARestartedSearchReachesTheEnumeratedOptimum) {
  // Heuristics on, so an incumbent exists early enough for the fixing to matter at the
  // root; the restart threshold is low so the small models here actually restart.
  std::mt19937 rng(4180);
  Count restarts = 0;
  for (int trial = 0; trial < 150; ++trial) {
    const Model m = random_model(rng, trial);
    Options options = quiet();
    options.set_bool("mip_heuristics", true);
    options.set_bool("mip_reduced_cost_fixing", true);
    options.set_int("mip_restarts", 2);
    options.set_double("mip_restart_fraction", 0.05);
    const Solution solved = solve(m, options);
    expect_enumerated(m, solved, trial, "restarts");
    restarts += solved.restarts;
    EXPECT_LE(solved.restarts, 2) << "trial " << trial;
  }
  EXPECT_GT(restarts, 0) << "no model restarted, so the restart path was not exercised";
}

TEST(Restarts, DeterministicRunsRepeatWithRestartsOn) {
  std::mt19937 rng(41800);
  const Model m = random_model(rng, 3);
  Options options = quiet();
  options.set_bool("deterministic", true);
  options.set_bool("mip_heuristics", true);
  options.set_bool("mip_reduced_cost_fixing", true);
  options.set_int("mip_restarts", 2);
  options.set_double("mip_restart_fraction", 0.05);
  const Solution first = solve(m, options);
  const Solution second = solve(m, options);
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  EXPECT_EQ(second.status, first.status);
  EXPECT_EQ(second.objective, first.objective);
  EXPECT_EQ(second.nodes, first.nodes);
  EXPECT_EQ(second.restarts, first.restarts);
  EXPECT_EQ(second.reduced_cost_fixings, first.reduced_cost_fixings);
  EXPECT_EQ(second.col_value, first.col_value);
}

TEST(Restarts, AParallelTreeDoesNotRestart) {
  // The tree belongs to the shared search (#222), so a worker never throws it away; the
  // answer must still be the enumerated one, and the fixing itself is allowed.
  std::mt19937 rng(418000);
  for (int trial = 0; trial < 20; ++trial) {
    const Model m = random_model(rng, trial);
    Options options = quiet();
    options.set_int("mip_threads", 2);
    options.set_bool("mip_reduced_cost_fixing", true);
    options.set_int("mip_restarts", 2);
    options.set_double("mip_restart_fraction", 0.0);
    const Solution solved = solve(m, options);
    expect_enumerated(m, solved, trial, "parallel");
    EXPECT_EQ(solved.restarts, 0) << "trial " << trial;
  }
}

TEST(Restarts, NothingIsFixedWhileThePoolAsksForAlternatives) {
  // A pool gap asks for solutions no better than the incumbent, which is exactly what
  // reduced-cost fixing removes, so fixing stands down and the pool is what it promised.
  std::mt19937 rng(4180000);
  for (int trial = 0; trial < 20; ++trial) {
    const Model m = random_model(rng, trial);
    Options options = quiet();
    options.set_bool("mip_reduced_cost_fixing", true);
    options.set_int("pool_size", 5);
    options.set_double("pool_gap", 0.5);
    const Solution solved = solve(m, options);
    expect_enumerated(m, solved, trial, "pool");
    EXPECT_EQ(solved.reduced_cost_fixings, 0) << "trial " << trial;
  }
}

}  // namespace
}  // namespace sankhya::mip
