// SPDX-License-Identifier: Apache-2.0
// SANKHYA - reduced-cost fixing and restarts (#418).
//
// Both change the tree the search explores and neither may change its answer: a bound
// tightened by a reduced cost the root did not prove, or a restart that dropped the wrong
// thing, would make the search prove the second-best answer optimal, and nothing downstream
// could tell. So every model here is also solved by enumeration, and the search must report
// exactly what enumeration says - status and optimum - with fixing on, with restarts on, in
// deterministic mode twice, and in the one place restarts are declined (a parallel tree).
// Past enumeration, a restarted search is held to the same search without restarts, and
// deterministic mode to its own earlier runs, bit for bit. The exact rational oracle sweeps
// with restarts on are in test_branch_and_bound.cpp beside the other MILP sweeps.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
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
  // One root round: these tests measure the tree, and on models this small the root
  // separation loop (#495, on by default) closes the gap before there is a tree.
  options.set_bool("root_cut_loop", false);
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

// ---- Larger models: past enumeration, so the reference is the search without restarts ----

/// A mixed-integer model too large to enumerate: general-integer columns with small boxes,
/// a few continuous columns, covering and packing rows. The integer columns outnumber the
/// continuous ones so the root has plenty to fix by reduced cost.
Model larger_model(std::mt19937& rng, int trial) {
  std::uniform_int_distribution<int> coefficient(0, 5);
  std::uniform_int_distribution<int> cost_value(1, 40);
  std::uniform_int_distribution<int> box(1, 3);
  const std::size_t n = 22 + static_cast<std::size_t>(trial % 7);
  const std::size_t continuous = 3;
  const std::size_t rows = 6;
  Model m;
  m.sense = trial % 2 == 1 ? ObjSense::kMaximize : ObjSense::kMinimize;
  m.col_cost.resize(n);
  m.col_lower.assign(n, 0.0);
  m.col_upper.resize(n);
  m.col_type.resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    m.col_cost[j] = cost_value(rng);
    const bool integer = j >= continuous;
    m.col_type[j] = integer ? VarType::kInteger : VarType::kContinuous;
    m.col_upper[j] = integer ? box(rng) : 2.5;
  }
  m.matrix.reset(static_cast<Index>(rows), static_cast<Index>(n));
  m.row_lower.assign(rows, -kInf);
  m.row_upper.assign(rows, kInf);
  for (std::size_t i = 0; i < rows; ++i) {
    double reach = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      const int a = coefficient(rng);
      if (a == 0) continue;
      m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), a);
      reach += a * m.col_upper[j];
    }
    if (i % 2 == 0) {
      m.row_upper[i] = std::floor(reach / 2.0);
    } else {
      m.row_lower[i] = std::floor(reach / 4.0);
    }
  }
  m.matrix.finalize();
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

/// Independent of everything under test: integer columns integral, every column in its box,
/// every row satisfied.
bool feasible_mixed(const Model& m, const std::vector<double>& x) {
  if (x.size() != static_cast<std::size_t>(m.num_cols())) return false;
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (m.col_type[j] == VarType::kInteger && std::fabs(x[j] - std::round(x[j])) > 1e-9) {
      return false;
    }
    if (x[j] < m.col_lower[j] - 1e-9 || x[j] > m.col_upper[j] + 1e-9) return false;
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    double activity = 0.0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      activity += m.matrix.at(i, j) * x[static_cast<std::size_t>(j)];
    }
    const auto u = static_cast<std::size_t>(i);
    if (activity < m.row_lower[u] - 1e-7 || activity > m.row_upper[u] + 1e-7) return false;
  }
  return true;
}

/// Closed gap targets: the search must exhaust its tree, so "optimal" means the optimum and
/// two runs that both say so must agree on the number.
Options exact_search() {
  Options options = quiet();
  options.set_bool("mip_heuristics", true);  // an incumbent early, so fixing has material
  options.set_double("mip_relative_gap", 0.0);
  options.set_double("mip_absolute_gap", 1e-9);
  options.set_int("node_limit", 200000);
  return options;
}

Options with_restarts(Options options) {
  options.set_bool("mip_reduced_cost_fixing", true);
  options.set_int("mip_restarts", 2);
  options.set_double("mip_restart_fraction", 0.05);
  return options;
}

TEST(Restarts, ARestartedSearchReachesTheSameOptimumAsOneWithout) {
  // #418's acceptance, literally: the same model solved with restarts off and on, both to a
  // closed gap. The models are past enumeration, so the reference is the search without
  // restarts (itself held to the exact oracle by the MILP sweeps in test_branch_and_bound.cpp);
  // the claim here is only that throwing the tree away and re-solving the root never moves
  // the answer. Only a model that actually restarted counts towards the evidence.
  std::mt19937 rng(41804);
  int restarted_models = 0;
  for (int trial = 0; trial < 30; ++trial) {
    const Model m = larger_model(rng, trial);
    const Solution plain = solve(m, exact_search());
    const Solution restarted = solve(m, with_restarts(exact_search()));
    EXPECT_EQ(plain.restarts, 0) << "trial " << trial;
    ASSERT_EQ(restarted.status, plain.status)
        << "trial " << trial << ": " << plain.message << " | " << restarted.message;
    if (plain.status != SolveStatus::kOptimal) continue;
    const double scale = std::max(1.0, std::fabs(plain.objective));
    EXPECT_NEAR(restarted.objective, plain.objective, 1e-9 * scale)
        << "trial " << trial << ", " << restarted.restarts << " restart(s)";
    EXPECT_NEAR(restarted.dual_bound, restarted.objective, 1e-6 * scale) << "trial " << trial;
    EXPECT_TRUE(feasible_mixed(m, plain.col_value)) << "trial " << trial;
    EXPECT_TRUE(feasible_mixed(m, restarted.col_value)) << "trial " << trial;
    EXPECT_LE(restarted.restarts, 2) << "trial " << trial;
    if (restarted.restarts > 0) ++restarted_models;
  }
  std::cout << "restarted " << restarted_models << " of 30 models, same optimum both ways"
            << std::endl;
  EXPECT_GE(restarted_models, 10)
      << "too few models restarted for the comparison to say anything about restarts";
}

/// The bits of a double: == says 0.0 equals -0.0 and NaN differs from itself, and neither is
/// what "bit for bit" means.
std::vector<std::uint64_t> bits(const std::vector<double>& values) {
  std::vector<std::uint64_t> out(values.size());
  for (std::size_t k = 0; k < values.size(); ++k) {
    std::memcpy(&out[k], &values[k], sizeof(double));
  }
  return out;
}

std::uint64_t bits(double value) {
  return bits(std::vector<double>{value}).front();
}

TEST(Restarts, DeterministicModeRepeatsBitForBitAcrossRunsThatRestart) {
  // deterministic (#288) promises the same numbers from the same model, options and build.
  // A restart is a decision taken between nodes on counts alone (columns fixed, nodes
  // explored), never on the clock, so it must keep that promise: three runs of each model,
  // every number compared by its bits, and only models whose runs actually restarted count.
  std::mt19937 rng(418418);
  int restarted_models = 0;
  for (int trial = 0; trial < 15; ++trial) {
    const Model m = larger_model(rng, trial);
    Options options = with_restarts(exact_search());
    options.set_bool("deterministic", true);
    const Solution first = solve(m, options);
    if (first.restarts == 0) continue;
    ++restarted_models;
    for (int run = 1; run < 3; ++run) {
      const Solution again = solve(m, options);
      const std::string what = "trial " + std::to_string(trial) + " run " + std::to_string(run);
      EXPECT_EQ(again.status, first.status) << what;
      EXPECT_EQ(bits(again.objective), bits(first.objective)) << what;
      EXPECT_EQ(bits(again.dual_bound), bits(first.dual_bound)) << what;
      EXPECT_EQ(bits(again.root_bound), bits(first.root_bound)) << what;
      EXPECT_EQ(bits(again.col_value), bits(first.col_value)) << what << ": the points differ";
      EXPECT_EQ(again.nodes, first.nodes) << what;
      EXPECT_EQ(again.iterations, first.iterations) << what;
      EXPECT_EQ(again.restarts, first.restarts) << what;
      EXPECT_EQ(again.reduced_cost_fixings, first.reduced_cost_fixings) << what;
      EXPECT_EQ(again.cuts_applied, first.cuts_applied) << what;
    }
  }
  std::cout << "restarted " << restarted_models << " of 15 models, three runs each, same bits"
            << std::endl;
  EXPECT_GE(restarted_models, 5)
      << "too few deterministic runs restarted for the repeat to say anything about restarts";
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
