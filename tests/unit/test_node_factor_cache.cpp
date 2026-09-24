// SPDX-License-Identifier: Apache-2.0
// SANKHYA - first factorizations kept across node LPs (#501).
//
// The claim the cache makes is that it changes no answer: a node LP that starts from factors
// it took from the cache continues exactly as it would have after factorizing that basis
// itself. So every test here compares against the same solve without a cache, to the bit,
// and checks separately that the cache was actually used - an equality that holds because
// nothing was ever reused would prove nothing.

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "simplex/factor_cache.hpp"
#include "simplex/primal_simplex.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

/// A bounded LP with enough coupled rows that the optimal basis holds structural columns.
Model coupled_lp() {
  Model model;
  const Index n = 10;
  const Index m = 6;
  model.col_cost.resize(static_cast<std::size_t>(n));
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 8.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(m), -kInfinity);
  model.row_upper.resize(static_cast<std::size_t>(m));
  model.matrix.reset(m, n);
  for (Index j = 0; j < n; ++j) {
    model.col_cost[static_cast<std::size_t>(j)] = -(3.0 + static_cast<double>((j * 7) % 5));
  }
  for (Index i = 0; i < m; ++i) {
    model.row_upper[static_cast<std::size_t>(i)] = 20.0 + static_cast<double>((i * 3) % 7);
    for (Index j = 0; j < n; ++j) {
      if ((i + 2 * j) % 3 == 0) continue;
      model.matrix.add_entry(i, j, 1.0 + static_cast<double>((i * 5 + j * 3) % 4));
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

WarmStart basis_of(const Solution& solved) {
  WarmStart warm;
  warm.col_status = solved.col_status;
  warm.row_status = solved.row_status;
  return warm;
}

void expect_same_solve(const Solution& a, const Solution& b, const char* what) {
  EXPECT_EQ(a.status, b.status) << what;
  EXPECT_EQ(a.iterations, b.iterations) << what;
  EXPECT_EQ(a.objective, b.objective) << what << ": to the bit";
  EXPECT_EQ(a.col_value, b.col_value) << what;
  EXPECT_EQ(a.row_dual, b.row_dual) << what;
  EXPECT_EQ(a.col_dual, b.col_dual) << what;
  EXPECT_EQ(a.col_status, b.col_status) << what;
  EXPECT_EQ(a.row_status, b.row_status) << what;
}

TEST(NodeFactorCache, AReusedFactorizationGivesTheSameSolveToTheBit) {
  const Model root = coupled_lp();
  const Options options = quiet();
  Logger logger(nullptr);
  const NodeScaling scaling = build_node_scaling(root, options);
  ASSERT_TRUE(scaling.valid);
  ASSERT_NE(scaling.id, 0u);

  const Solution root_solve = solve_dual_simplex(root, options, logger, scaling);
  ASSERT_EQ(root_solve.status, SolveStatus::kOptimal);
  const WarmStart warm = basis_of(root_solve);

  // Two "children": the same starting basis, different bounds on a fractional-ish column.
  // Bounds do not enter the factorization, so the second child can take the first's.
  Model down = root;
  Model up = root;
  Index moved = -1;
  for (Index j = 0; j < root.num_cols(); ++j) {
    const double v = root_solve.col_value[static_cast<std::size_t>(j)];
    if (v > 1e-6 && v < 8.0 - 1e-6) {
      moved = j;
      down.col_upper[static_cast<std::size_t>(j)] = std::floor(v);
      up.col_lower[static_cast<std::size_t>(j)] = std::floor(v) + 1.0;
      break;
    }
  }
  ASSERT_GE(moved, 0) << "the test LP needs an interior column to branch on";

  NodeFactorCache cache(4);
  const Solution down_cached =
      solve_dual_simplex(down, options, logger, scaling, nullptr, &warm, &cache);
  EXPECT_EQ(cache.hits(), 0);
  EXPECT_EQ(cache.misses(), 1);
  const Solution up_cached =
      solve_dual_simplex(up, options, logger, scaling, nullptr, &warm, &cache);
  EXPECT_EQ(cache.hits(), 1) << "the second child starts from the same basis";

  const Solution down_plain =
      solve_dual_simplex(down, options, logger, scaling, nullptr, &warm);
  const Solution up_plain = solve_dual_simplex(up, options, logger, scaling, nullptr, &warm);
  expect_same_solve(down_cached, down_plain, "down child");
  expect_same_solve(up_cached, up_plain, "up child (from reused factors)");

  // The primal path takes the same cache through the same prepare().
  const Solution primal_cached =
      solve_primal_simplex(up, options, logger, scaling, nullptr, &warm, &cache);
  EXPECT_EQ(cache.hits(), 2);
  const Solution primal_plain =
      solve_primal_simplex(up, options, logger, scaling, nullptr, &warm);
  expect_same_solve(primal_cached, primal_plain, "primal from reused factors");
}

TEST(NodeFactorCache, AnotherScaledMatrixIsNeverAHit) {
  const Model model = coupled_lp();
  const Options options = quiet();
  Logger logger(nullptr);
  const NodeScaling first = build_node_scaling(model, options);
  const NodeScaling second = build_node_scaling(model, options);
  EXPECT_NE(first.id, second.id) << "every build names its own matrix";
  const NodeScaling copy = first;
  EXPECT_EQ(copy.id, first.id) << "a copy holds the same matrix, so the same name";

  const Solution solved = solve_dual_simplex(model, options, logger, first);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal);
  const WarmStart warm = basis_of(solved);
  NodeFactorCache cache(4);
  (void)solve_dual_simplex(model, options, logger, first, nullptr, &warm, &cache);
  (void)solve_dual_simplex(model, options, logger, second, nullptr, &warm, &cache);
  EXPECT_EQ(cache.hits(), 0);
  (void)solve_dual_simplex(model, options, logger, copy, nullptr, &warm, &cache);
  EXPECT_EQ(cache.hits(), 1);
}

TEST(NodeFactorCache, TheLeastRecentlyUsedEntryIsDropped) {
  const Model model = coupled_lp();
  const Options options = quiet();
  Logger logger(nullptr);
  const NodeScaling scaling = build_node_scaling(model, options);
  const Solution solved = solve_dual_simplex(model, options, logger, scaling);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal);
  const WarmStart optimal = basis_of(solved);
  const WarmStart slack{};  // the slack basis: another starting basis

  NodeFactorCache one(1);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &optimal, &one);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &slack, &one);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &optimal, &one);
  EXPECT_EQ(one.hits(), 0) << "one entry cannot hold two alternating bases";

  NodeFactorCache two(2);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &optimal, &two);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &slack, &two);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &optimal, &two);
  EXPECT_EQ(two.hits(), 1);

  NodeFactorCache none(0);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &optimal, &none);
  (void)solve_dual_simplex(model, options, logger, scaling, nullptr, &optimal, &none);
  EXPECT_EQ(none.hits(), 0) << "capacity 0 keeps nothing";
}

/// A multi-row 0-1 knapsack: strong branching and a real tree in well under a second.
Model multi_knapsack() {
  Model model;
  const Index n = 24;
  const Index m = 3;
  model.sense = ObjSense::kMaximize;
  model.col_cost.resize(static_cast<std::size_t>(n));
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  model.row_lower.assign(static_cast<std::size_t>(m), -kInfinity);
  model.row_upper.assign(static_cast<std::size_t>(m), 0.0);
  model.matrix.reset(m, n);
  for (Index j = 0; j < n; ++j) {
    double value = 0.0;
    for (Index i = 0; i < m; ++i) {
      const double w = 10.0 + static_cast<double>((j * 37 + i * 11) % 41);
      model.matrix.add_entry(i, j, w);
      model.row_upper[static_cast<std::size_t>(i)] += w;
      value += w;
    }
    model.col_cost[static_cast<std::size_t>(j)] = value / 3.0 + 5.0;
  }
  for (double& cap : model.row_upper) cap = std::floor(cap / 2.0);
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

TEST(NodeFactorCache, TheSearchIsTheSameTreeWithTheCacheOn) {
  const Model model = multi_knapsack();
  Options plain = quiet();
  plain.set_bool("deterministic", true);
  plain.set_int("node_limit", 400);
  const Solution off = solve(model, plain);
  ASSERT_GT(off.nodes, 10) << "the test needs a real tree";

  testing::TempFile file("", ".json");
  Options cached = plain;
  cached.set_int("mip_node_factor_cache", 8);
  cached.set_string("profile", "basic");
  cached.set_string("profile_out", file.path());
  const Solution on = solve(model, cached);

  EXPECT_EQ(on.status, off.status);
  EXPECT_EQ(on.nodes, off.nodes) << "the same tree, node for node";
  EXPECT_EQ(on.objective, off.objective) << "to the bit";
  EXPECT_EQ(on.dual_bound, off.dual_bound);
  EXPECT_EQ(on.col_value, off.col_value);

  std::ifstream in(file.path());
  std::stringstream text;
  text << in.rdbuf();
  ASSERT_FALSE(text.str().empty()) << "profile_out was not written";
  const nlohmann::json blob = nlohmann::json::parse(text.str());
  ASSERT_TRUE(blob["counters"].contains("node LP factorizations reused")) << blob.dump(2);
  EXPECT_GT(blob["counters"]["node LP factorizations reused"].get<std::int64_t>(), 0)
      << "an equality with nothing reused would prove nothing";
}

}  // namespace
}  // namespace sankhya
