// SPDX-License-Identifier: Apache-2.0
// SANKHYA - formulation symmetry (#413).
//
// A symmetry-breaking row that is not implied by a real automorphism cuts off the optimum
// and nothing downstream can tell, so the detection is held to three things: on models
// whose symmetry is known by construction it reports that group's orbits; every generator
// it returns passes an independent, dense check that it is an automorphism; and a search
// with the rows on reports exactly what enumeration or the exact oracle says.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mip/symmetry.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Rows given densely; every column integer in [lower, upper].
Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost, double lower,
            double upper, bool maximize = false) {
  Model m;
  const auto n = cost.size();
  m.sense = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  m.col_cost = cost;
  m.col_lower.assign(n, lower);
  m.col_upper.assign(n, upper);
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

/// An automorphism check that shares nothing with the detector: dense matrices, compared
/// entry by entry after the permutation is applied.
bool dense_automorphism(const Model& m, const Permutation& p) {
  const auto n = static_cast<std::size_t>(m.num_cols());
  const auto rows = static_cast<std::size_t>(m.num_rows());
  if (p.columns.size() != n || p.rows.size() != rows) return false;
  std::vector<std::vector<double>> a(rows, std::vector<double>(n, 0.0));
  for (Index j = 0; j < m.num_cols(); ++j) {
    const ColumnView view = m.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) {
      a[static_cast<std::size_t>(view.rows[k])][static_cast<std::size_t>(j)] = view.values[k];
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    const auto w = static_cast<std::size_t>(p.columns[j]);
    if (m.col_cost[j] != m.col_cost[w] || m.col_lower[j] != m.col_lower[w] ||
        m.col_upper[j] != m.col_upper[w] || m.col_type[j] != m.col_type[w]) {
      return false;
    }
  }
  for (std::size_t i = 0; i < rows; ++i) {
    const auto s = static_cast<std::size_t>(p.rows[i]);
    if (m.row_lower[i] != m.row_lower[s] || m.row_upper[i] != m.row_upper[s]) return false;
    for (std::size_t j = 0; j < n; ++j) {
      if (a[i][j] != a[s][static_cast<std::size_t>(p.columns[j])]) return false;
    }
  }
  return true;
}

TEST(Symmetry, IdenticalColumnsFormOneOrbit) {
  // Four identical columns in a knapsack row beside two distinct ones: the symmetric group
  // on the four is the formulation's symmetry, and nothing moves the other two.
  const Model m = build({{3, 3, 3, 3, 5, 7}}, {-kInf}, {10.0}, {2, 2, 2, 2, 4, 6}, 0.0, 1.0,
                        /*maximize=*/true);
  const SymmetryGroup group = detect_symmetry(m, 1000);
  ASSERT_FALSE(group.generators.empty());
  EXPECT_FALSE(group.budget_exhausted);
  EXPECT_EQ(group.nontrivial_orbits, 1);
  EXPECT_EQ(group.largest_orbit, 4);
  for (Index j = 1; j < 4; ++j)
    EXPECT_EQ(group.orbit_of[static_cast<std::size_t>(j)], group.orbit_of[0]);
  EXPECT_EQ(group.orbit_of[4], 4);
  EXPECT_EQ(group.orbit_of[5], 5);
  for (const Permutation& p : group.generators) {
    EXPECT_TRUE(is_automorphism(m, p));
    EXPECT_TRUE(dense_automorphism(m, p));
    EXPECT_EQ(p.columns[4], 4);
    EXPECT_EQ(p.columns[5], 5);
  }
  // The ordering rows chain the orbit: every pair has its first column before its second.
  for (const auto& [i, k] : ordering_rows(group)) {
    EXPECT_LT(i, k);
    EXPECT_LT(k, 4);
  }
}

TEST(Symmetry, SwappedBlocksMoveRowsAsWellAsColumns) {
  // Two copies of the same two-column block on their own rows: the swap that exchanges the
  // blocks exchanges the rows too, and both orbits have two columns.
  const Model m =
      build({{1, 1, 0, 0}, {0, 0, 1, 1}}, {-kInf, -kInf}, {1.0, 1.0}, {1, 2, 1, 2}, 0.0, 1.0);
  const SymmetryGroup group = detect_symmetry(m, 1000);
  ASSERT_EQ(group.generators.size(), 1u);
  const Permutation& p = group.generators.front();
  EXPECT_EQ(p.columns[0], 2);
  EXPECT_EQ(p.columns[1], 3);
  EXPECT_EQ(p.rows[0], 1);
  EXPECT_EQ(p.rows[1], 0);
  EXPECT_TRUE(dense_automorphism(m, p));
  EXPECT_EQ(group.nontrivial_orbits, 2);
  EXPECT_EQ(group.largest_orbit, 2);
  const auto rows = ordering_rows(group);
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows.front().first, 0);
  EXPECT_EQ(rows.front().second, 2);
}

TEST(Symmetry, AModelWithoutSymmetryCostsNoSearch) {
  // Distinct costs everywhere: the first refinement is already discrete, no vertex is
  // individualised, and nothing is reported.
  const Model m =
      build({{3, 4, 5, 6}, {1, 0, 2, 0}}, {-kInf, -kInf}, {10.0, 2.0}, {1, 2, 3, 4}, 0.0, 1.0);
  const SymmetryGroup group = detect_symmetry(m, 1000);
  EXPECT_TRUE(group.generators.empty());
  EXPECT_EQ(group.search_nodes, 0);
  EXPECT_LE(group.refinements, 3);
  EXPECT_EQ(group.nontrivial_orbits, 0);
  EXPECT_TRUE(ordering_rows(group).empty());
}

TEST(Symmetry, TheBudgetStopsTheSearchWithExactGeneratorsOnly) {
  const Model m = build({{3, 3, 3, 3, 5, 7}}, {-kInf}, {10.0}, {2, 2, 2, 2, 4, 6}, 0.0, 1.0);
  const SymmetryGroup none = detect_symmetry(m, 0);
  EXPECT_TRUE(none.generators.empty());
  EXPECT_TRUE(none.budget_exhausted);
  const SymmetryGroup some = detect_symmetry(m, 3);
  EXPECT_TRUE(some.budget_exhausted);
  for (const Permutation& p : some.generators) EXPECT_TRUE(dense_automorphism(m, p));
}

// ---- Searches with the rows on, against enumeration and the exact oracle ----------------

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

/// A random binary model built to be symmetric: the columns come in groups whose members
/// share a cost, and each row treats every member of a group alike, so every permutation
/// within a group is an automorphism. A distinct column is added so not everything moves.
Model symmetric_model(std::mt19937& rng, int trial) {
  std::uniform_int_distribution<int> coefficient(0, 3);
  std::uniform_int_distribution<int> cost_value(1, 9);
  const int groups = 2 + trial % 2;
  const int width = 2 + trial % 3;
  const std::size_t n = static_cast<std::size_t>(groups * width) + 1;
  const std::size_t rows_count = 3;
  std::vector<std::vector<double>> rows(rows_count, std::vector<double>(n, 0.0));
  std::vector<double> lower(rows_count, -kInf);
  std::vector<double> upper(rows_count, kInf);
  for (std::size_t i = 0; i < rows_count; ++i) {
    double sum = 0.0;
    for (int g = 0; g < groups; ++g) {
      const double a = coefficient(rng);
      for (int w = 0; w < width; ++w) {
        rows[i][static_cast<std::size_t>(g * width + w)] = a;
        sum += a;
      }
    }
    rows[i][n - 1] = coefficient(rng);
    sum += rows[i][n - 1];
    if (i % 2 == 0) {
      upper[i] = std::floor(sum / 2.0);
    } else {
      lower[i] = std::floor(sum / 4.0);
    }
  }
  std::vector<double> cost(n);
  for (int g = 0; g < groups; ++g) {
    const double c = cost_value(rng);
    for (int w = 0; w < width; ++w) cost[static_cast<std::size_t>(g * width + w)] = c;
  }
  cost[n - 1] = cost_value(rng) + 10.0;
  return build(rows, lower, upper, cost, 0.0, 1.0, trial % 2 == 1);
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_bool("mip_symmetry", true);
  return options;
}

TEST(Symmetry, OrderingRowsKeepTheEnumeratedOptimum) {
  std::mt19937 rng(413);
  Count generators = 0;
  for (int trial = 0; trial < 120; ++trial) {
    const Model m = symmetric_model(rng, trial);
    const Solution solved = solve(m, quiet());
    const auto [any, best] = enumerate(m);
    if (!any) {
      EXPECT_EQ(solved.status, SolveStatus::kInfeasible) << "trial " << trial;
      continue;
    }
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << solved.message;
    EXPECT_NEAR(solved.objective, best, 1e-6) << "trial " << trial;
    EXPECT_TRUE(feasible(m, solved.col_value)) << "trial " << trial;
    generators += solved.symmetry_generators;
  }
  EXPECT_GT(generators, 0) << "no generator was found on models built to be symmetric";
}

TEST(Symmetry, DeterministicRunsRepeat) {
  std::mt19937 rng(4130);
  const Model m = symmetric_model(rng, 5);
  Options options = quiet();
  options.set_bool("deterministic", true);
  const Solution first = solve(m, options);
  const Solution second = solve(m, options);
  EXPECT_EQ(second.status, first.status);
  EXPECT_EQ(second.objective, first.objective);
  EXPECT_EQ(second.nodes, first.nodes);
  EXPECT_EQ(second.symmetry_generators, first.symmetry_generators);
  EXPECT_EQ(second.col_value, first.col_value);
}

bool agree(const oracle::OracleResult& exact, const Solution& got, std::string* why) {
  switch (exact.status) {
    case oracle::OracleStatus::kOptimal: {
      if (got.status != SolveStatus::kOptimal) {
        *why = "the exact optimum exists but the solver said " +
               std::string(to_string(got.status)) + ": " + got.message;
        return false;
      }
      const double expected = exact.objective.to_double();
      const double difference = std::fabs(got.objective - expected);
      if (difference > 1e-6 * std::max(1.0, std::fabs(expected))) {
        *why = "objectives differ by " + std::to_string(difference);
        return false;
      }
      return true;
    }
    case oracle::OracleStatus::kInfeasible:
      if (got.status != SolveStatus::kInfeasible) {
        *why = "exactly infeasible, the solver said " + std::string(to_string(got.status));
        return false;
      }
      return true;
    case oracle::OracleStatus::kUnbounded:
    case oracle::OracleStatus::kOverflow:
    case oracle::OracleStatus::kIterationLimit: return true;
  }
  return true;
}

/// Append an exact copy of a random column, which makes the transposition of the two an
/// automorphism of the augmented instance.
void duplicate_a_column(oracle::GeneratedLp* lp, std::mt19937_64& rng) {
  if (lp->num_cols == 0) return;
  std::uniform_int_distribution<Index> pick(0, lp->num_cols - 1);
  const auto source = static_cast<std::size_t>(pick(rng));
  for (auto& row : lp->a) row.push_back(row[source]);
  lp->c.push_back(lp->c[source]);
  lp->upper.push_back(lp->upper[source]);
  if (!lp->integral.empty()) lp->integral.push_back(lp->integral[source]);
  if (!lp->lower.empty()) lp->lower.push_back(lp->lower[source]);
  ++lp->num_cols;
}

TEST(Symmetry, RandomMilpsWithADuplicatedColumnAgreeWithTheExactOracle) {
  std::mt19937_64 rng(41300);
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  int compared = 0;
  Count generators = 0;
  for (int trial = 0; trial < 200; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 2 == 0 ? 1 : 0;
    }
    duplicate_a_column(&lp, rng);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;
    }
    Model model = oracle::to_model(lp);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.integral[static_cast<std::size_t>(j)] != 0) {
        model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      }
    }
    Options options = quiet();
    options.set_int("node_limit", 100000);
    const Solution got = solve(model, options);
    generators += got.symmetry_generators;
    std::string why;
    EXPECT_TRUE(agree(exact, got, &why)) << "trial " << trial << ": " << why << "\n"
                                         << lp.to_text();
    ++compared;
  }
  EXPECT_GT(compared, 100);
  EXPECT_GT(generators, 0) << "a duplicated column should be found on most instances";
}

TEST(Symmetry, TheOptionIsRegisteredAndOffByDefault) {
  const Options options;
  EXPECT_FALSE(options.get_bool("mip_symmetry"));
  EXPECT_EQ(options.get_int("mip_symmetry_search_limit"), 1000);
}

}  // namespace
}  // namespace sankhya::mip
