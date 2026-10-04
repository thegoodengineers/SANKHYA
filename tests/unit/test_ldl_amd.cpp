// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the supervariable ordering against the unweighted one (#471).
//
// Any ordering is a permutation and a factorization under a permutation is exact, so what
// this checks is that the new ordering (a) is always a permutation, (b) feeds a factorization
// whose solve agrees with the one from the oracle ordering and leaves a small residual, (c)
// merges variables where the structure has twins and finds none where it has none, (d) is
// deterministic, (e) still honours the deadline and the memory budget, and (f) what it does
// to the fill on the Netlib normal equations, printed and not asserted against the oracle:
// minimum degree is a heuristic, and neither ordering wins everywhere.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "la/ldl.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

using Edges = std::vector<std::pair<Index, Index>>;

/// A symmetric, strictly diagonally dominant (so positive definite) matrix on the graph
/// `edges`, as a lower triangle. Off-diagonals are -1, the diagonal is degree + 1.
SparseMatrix laplacian_plus_identity(Index n, const Edges& edges) {
  std::vector<std::vector<Index>> neighbours(static_cast<std::size_t>(n));
  for (const auto& [a, b] : edges) {
    if (a == b) continue;
    neighbours[static_cast<std::size_t>(a)].push_back(b);
    neighbours[static_cast<std::size_t>(b)].push_back(a);
  }
  SparseMatrix lower;
  lower.reset(n, n);
  for (Index i = 0; i < n; ++i) {
    auto& list = neighbours[static_cast<std::size_t>(i)];
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    lower.add_entry(i, i, static_cast<double>(list.size()) + 1.0);
    for (const Index j : list) {
      if (j < i) lower.add_entry(i, j, -1.0);
    }
  }
  lower.finalize(0.0);
  return lower;
}

Edges grid(Index side) {
  Edges edges;
  for (Index r = 0; r < side; ++r) {
    for (Index c = 0; c < side; ++c) {
      if (c + 1 < side) edges.emplace_back(r * side + c, r * side + c + 1);
      if (r + 1 < side) edges.emplace_back(r * side + c, (r + 1) * side + c);
    }
  }
  return edges;
}

Edges random_graph(std::mt19937_64& rng, Index n, double average_degree) {
  std::uniform_int_distribution<Index> node(0, n - 1);
  Edges edges;
  const auto count = static_cast<std::size_t>(average_degree * static_cast<double>(n) / 2.0);
  for (std::size_t e = 0; e < count; ++e) edges.emplace_back(node(rng), node(rng));
  return edges;
}

/// `groups` cliques of `size` nodes each, the cliques joined in a ring by one node from each
/// to the next: inside a clique every node but the two joiners is a twin of the others.
Edges clique_ring(Index groups, Index size) {
  Edges edges;
  for (Index g = 0; g < groups; ++g) {
    const Index base = g * size;
    for (Index a = 0; a < size; ++a) {
      for (Index b = a + 1; b < size; ++b) edges.emplace_back(base + a, base + b);
    }
    edges.emplace_back(base, ((g + 1) % groups) * size + 1);
  }
  return edges;
}

/// Every node has a copy with the same closed neighbourhood: the graph `edges` with each node
/// doubled, the two copies adjacent to each other and to each other's neighbours.
Edges doubled(Index n, const Edges& edges) {
  Edges out;
  for (Index i = 0; i < n; ++i) out.emplace_back(2 * i, 2 * i + 1);
  for (const auto& [a, b] : edges) {
    for (Index x = 0; x < 2; ++x) {
      for (Index y = 0; y < 2; ++y) out.emplace_back(2 * a + x, 2 * b + y);
    }
  }
  return out;
}

void expect_permutation(const std::vector<Index>& perm, Index n, const std::string& what) {
  ASSERT_EQ(static_cast<Index>(perm.size()), n) << what;
  std::vector<bool> seen(static_cast<std::size_t>(n), false);
  for (const Index p : perm) {
    ASSERT_GE(p, 0) << what;
    ASSERT_LT(p, n) << what;
    ASSERT_FALSE(seen[static_cast<std::size_t>(p)]) << what << ": index " << p << " twice";
    seen[static_cast<std::size_t>(p)] = true;
  }
}

/// Analyse, factorize and solve A x = b both ways. The solutions must agree, and each must
/// leave a small residual against A itself (the lower triangle applied symmetrically).
void expect_same_solve(const SparseMatrix& lower, const std::string& what) {
  const Index n = lower.num_rows();
  std::vector<double> b(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + 0.5 * std::sin(i);
  std::vector<std::vector<double>> answers;
  for (const bool weighted : {false, true}) {
    SparseLdl ldl;
    ldl.set_supervariables(weighted);
    ASSERT_TRUE(ldl.analyze(lower)) << what;
    expect_permutation(ldl.permutation(), n, what);
    ASSERT_TRUE(ldl.factorize(lower, 0.0)) << what;
    std::vector<double> x = b;
    ldl.solve(x.data());
    std::vector<double> r(b.size(), 0.0);
    for (Index c = 0; c < n; ++c) {
      const ColumnView column = lower.column(c);
      for (Index p = 0; p < column.size; ++p) {
        const Index row = column.rows[p];
        r[static_cast<std::size_t>(row)] += column.values[p] * x[static_cast<std::size_t>(c)];
        if (row != c)
          r[static_cast<std::size_t>(c)] += column.values[p] * x[static_cast<std::size_t>(row)];
      }
    }
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      worst = std::max(worst, std::fabs(r[i] - b[i]));
      scale = std::max(scale, std::fabs(b[i]));
    }
    EXPECT_LE(worst / scale, 1e-10) << what << (weighted ? " (supervariables)" : " (oracle)");
    answers.push_back(std::move(x));
  }
  double worst = 0.0;
  double largest = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    worst = std::max(worst, std::fabs(answers[0][i] - answers[1][i]));
    largest = std::max(largest, std::fabs(answers[0][i]));
  }
  EXPECT_LE(worst / std::max(1.0, largest), 1e-10) << what << ": the two solves differ";
}

TEST(SupervariableAmd, IsOffByDefault) {
  SparseLdl ldl;
  EXPECT_FALSE(ldl.supervariables());
  const SparseMatrix lower = laplacian_plus_identity(30, grid(5));
  ASSERT_TRUE(ldl.analyze(lower));
  EXPECT_EQ(ldl.variables_absorbed(), 0) << "the unweighted ordering absorbs nothing";
}

// variables_absorbed() describes the last analyze(): an unweighted ordering after a weighted
// one, or a weighted one stopped by the deadline, must not report the earlier count.
TEST(SupervariableAmd, TheAbsorbedCountIsTheLastAnalysisOnly) {
  const SparseMatrix lower = laplacian_plus_identity(40, clique_ring(1, 40));
  SparseLdl ldl;
  ldl.set_supervariables(true);
  ASSERT_TRUE(ldl.analyze(lower));
  ASSERT_EQ(ldl.variables_absorbed(), 39);
  ASSERT_FALSE(ldl.analyze(lower, [] { return true; }));
  EXPECT_EQ(ldl.variables_absorbed(), 0) << "a stopped ordering absorbed nothing";
  ASSERT_TRUE(ldl.analyze(lower));
  ASSERT_EQ(ldl.variables_absorbed(), 39);
  ldl.set_supervariables(false);
  ASSERT_TRUE(ldl.analyze(lower));
  EXPECT_EQ(ldl.variables_absorbed(), 0) << "the unweighted ordering absorbs nothing";
}

TEST(SupervariableAmd, TheOrderingIsAPermutationAndTheSolveMatchesTheOracleOnManyShapes) {
  std::mt19937_64 rng(471);
  expect_same_solve(laplacian_plus_identity(1, {}), "one node");
  expect_same_solve(laplacian_plus_identity(50, {}), "diagonal matrix, no edges");
  expect_same_solve(laplacian_plus_identity(40, grid(1)), "single node grid");
  expect_same_solve(laplacian_plus_identity(400, grid(20)), "20 x 20 grid");
  expect_same_solve(laplacian_plus_identity(36, clique_ring(6, 6)), "ring of cliques");
  expect_same_solve(laplacian_plus_identity(200, doubled(100, grid(10))), "doubled grid");
  for (int trial = 0; trial < 60; ++trial) {
    const Index n = 5 + static_cast<Index>(rng() % 120);
    const double degree = 0.5 + 6.0 * static_cast<double>(rng() % 100) / 100.0;
    expect_same_solve(laplacian_plus_identity(n, random_graph(rng, n, degree)),
                      "random graph " + std::to_string(trial) + ", n = " + std::to_string(n));
  }
}

TEST(SupervariableAmd, FindsTheTwinsAThatStructureHasAndNoneWhereThereAreNone) {
  // A path has no two nodes with the same closed neighbourhood, but its two ends each end up
  // with nothing but a pivot's clique, which mass elimination takes with it; the count is
  // therefore small but not zero, and the factor must still be the no-fill one.
  const Index n = 500;
  Edges path;
  for (Index i = 0; i + 1 < n; ++i) path.emplace_back(i, i + 1);
  SparseLdl on_path;
  on_path.set_supervariables(true);
  ASSERT_TRUE(on_path.analyze(laplacian_plus_identity(n, path)));
  EXPECT_EQ(on_path.factor_nonzeros(), n - 1) << "a path must not fill";

  // Every node doubled: at least half of the variables are the second copy of one.
  const Index m = 144;
  SparseLdl twins;
  twins.set_supervariables(true);
  ASSERT_TRUE(twins.analyze(laplacian_plus_identity(2 * m, doubled(m, grid(12)))));
  EXPECT_GE(twins.variables_absorbed(), m / 2)
      << "twin nodes are not being merged: " << twins.variables_absorbed() << " absorbed of "
      << 2 * m;

  // A clique of k nodes is one supervariable: k - 1 are absorbed.
  const Index k = 40;
  SparseLdl clique;
  clique.set_supervariables(true);
  ASSERT_TRUE(clique.analyze(laplacian_plus_identity(k, clique_ring(1, k))));
  EXPECT_EQ(clique.variables_absorbed(), k - 1);
}

TEST(SupervariableAmd, TheSamePermutationOnEveryRun) {
  std::mt19937_64 rng(1996);
  const SparseMatrix lower = laplacian_plus_identity(300, random_graph(rng, 300, 4.0));
  SparseLdl a;
  SparseLdl b;
  a.set_supervariables(true);
  b.set_supervariables(true);
  ASSERT_TRUE(a.analyze(lower));
  ASSERT_TRUE(b.analyze(lower));
  EXPECT_EQ(a.permutation(), b.permutation());
}

TEST(SupervariableAmd, StillHonoursTheDeadlineAndTheOrderingBudget) {
  std::mt19937_64 rng(7);
  const SparseMatrix lower = laplacian_plus_identity(400, random_graph(rng, 400, 8.0));
  SparseLdl stopped;
  stopped.set_supervariables(true);
  EXPECT_FALSE(stopped.analyze(lower, [] { return true; }));

  SparseLdl capped;
  capped.set_supervariables(true);
  capped.set_ordering_budget(10);
  EXPECT_FALSE(capped.analyze(lower));
  EXPECT_TRUE(capped.ordering_too_large());
}

SparseMatrix netlib_normal_equations(const std::string& name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (name + ".mps"))
          .string();
  EXPECT_TRUE(io::read_model(path, &model).ok) << path;
  std::mt19937_64 rng(static_cast<std::uint64_t>(model.num_cols()));
  std::uniform_real_distribution<double> decades(-2.0, 2.0);
  std::vector<double> theta(static_cast<std::size_t>(model.num_cols()));
  for (double& t : theta) t = std::pow(10.0, decades(rng));
  const std::vector<double> shift(static_cast<std::size_t>(model.num_rows()), 1.0);
  SparseMatrix lower;
  EXPECT_TRUE(normal_equations_lower(model.matrix, theta, shift, 1e-10, &lower));
  return lower;
}

// What the ordering does to the fill on real structure, printed for the record. Asserted: the
// solves agree. Not asserted: that the fill is smaller, because it is a heuristic and the
// numbers below are the evidence either way.
TEST(SupervariableAmd, PrintsTheFillOfBothOrderingsOnNetlibNormalEquations) {
  for (const char* name :
       {"afiro", "adlittle", "blend", "share2b", "scagr7", "stocfor1", "israel", "bandm",
        "brandy", "ship04s", "scsd1", "25fv47", "pilot87", "dfl001"}) {
    const SparseMatrix lower = netlib_normal_equations(name);
    SparseLdl oracle;
    SparseLdl weighted;
    weighted.set_supervariables(true);
    ASSERT_TRUE(oracle.analyze(lower)) << name;
    ASSERT_TRUE(weighted.analyze(lower)) << name;
    expect_permutation(weighted.permutation(), lower.num_rows(), name);
    std::cout << "amd " << name << ": n " << lower.num_rows() << ", factor nonzeros "
              << oracle.factor_nonzeros() << " (oracle) -> " << weighted.factor_nonzeros()
              << " (supervariables), absorbed " << weighted.variables_absorbed() << "\n";
  }
}

// The option, end to end: the interior point reaches the same optimum with the ordering on.
TEST(SupervariableAmd, TheInteriorPointFindsTheSameOptimumWithTheOptionOn) {
  for (const char* name : {"afiro", "adlittle", "share2b", "bandm"}) {
    Model model;
    const std::string path =
        (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
         "data/netlib" / (std::string(name) + ".mps"))
            .string();
    ASSERT_TRUE(io::read_model(path, &model).ok) << path;
    double objective[2] = {0.0, 0.0};
    for (int on = 0; on < 2; ++on) {
      Options options;
      options.set_bool("log_to_console", false);
      options.set_string("algorithm", "ipm");
      options.set_bool("ipm_amd_supervariables", on == 1);
      const Solution solution = solve(model, options);
      ASSERT_EQ(solution.status, SolveStatus::kOptimal) << name << " option " << on;
      objective[on] = solution.objective;
    }
    EXPECT_NEAR(objective[0], objective[1], 1e-6 * std::max(1.0, std::fabs(objective[0])))
        << name;
  }
}

}  // namespace
}  // namespace sankhya
