// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the supernodal numeric factorization against the scalar one (#470).
//
// The supernodal path shares the analysis with the scalar up-looking path and writes its
// factor into the same layout, so the two can be compared entry by entry: every L(i, j)
// and every pivot must agree to rounding, the regularized-pivot count exactly, and the
// solves must agree with each other and with a dense LU that shares no code with either.
// Eigen is not used: it is not fetched by this build and this machine has no network for it;
// the dense LU (simplex/dense_lu.hpp) is the independent reference instead.

#include <algorithm>
#include <chrono>
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
#include "sankhya/sparse.hpp"
#include "simplex/dense_lu.hpp"

namespace sankhya {
namespace {

struct System {
  SparseMatrix lower;
  std::vector<double> dense;  // column-major n x n
  Index n = 0;
};

/// M = B B^T + shift I with B n x n of the given density, lower triangle and dense copy.
System random_spd(std::mt19937_64& rng, Index n, double density, double shift) {
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const auto N = static_cast<std::size_t>(n);
  std::vector<double> b(N * N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (unit(rng) < density) b[i + j * N] = value(rng);
    }
  }
  System s;
  s.n = n;
  s.dense.assign(N * N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      double sum = i == j ? shift : 0.0;
      for (std::size_t k = 0; k < N; ++k) sum += b[i + k * N] * b[j + k * N];
      s.dense[i + j * N] = sum;
    }
  }
  s.lower.reset(n, n);
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = j; i < N; ++i) {
      if (s.dense[i + j * N] != 0.0) {
        s.lower.add_entry(static_cast<Index>(i), static_cast<Index>(j), s.dense[i + j * N]);
      }
    }
  }
  s.lower.finalize(0.0);
  return s;
}

/// Factor `lower` both ways from the same analysis and compare everything stored.
void expect_same_factor(const SparseMatrix& lower, double regularization,
                        const std::string& what, double tolerance = 1e-12) {
  SparseLdl scalar;
  SparseLdl blocked;
  blocked.set_supernodal(true);
  ASSERT_TRUE(scalar.analyze(lower)) << what;
  ASSERT_TRUE(blocked.analyze(lower)) << what;
  ASSERT_EQ(scalar.permutation(), blocked.permutation()) << what;
  ASSERT_TRUE(scalar.factorize(lower, regularization)) << what;
  ASSERT_TRUE(blocked.factorize(lower, regularization)) << what;
  EXPECT_GT(blocked.supernode_count(), 0) << what;
  EXPECT_LE(blocked.supernode_count(), blocked.dimension()) << what;
  EXPECT_EQ(scalar.regularized_pivots(), blocked.regularized_pivots()) << what;
  const std::vector<double>& d1 = scalar.pivots();
  const std::vector<double>& d2 = blocked.pivots();
  ASSERT_EQ(d1.size(), d2.size());
  // Normwise: each difference against the largest entry of its kind. A pivot of order one
  // under entries of 1e8 (israel's normal equations) carries rounding of 1e-8 in either
  // order of summation, and is compared at the scale the arithmetic had.
  double worst_d = 0.0;
  double largest_d = 0.0;
  for (std::size_t k = 0; k < d1.size(); ++k) {
    worst_d = std::max(worst_d, std::fabs(d1[k] - d2[k]));
    largest_d = std::max(largest_d, std::fabs(d1[k]));
  }
  worst_d /= std::max(1.0, largest_d);
  EXPECT_LE(worst_d, tolerance) << what << ": pivots";
  const std::vector<double>& l1 = scalar.factor_values();
  const std::vector<double>& l2 = blocked.factor_values();
  ASSERT_EQ(l1.size(), l2.size());
  double worst_l = 0.0;
  double largest_l = 0.0;
  for (std::size_t p = 0; p < l1.size(); ++p) {
    worst_l = std::max(worst_l, std::fabs(l1[p] - l2[p]));
    largest_l = std::max(largest_l, std::fabs(l1[p]));
  }
  worst_l /= std::max(1.0, largest_l);
  EXPECT_LE(worst_l, tolerance) << what << ": L";
  // And the solves.
  std::mt19937_64 rng(static_cast<std::uint64_t>(lower.num_rows()));
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> x1(static_cast<std::size_t>(lower.num_rows()));
  for (double& v : x1) v = value(rng);
  std::vector<double> x2 = x1;
  scalar.solve(x1.data());
  blocked.solve(x2.data());
  double worst_x = 0.0;
  double size = 0.0;
  for (std::size_t i = 0; i < x1.size(); ++i) {
    worst_x = std::max(worst_x, std::fabs(x1[i] - x2[i]));
    size = std::max(size, std::fabs(x1[i]));
  }
  EXPECT_LE(worst_x, 1e3 * tolerance * std::max(1.0, size)) << what << ": solve";
}

TEST(SupernodalLdl, MatchesTheScalarFactorOnRandomSpdMatrices) {
  std::mt19937_64 rng(470);
  for (int trial = 0; trial < 60; ++trial) {
    const Index n = 5 + static_cast<Index>(trial * 3 % 120);
    const double density = trial % 3 == 0 ? 0.05 : (trial % 3 == 1 ? 0.2 : 0.6);
    const System s = random_spd(rng, n, density, 1.0);
    expect_same_factor(s.lower, 1e-12, "trial " + std::to_string(trial));
  }
}

TEST(SupernodalLdl, AgreesWithTheDenseLuOnRandomSpdMatrices) {
  std::mt19937_64 rng(4701);
  for (int trial = 0; trial < 30; ++trial) {
    const Index n = 4 + static_cast<Index>(trial * 5 % 90);
    const System s = random_spd(rng, n, 0.3, 1.0);
    SparseLdl ldl;
    ldl.set_supernodal(true);
    ASSERT_TRUE(ldl.analyze(s.lower));
    ASSERT_TRUE(ldl.factorize(s.lower, 1e-12));
    DenseLu dense;
    ASSERT_TRUE(dense.factorize(s.dense, n, 1e-14));
    std::uniform_real_distribution<double> value(-5.0, 5.0);
    std::vector<double> rhs(static_cast<std::size_t>(n));
    for (double& v : rhs) v = value(rng);
    std::vector<double> x1 = rhs;
    std::vector<double> x2 = rhs;
    ldl.solve(x1.data());
    dense.solve(x2.data());
    for (std::size_t i = 0; i < x1.size(); ++i) {
      EXPECT_NEAR(x1[i], x2[i], 1e-9 * std::max(1.0, std::fabs(x2[i]))) << "trial " << trial;
    }
  }
}

TEST(SupernodalLdl, RegularizesTheSamePivotsOnASingularMatrix) {
  // B B^T with B of rank n/2: half the pivots fall to the floor, in both paths alike.
  std::mt19937_64 rng(4702);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  const Index n = 60;
  const auto N = static_cast<std::size_t>(n);
  std::vector<double> b(N * (N / 2));
  for (double& v : b) v = value(rng);
  SparseMatrix lower(n, n);
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = j; i < N; ++i) {
      double sum = 0.0;
      for (std::size_t k = 0; k < N / 2; ++k) sum += b[i + k * N] * b[j + k * N];
      lower.add_entry(static_cast<Index>(i), static_cast<Index>(j), sum);
    }
  }
  lower.finalize(0.0);
  SparseLdl scalar;
  SparseLdl blocked;
  blocked.set_supernodal(true);
  ASSERT_TRUE(scalar.analyze(lower) && blocked.analyze(lower));
  ASSERT_TRUE(scalar.factorize(lower, 1e-8) && blocked.factorize(lower, 1e-8));
  EXPECT_GT(scalar.regularized_pivots(), 0);
  EXPECT_EQ(scalar.regularized_pivots(), blocked.regularized_pivots());
}

/// The normal equations A Theta A^T + I of a Netlib instance with Theta over four decades.
/// The shift of one on every row keeps M well conditioned, so that an entrywise comparison
/// of two factors computed in different orders measures the algorithm and not the
/// conditioning: with a shift of 1e-6 israel's M is singular to within that shift and its
/// two factors, both backward stable, differ by 6% in L.
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

TEST(SupernodalLdl, MatchesTheScalarFactorOnNetlibNormalEquations) {
  for (const char* name : {"afiro", "adlittle", "blend", "share2b", "scagr7", "stocfor1",
                           "israel", "bandm", "brandy", "ship04s", "scsd1", "25fv47"}) {
    // 1e-9 normwise: eleven of the twelve agree to 1e-10, israel's L to 1.5e-10 - its
    // normal equations span eight decades, and two orders of summation differ by that much.
    expect_same_factor(netlib_normal_equations(name), 1e-10, name, 1e-9);
  }
}

// Timing, printed for the record and asserted on nothing: the evidence for or against the
// blocked path's speed is a run on a large model, not a unit test on a small one.
TEST(SupernodalLdl, PrintsTheFactorizationTimesOfBothPathsOnLargerNetlibModels) {
  for (const char* name : {"25fv47", "pilot87", "dfl001"}) {
    const SparseMatrix lower = netlib_normal_equations(name);
    for (const bool supernodal : {false, true}) {
      SparseLdl ldl;
      ldl.set_supernodal(supernodal);
      ASSERT_TRUE(ldl.analyze(lower));
      const auto start = std::chrono::steady_clock::now();
      ASSERT_TRUE(ldl.factorize(lower, 1e-10));
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::cout << "ldl " << name << (supernodal ? " supernodal" : " scalar    ") << ": factor "
                << ldl.factor_nonzeros() << " nonzeros, "
                << (supernodal ? ldl.supernode_count() : 0) << " supernodes, " << seconds
                << " s\n";
    }
  }
}

}  // namespace
}  // namespace sankhya
