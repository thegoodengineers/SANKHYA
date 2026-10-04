// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the certified bracket on ||A||_2 (#482, src/la/operator_norm.hpp).
//
// The constant PDHG step (pdhg_constant_step) is a share of 1 / upper, and is only safe if
// upper really is at least ||A||_2. Each matrix here is small enough for the exact norm: the
// square root of the largest eigenvalue of A^T A from a cyclic Jacobi eigensolver (Golub &
// Van Loan, Matrix Computations, 4th ed., section 8.5), written in this file so the oracle
// shares no code with the bound. On every matrix the bound must satisfy
//
//     lower <= ||A||_2 <= upper,
//
// and the looseness upper / ||A||_2 is printed and held under a per-family ceiling, so a
// change that keeps the bound valid but makes it useless also fails. Families: dense
// Gaussian of three shapes (the worst case for an entrywise bound: || |A| ||_2 is of order n
// against sqrt(n) for ||A||_2), sparse random signs, dense nonnegative, signed rank one,
// diagonal, the first-difference matrix (norm 2 cos(pi / (2 (n + 1))) in closed form), a
// random graph's incidence matrix, and the nine committed Netlib instances after the
// engine's own scaling (build_scaling), which is the matrix the step is taken on.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "la/operator_norm.hpp"
#include "la/scaling.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

namespace sankhya {
namespace {

using Dense = std::vector<std::vector<double>>;  // row-major, rows x cols

SparseMatrix to_sparse(const Dense& a) {
  const auto rows = static_cast<Index>(a.size());
  const auto cols = rows == 0 ? Index{0} : static_cast<Index>(a[0].size());
  SparseMatrix matrix(rows, cols);
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      const double v = a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) matrix.add_entry(i, j, v);
    }
  }
  matrix.finalize(0.0);
  return matrix;
}

Dense to_dense(const SparseMatrix& matrix) {
  Dense a(static_cast<std::size_t>(matrix.num_rows()),
          std::vector<double>(static_cast<std::size_t>(matrix.num_cols()), 0.0));
  for (Index j = 0; j < matrix.num_cols(); ++j) {
    const ColumnView column = matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      a[static_cast<std::size_t>(column.rows[k])][static_cast<std::size_t>(j)] =
          column.values[k];
    }
  }
  return a;
}

/// Largest eigenvalue of the symmetric matrix s by cyclic Jacobi rotations (Golub & Van Loan
/// section 8.5.2 and 8.5.3), run until the off-diagonal mass is at rounding level.
double largest_eigenvalue(Dense s) {
  const std::size_t n = s.size();
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    double total = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = 0; q < n; ++q) {
        total += s[p][q] * s[p][q];
        if (p != q) off += s[p][q] * s[p][q];
      }
    }
    if (off <= 1e-30 * total) break;
    for (std::size_t p = 0; p + 1 < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        if (s[p][q] == 0.0) continue;
        const double tau = (s[q][q] - s[p][p]) / (2.0 * s[p][q]);
        const double t =
            (tau >= 0.0 ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double sn = t * c;
        for (std::size_t k = 0; k < n; ++k) {  // columns p and q
          const double skp = s[k][p];
          const double skq = s[k][q];
          s[k][p] = c * skp - sn * skq;
          s[k][q] = sn * skp + c * skq;
        }
        for (std::size_t k = 0; k < n; ++k) {  // rows p and q
          const double spk = s[p][k];
          const double sqk = s[q][k];
          s[p][k] = c * spk - sn * sqk;
          s[q][k] = sn * spk + c * sqk;
        }
      }
    }
  }
  double largest = 0.0;
  for (std::size_t p = 0; p < n; ++p) largest = std::max(largest, s[p][p]);
  return largest;
}

/// ||A||_2 exactly (to rounding): the square root of the largest eigenvalue of the smaller
/// of A^T A and A A^T.
double exact_norm(const Dense& a) {
  const std::size_t m = a.size();
  const std::size_t n = m == 0 ? 0 : a[0].size();
  const bool gram_of_columns = n <= m;
  const std::size_t k = gram_of_columns ? n : m;
  Dense g(k, std::vector<double>(k, 0.0));
  for (std::size_t p = 0; p < k; ++p) {
    for (std::size_t q = p; q < k; ++q) {
      double sum = 0.0;
      if (gram_of_columns) {
        for (std::size_t i = 0; i < m; ++i) sum += a[i][p] * a[i][q];
      } else {
        for (std::size_t j = 0; j < n; ++j) sum += a[p][j] * a[q][j];
      }
      g[p][q] = g[q][p] = sum;
    }
  }
  return std::sqrt(largest_eigenvalue(g));
}

/// Checks the bracket on one matrix and returns upper / exact. `known`, when positive, is a
/// closed-form norm the Jacobi oracle is checked against first.
double check(const std::string& name, const SparseMatrix& matrix, double ceiling,
             double known = 0.0) {
  const double exact = exact_norm(to_dense(matrix));
  if (known > 0.0) {
    EXPECT_NEAR(exact, known, 1e-12 * known) << name << ": the oracle against the closed form";
  }
  const OperatorNormBound bound = bound_spectral_norm(matrix, 30, 30, 7u);
  EXPECT_GE(bound.upper, exact) << name << ": the upper bound is below ||A||_2";
  EXPECT_LE(bound.lower, exact * (1.0 + 1e-12)) << name << ": the lower bound is above ||A||_2";
  EXPECT_GE(bound.frobenius, exact) << name;
  EXPECT_GE(bound.holder, exact) << name;
  EXPECT_GE(bound.collatz_wielandt, exact) << name;
  const double ratio = bound.upper / exact;
  EXPECT_LE(ratio, ceiling) << name << ": the bound is valid but looser than its ceiling";
  std::printf(
      "  %-26s %4d x %-4d  ||A||_2 %.6e  upper/exact %.4f  lower/exact %.6f  (F %.3f, H %.3f, "
      "CW %.3f)\n",
      name.c_str(), static_cast<int>(matrix.num_rows()), static_cast<int>(matrix.num_cols()),
      exact, ratio, bound.lower / exact, bound.frobenius / exact, bound.holder / exact,
      bound.collatz_wielandt / exact);
  return ratio;
}

Dense gaussian(std::size_t m, std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  Dense a(m, std::vector<double>(n));
  for (auto& row : a) {
    for (double& v : row) v = normal(rng);
  }
  return a;
}

TEST(OperatorNormBound, DenseGaussianMatricesTheEntrywiseWorstCase) {
  // || |A| ||_2 / ||A||_2 grows like sqrt(n) here, so this family has the loosest ceiling.
  check("gaussian 20x30", to_sparse(gaussian(20, 30, 1u)), 3.0);
  check("gaussian 50x50", to_sparse(gaussian(50, 50, 2u)), 4.0);
  check("gaussian 80x40", to_sparse(gaussian(80, 40, 3u)), 4.0);
}

TEST(OperatorNormBound, SparseRandomSigns) {
  std::mt19937 rng(11u);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  Dense a(100, std::vector<double>(80, 0.0));
  for (auto& row : a) {
    for (double& v : row) {
      if (uniform(rng) < 0.08) v = (uniform(rng) < 0.5 ? -1.0 : 1.0) * (0.5 + uniform(rng));
    }
  }
  check("sparse signs 100x80", to_sparse(a), 3.0);
}

TEST(OperatorNormBound, NonnegativeAndSignEquivalentMatricesAreTight) {
  // For a nonnegative matrix || |A| ||_2 = ||A||_2 and the Collatz-Wielandt ratio converges
  // to it from above; the same holds for a signed rank one, whose |A| is |u| |v|^T.
  std::mt19937 rng(5u);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  Dense positive(40, std::vector<double>(60));
  for (auto& row : positive) {
    for (double& v : row) v = uniform(rng);
  }
  check("dense nonnegative 40x60", to_sparse(positive), 1.0 + 1e-6);

  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> u(30);
  std::vector<double> w(45);
  for (double& value : u) value = normal(rng);
  for (double& value : w) value = normal(rng);
  Dense rank_one(30, std::vector<double>(45));
  double uu = 0.0;
  double ww = 0.0;
  for (std::size_t i = 0; i < 30; ++i) {
    uu += u[i] * u[i];
    for (std::size_t j = 0; j < 45; ++j) rank_one[i][j] = u[i] * w[j];
  }
  for (const double value : w) ww += value * value;
  check("signed rank one 30x45", to_sparse(rank_one), 1.0 + 1e-9, std::sqrt(uu * ww));
}

TEST(OperatorNormBound, MatricesWithAKnownNorm) {
  Dense diagonal(25, std::vector<double>(25, 0.0));
  for (std::size_t i = 0; i < 25; ++i) {
    diagonal[i][i] = (i % 2 == 0 ? 1.0 : -1.0) * (1.0 + 0.37 * static_cast<double>(i));
  }
  check("diagonal 25x25", to_sparse(diagonal), 1.0 + 1e-9, 1.0 + 0.37 * 24.0);

  // The first-difference matrix D, (n + 1) x n with D_jj = 1 and D_(j+1)j = -1: D^T D is
  // tridiag(-1, 2, -1), whose largest eigenvalue is 2 + 2 cos(pi / (n + 1)), so ||D||_2 =
  // 2 cos(pi / (2 (n + 1))). Its sign pattern is bipartite, so || |D| ||_2 = ||D||_2.
  const std::size_t n = 40;
  Dense difference(n + 1, std::vector<double>(n, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    difference[j][j] = 1.0;
    difference[j + 1][j] = -1.0;
  }
  const double pi = std::acos(-1.0);
  check("first difference 41x40", to_sparse(difference), 1.01,
        2.0 * std::cos(pi / (2.0 * static_cast<double>(n + 1))));

  // 5 I: one entry per row and column.
  Dense scaled_identity(12, std::vector<double>(12, 0.0));
  for (std::size_t i = 0; i < 12; ++i) scaled_identity[i][i] = 5.0;
  check("5 I 12x12", to_sparse(scaled_identity), 1.0 + 1e-9, 5.0);
}

TEST(OperatorNormBound, GraphIncidenceMatrix) {
  // Node-arc incidence of a random graph with 30 nodes and 90 arcs, the shape of a network
  // LP's constraint matrix: B B^T is the graph Laplacian, |B| |B|^T the signless one.
  std::mt19937 rng(17u);
  std::uniform_int_distribution<int> node(0, 29);
  Dense b(30, std::vector<double>(90, 0.0));
  for (std::size_t arc = 0; arc < 90; ++arc) {
    int tail = node(rng);
    int head = node(rng);
    while (head == tail) head = node(rng);
    b[static_cast<std::size_t>(tail)][arc] = 1.0;
    b[static_cast<std::size_t>(head)][arc] = -1.0;
  }
  check("incidence 30 nodes 90 arcs", to_sparse(b), 1.5);
}

TEST(OperatorNormBound, EmptyAndAllZeroMatrices) {
  SparseMatrix empty(4, 3);
  empty.finalize(0.0);
  const OperatorNormBound none = bound_spectral_norm(empty, 30, 30, 1u);
  EXPECT_EQ(none.upper, 0.0);
  EXPECT_EQ(none.lower, 0.0);
  SparseMatrix no_shape(0, 0);
  no_shape.finalize(0.0);
  const OperatorNormBound shapeless = bound_spectral_norm(no_shape, 30, 30, 1u);
  EXPECT_EQ(shapeless.upper, 0.0);
}

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

TEST(OperatorNormBound, TheNineNetlibInstancesAfterTheEnginesScaling) {
  // The matrix the PDHG step is taken on: Ruiz then Pock-Chambolle alpha = 1, as
  // src/pdhg/pdhg.cpp builds it, with its 30 power steps on A^T A and on |A|^T |A|.
  const char* const names[] = {"afiro",   "sc50a", "sc50b",    "adlittle", "blend",
                               "share2b", "sc105", "stocfor1", "israel"};
  double worst = 0.0;
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    std::vector<double> cost(model.col_cost.begin(), model.col_cost.end());
    const Scaling scaling = build_scaling(model, cost, 10);
    worst = std::max(worst, check(std::string("netlib ") + name, scaling.matrix, 2.0));
  }
  std::printf("  worst upper/exact over the nine: %.4f\n", worst);
}

}  // namespace
}  // namespace sankhya
