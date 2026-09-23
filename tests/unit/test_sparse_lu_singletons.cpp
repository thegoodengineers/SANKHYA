// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the singleton fast path of the sparse LU (#463).
//
// A column-singleton pivot takes its U row from a row-wise copy of the values instead of
// opening every column its row crosses, and a row candidate's threshold test reads a cached
// column maximum instead of rescanning the column (Suhl & Suhl 1990; Koberstein 2005,
// sec. 5.3). The claim in lu_eliminate.cpp is stronger than "an equivalent factorization":
// it is the SAME one, pivot for pivot and bit for bit, as the elimination before #463. So
// every matrix here is factorized three ways:
//
//   1. the fast path (the default);
//   2. the reference elimination (use_reference_elimination), which is the code as it was;
//   3. DenseLu, the independent oracle (lu.hpp, VERIFICATION).
//
// 1 and 2 must agree EXACTLY: the same success or failure, the same pivot sequence, the same
// factor sizes and pivot magnitudes, the same rank defect when singular, and FTRAN and BTRAN
// results equal to the last bit. 1 must also agree with 3 to rounding, and reproduce the
// right-hand side through the original matrix.
//
// The matrices are built to reach the fast path's cases: a permuted diagonal (column
// singletons once the off-diagonals are stripped), planted dense columns (the #463 shape:
// bdry2 has one column with 126,002 entries in a slack basis), planted row singletons,
// values drawn from a small set so that ties in the pivot search are common - a tie broken
// differently is exactly how a "same factorization" claim would fail - and values spread
// over six decades so that the threshold test rejects candidates, which is the only way a
// wrong cached column maximum can change a pivot. Both were checked by breaking the code on
// purpose (reversing the fast path's column order; never invalidating the cached maximum):
// the fuzz test below fails on each.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

#include "la/lu.hpp"
#include "simplex/dense_lu.hpp"

namespace sankhya {
namespace {

constexpr double kThreshold = tol::kMarkowitzThreshold;

/// A square matrix as sparse columns (SparseLu) and dense column-major (DenseLu).
struct Matrix {
  Index m = 0;
  std::vector<std::vector<Index>> rows;
  std::vector<std::vector<double>> values;

  explicit Matrix(Index dimension)
      : m(dimension),
        rows(static_cast<std::size_t>(dimension)),
        values(static_cast<std::size_t>(dimension)) {}

  void set(Index row, Index col, double value) {
    rows[static_cast<std::size_t>(col)].push_back(row);
    values[static_cast<std::size_t>(col)].push_back(value);
  }

  [[nodiscard]] std::vector<LuColumn> columns() const {
    std::vector<LuColumn> out(static_cast<std::size_t>(m));
    for (std::size_t j = 0; j < out.size(); ++j) {
      out[j].size = static_cast<Index>(rows[j].size());
      if (out[j].size > 0) {
        out[j].rows = rows[j].data();
        out[j].values = values[j].data();
      }
    }
    return out;
  }

  [[nodiscard]] std::vector<double> dense() const {
    const auto n = static_cast<std::size_t>(m);
    std::vector<double> out(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < rows[j].size(); ++k) {
        out[j * n + static_cast<std::size_t>(rows[j][k])] += values[j][k];
      }
    }
    return out;
  }

  [[nodiscard]] std::vector<double> multiply(const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(m), 0.0);
    for (std::size_t j = 0; j < rows.size(); ++j) {
      for (std::size_t k = 0; k < rows[j].size(); ++k) {
        y[static_cast<std::size_t>(rows[j][k])] += values[j][k] * x[j];
      }
    }
    return y;
  }

  [[nodiscard]] std::vector<double> transpose_multiply(const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(m), 0.0);
    for (std::size_t j = 0; j < rows.size(); ++j) {
      for (std::size_t k = 0; k < rows[j].size(); ++k) {
        y[j] += values[j][k] * x[static_cast<std::size_t>(rows[j][k])];
      }
    }
    return y;
  }
};

[[nodiscard]] bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

[[nodiscard]] double relative_difference(const std::vector<double>& a,
                                         const std::vector<double>& b) {
  double scale = 1.0;
  for (const double v : b) scale = std::max(scale, std::fabs(v));
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
  return worst / scale;
}

/// Normwise backward error of x as a solution of A x = b (or A^T x = b):
/// ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf), with ||A||_inf taken as the largest
/// absolute row sum of whichever of A, A^T was solved. Unlike the forward error it does not
/// grow with the condition number, so it is the measure that means something on the
/// ill-conditioned draws of Values::kWide.
[[nodiscard]] double backward_error(const std::vector<double>& product,
                                    const std::vector<double>& x, const std::vector<double>& b,
                                    double matrix_norm) {
  double r = 0.0;
  double xn = 0.0;
  double bn = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    r = std::max(r, std::fabs(product[i] - b[i]));
    xn = std::max(xn, std::fabs(x[i]));
    bn = std::max(bn, std::fabs(b[i]));
  }
  const double denominator = matrix_norm * xn + bn;
  return denominator > 0.0 ? r / denominator : r;
}

struct Worst {
  double backward = 0.0;  ///< worst backward error, FTRAN and BTRAN, every nonsingular draw
  double oracle_ftran = 0.0;
  double oracle_btran = 0.0;
  double residual = 0.0;
  double transpose_residual = 0.0;
  int compared_with_oracle = 0;
  int singular = 0;
};

/// Factorize `matrix` both ways, require them identical, and (when nonsingular) compare the
/// fast path against DenseLu and the residual. Returns false on the first disagreement,
/// having already reported it through gtest.
bool check_matrix(const Matrix& matrix, std::mt19937& rng, Worst& worst, int trial) {
  const Index m = matrix.m;
  SparseLu fast;
  SparseLu reference;
  reference.use_reference_elimination(true);
  const bool fast_ok = fast.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold);
  const bool reference_ok =
      reference.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold);

  EXPECT_EQ(fast_ok, reference_ok) << "trial " << trial;
  if (fast_ok != reference_ok) return false;
  EXPECT_EQ(fast.pivot_rows(), reference.pivot_rows()) << "trial " << trial;
  EXPECT_EQ(fast.pivot_columns(), reference.pivot_columns()) << "trial " << trial;
  if (fast.pivot_rows() != reference.pivot_rows() ||
      fast.pivot_columns() != reference.pivot_columns()) {
    return false;
  }
  if (!fast_ok) {
    ++worst.singular;
    EXPECT_EQ(fast.dependent_positions(), reference.dependent_positions()) << "trial " << trial;
    EXPECT_EQ(fast.uncovered_rows(), reference.uncovered_rows()) << "trial " << trial;
    return fast.dependent_positions() == reference.dependent_positions() &&
           fast.uncovered_rows() == reference.uncovered_rows();
  }
  EXPECT_EQ(fast.factor_nonzeros(), reference.factor_nonzeros()) << "trial " << trial;
  EXPECT_EQ(fast.smallest_pivot(), reference.smallest_pivot()) << "trial " << trial;
  EXPECT_EQ(fast.largest_pivot(), reference.largest_pivot()) << "trial " << trial;

  std::uniform_real_distribution<double> value(-5.0, 5.0);
  std::vector<double> x(static_cast<std::size_t>(m));
  for (double& v : x) v = value(rng);
  const std::vector<double> b = matrix.multiply(x);
  const std::vector<double> bt = matrix.transpose_multiply(x);

  std::vector<double> fast_x = b;
  std::vector<double> reference_x = b;
  fast.solve(fast_x.data());
  reference.solve(reference_x.data());
  EXPECT_TRUE(bitwise_equal(fast_x, reference_x)) << "FTRAN differs, trial " << trial;

  std::vector<double> fast_y = bt;
  std::vector<double> reference_y = bt;
  fast.solve_transpose(fast_y.data());
  reference.solve_transpose(reference_y.data());
  EXPECT_TRUE(bitwise_equal(fast_y, reference_y)) << "BTRAN differs, trial " << trial;

  std::vector<double> row_sum(static_cast<std::size_t>(m), 0.0);
  double column_sum_max = 0.0;
  for (std::size_t j = 0; j < matrix.rows.size(); ++j) {
    double column_sum = 0.0;
    for (std::size_t k = 0; k < matrix.rows[j].size(); ++k) {
      row_sum[static_cast<std::size_t>(matrix.rows[j][k])] += std::fabs(matrix.values[j][k]);
      column_sum += std::fabs(matrix.values[j][k]);
    }
    column_sum_max = std::max(column_sum_max, column_sum);
  }
  const double row_sum_max = *std::max_element(row_sum.begin(), row_sum.end());
  worst.backward =
      std::max(worst.backward, backward_error(matrix.multiply(fast_x), fast_x, b, row_sum_max));
  worst.backward = std::max(worst.backward, backward_error(matrix.transpose_multiply(fast_y),
                                                           fast_y, bt, column_sum_max));

  worst.residual = std::max(worst.residual, relative_difference(matrix.multiply(fast_x), b));
  worst.transpose_residual = std::max(
      worst.transpose_residual, relative_difference(matrix.transpose_multiply(fast_y), bt));

  DenseLu dense;
  if (m <= 120 && dense.factorize(matrix.dense(), m, tol::kPivotTolerance)) {
    ++worst.compared_with_oracle;
    std::vector<double> dense_x = b;
    dense.solve(dense_x.data());
    std::vector<double> dense_y = bt;
    dense.solve_transpose(dense_y.data());
    worst.oracle_ftran = std::max(worst.oracle_ftran, relative_difference(fast_x, dense_x));
    worst.oracle_btran = std::max(worst.oracle_btran, relative_difference(fast_y, dense_y));
  }
  return bitwise_equal(fast_x, reference_x) && bitwise_equal(fast_y, reference_y);
}

/// A random basis-shaped matrix: a permuted diagonal, sparse off-diagonal entries, some
/// planted dense columns, some columns and rows stripped to singletons.
/// How the values are drawn. kTies makes the pivot search's tie-breaking decide; kWide
/// spreads magnitudes over six decades so the threshold test |a_ij| >= tau max_i |a_ij|
/// rejects candidates, which is the only way a wrong cached column maximum can show.
enum class Values { kTies, kModerate, kWide };

Matrix random_basis(std::mt19937& rng, Index m, Values values) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> magnitude(0.5, 4.0);
  std::uniform_real_distribution<double> decade(-4.0, 2.0);
  const double ties[] = {1.0, -1.0, 2.0, -2.0, 0.5};
  std::uniform_int_distribution<int> tie_pick(0, 4);
  const auto draw = [&] {
    const double sign = unit(rng) < 0.5 ? -1.0 : 1.0;
    switch (values) {
      case Values::kTies: return ties[tie_pick(rng)];
      case Values::kModerate: return sign * magnitude(rng);
      case Values::kWide: break;
    }
    return sign * std::pow(10.0, decade(rng));
  };

  std::vector<Index> diagonal_row(static_cast<std::size_t>(m));
  for (Index i = 0; i < m; ++i) diagonal_row[static_cast<std::size_t>(i)] = i;
  std::shuffle(diagonal_row.begin(), diagonal_row.end(), rng);

  // pattern[j][i] != 0 means an entry; built densely (m is small) then emitted by column.
  std::vector<std::vector<double>> a(static_cast<std::size_t>(m),
                                     std::vector<double>(static_cast<std::size_t>(m), 0.0));
  const double density = 0.02 + 0.10 * unit(rng);
  for (Index j = 0; j < m; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    a[uj][static_cast<std::size_t>(diagonal_row[uj])] = draw();
    for (Index i = 0; i < m; ++i) {
      if (unit(rng) < density) a[uj][static_cast<std::size_t>(i)] = draw();
    }
  }
  // Planted dense columns: the #463 shape.
  const Index dense_columns = m < 4 ? 0 : static_cast<Index>(unit(rng) * 3.0);
  for (Index d = 0; d < dense_columns; ++d) {
    const auto uj = static_cast<std::size_t>(rng() % static_cast<unsigned>(m));
    const double fill = 0.5 + 0.5 * unit(rng);
    for (Index i = 0; i < m; ++i) {
      if (unit(rng) < fill) a[uj][static_cast<std::size_t>(i)] = draw();
    }
  }
  // Planted column singletons: keep only the diagonal entry.
  for (Index j = 0; j < m; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    if (unit(rng) >= 0.4) continue;
    const double keep = a[uj][static_cast<std::size_t>(diagonal_row[uj])];
    std::fill(a[uj].begin(), a[uj].end(), 0.0);
    a[uj][static_cast<std::size_t>(diagonal_row[uj])] = keep;
  }
  // Planted row singletons: keep only the entry in the column whose diagonal the row is.
  for (Index j = 0; j < m; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    if (unit(rng) >= 0.15) continue;
    const auto row = static_cast<std::size_t>(diagonal_row[uj]);
    for (std::size_t c = 0; c < a.size(); ++c) {
      if (c != uj) a[c][row] = 0.0;
    }
  }

  Matrix matrix(m);
  for (Index j = 0; j < m; ++j) {
    // Emit each column's rows in a shuffled order: the pivot search's tie-breaking follows
    // storage order, so a sorted pattern would exercise only one order.
    std::vector<Index> order;
    for (Index i = 0; i < m; ++i) {
      if (a[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] != 0.0)
        order.push_back(i);
    }
    std::shuffle(order.begin(), order.end(), rng);
    for (const Index i : order) {
      matrix.set(i, j, a[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)]);
    }
  }
  return matrix;
}

TEST(SparseLuSingletons, FuzzIdenticalToTheReferenceAndAgreesWithTheDenseOracle) {
  std::mt19937 rng(463);
  std::uniform_int_distribution<int> size(2, 90);
  // Two tallies: the well-scaled draws, where the forward error against DenseLu is bounded
  // exactly as in SparseLu.FuzzAgainstTheDenseOracle, and the six-decade draws, which exist
  // to make the threshold test bind and are ill-conditioned by construction - there two
  // backward-stable solvers may disagree in the forward error by the condition number, so
  // what is asserted is the bit-for-bit identity with the reference (inside check_matrix)
  // and the backward error.
  Worst scaled;
  Worst wide;
  int trials = 0;
  bool identical = true;
  for (int trial = 0; trial < 1500 && identical; ++trial) {
    const Values mode = trial % 3 == 0   ? Values::kTies
                        : trial % 3 == 1 ? Values::kModerate
                                         : Values::kWide;
    const Matrix matrix = random_basis(rng, static_cast<Index>(size(rng)), mode);
    ++trials;
    identical = check_matrix(matrix, rng, mode == Values::kWide ? wide : scaled, trial);
  }
  std::cout << "[ singletons ] " << trials << " matrices; well-scaled: " << scaled.singular
            << " singular, " << scaled.compared_with_oracle << " against DenseLu, worst "
            << "forward error " << scaled.oracle_ftran << " / " << scaled.oracle_btran
            << ", worst residual " << scaled.residual << " / " << scaled.transpose_residual
            << "; six decades: " << wide.singular << " singular, worst backward error "
            << wide.backward << ", forward error against DenseLu " << wide.oracle_ftran << " / "
            << wide.oracle_btran << '\n';
  EXPECT_TRUE(identical) << "the fast path and the reference elimination disagree";
  EXPECT_GT(scaled.compared_with_oracle, 700) << "too many singular draws to mean anything";
  EXPECT_GT(scaled.singular + wide.singular, 0)
      << "no singular draw: the rank-defect path went untested";
  // The bounds of SparseLu.FuzzAgainstTheDenseOracle, relative here because a planted dense
  // column puts entries of several units into the right-hand side.
  EXPECT_LT(scaled.oracle_ftran, 1e-8);
  EXPECT_LT(scaled.oracle_btran, 1e-8);
  EXPECT_LT(scaled.residual, 1e-9);
  EXPECT_LT(scaled.transpose_residual, 1e-9);
  // Backward error bounds set from the measurement, with a margin, not derived: 3.1e-13 on
  // the well-scaled draws and 1.1e-10 on the six-decade ones when this test was written.
  // The gap between them is element growth. Threshold pivoting with tau = 0.01 admits a
  // pivot a hundredth of its column's largest entry, so a step can grow the active
  // submatrix by up to 1 + 1/tau; over six decades of magnitudes it does, on the reference
  // elimination exactly as on the fast path (the two are bit-identical above).
  EXPECT_LT(scaled.backward, 1e-12);
  EXPECT_LT(wide.backward, 1e-9);
}

TEST(SparseLuSingletons, ASlackBasisWithOneDenseColumnIsTheSameFactorization) {
  // bdry2's shape (#463): every column a unit slack except one, which has an entry in every
  // row. The dense column's own row is the only one it is needed for, and every other pivot
  // is a column singleton whose row crosses the dense column.
  std::mt19937 rng(2026);
  std::uniform_real_distribution<double> value(-3.0, 3.0);
  for (const Index m : {Index{2}, Index{17}, Index{400}, Index{3000}}) {
    Matrix matrix(m);
    const Index dense_position = m / 2;
    for (Index j = 0; j < m; ++j) {
      if (j == dense_position) {
        for (Index i = 0; i < m; ++i) {
          const double v = value(rng);
          matrix.set(i, j, i == j ? 5.0 + std::fabs(v) : v);
        }
      } else {
        matrix.set(j, j, 1.0);
      }
    }
    Worst worst;
    EXPECT_TRUE(check_matrix(matrix, rng, worst, static_cast<int>(m))) << "m = " << m;
    EXPECT_EQ(worst.singular, 0) << "m = " << m;
    EXPECT_LT(worst.residual, 1e-12) << "m = " << m;
    EXPECT_LT(worst.transpose_residual, 1e-12) << "m = " << m;
  }
}

TEST(SparseLuSingletons, TwoDenseColumnsOverASparseNucleusAreTheSameFactorization) {
  // Linf_520c's shape (#463): two dense columns of different lengths among columns that are
  // not all singletons, so the elimination alternates between the fast path and the general
  // update and the row copy is invalidated part way through.
  std::mt19937 rng(520);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  const Index m = 600;
  Matrix matrix(m);
  for (Index j = 0; j < m; ++j) {
    if (j == 7 || j == 300) {
      const double fill = j == 7 ? 1.0 : 0.15;
      for (Index i = 0; i < m; ++i) {
        if (i == j) {
          matrix.set(i, j, 4.0);
        } else if (unit(rng) < fill) {
          matrix.set(i, j, value(rng));
        }
      }
      continue;
    }
    matrix.set(j, j, 2.0 + unit(rng));
    if (j % 5 == 0) {
      for (int k = 0; k < 3; ++k) {
        const auto i = static_cast<Index>(rng() % static_cast<unsigned>(m));
        if (i != j) matrix.set(i, j, value(rng));
      }
    }
  }
  // set() may repeat a row within a column above; rebuild without repeats, keeping the
  // first value, so the fast path is not simply switched off by the duplicate guard.
  Matrix clean(m);
  for (Index j = 0; j < m; ++j) {
    std::vector<char> seen(static_cast<std::size_t>(m), 0);
    const auto uj = static_cast<std::size_t>(j);
    for (std::size_t k = 0; k < matrix.rows[uj].size(); ++k) {
      const auto i = static_cast<std::size_t>(matrix.rows[uj][k]);
      if (seen[i] != 0) continue;
      seen[i] = 1;
      clean.set(matrix.rows[uj][k], j, matrix.values[uj][k]);
    }
  }
  Worst worst;
  EXPECT_TRUE(check_matrix(clean, rng, worst, 0));
  EXPECT_EQ(worst.singular, 0);
  EXPECT_LT(worst.residual, 1e-10);
  EXPECT_LT(worst.transpose_residual, 1e-10);
}

TEST(SparseLuSingletons, ARowRepeatedInAColumnFallsBackToTheReference) {
  // The general update merges a repeated row (the last value wins); the row copy cannot
  // reproduce that, so the input takes the reference path. It must still factorize to the
  // same thing the reference does.
  std::mt19937 rng(7);
  Matrix matrix(4);
  matrix.set(0, 0, 1.0);
  matrix.set(1, 0, 3.0);
  matrix.set(1, 0, 2.0);  // repeated
  matrix.set(1, 1, 1.0);
  matrix.set(2, 2, 1.0);
  matrix.set(3, 3, 1.0);
  matrix.set(0, 3, 4.0);
  Worst worst;
  EXPECT_TRUE(check_matrix(matrix, rng, worst, 0));
}

TEST(SparseLuSingletons, ASingularBasisReportsTheSameDefect) {
  // Two identical dense columns among slacks: rank one short, and the defect located by
  // the fast path must be the one the reference locates.
  std::mt19937 rng(11);
  const Index m = 50;
  Matrix matrix(m);
  for (Index j = 0; j < m; ++j) {
    if (j == 10 || j == 20) {
      for (Index i = 0; i < m; ++i) matrix.set(i, j, 1.0 + static_cast<double>(i % 3));
    } else {
      matrix.set(j, j, 1.0);
    }
  }
  Worst worst;
  EXPECT_TRUE(check_matrix(matrix, rng, worst, 0));
  EXPECT_EQ(worst.singular, 1);
}

}  // namespace
}  // namespace sankhya
