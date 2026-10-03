// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the convexity test against rounding, in both directions (#835).
//
// The whole-QPLIB run refused three instances QPLIB lists as convex: QPLIB_10056 (rank 31 of
// 175, the other 144 eigenvalues rounding of size 1e-14), QPLIB_10069 (positive definite,
// condition 1e11) and QPLIB_8515 (a 7.8e-9 entry beside a zero diagonal, which costs an
// eigenvalue of -3e-17). Each is reproduced small below and must now be ACCEPTED, by the
// sparse test and the dense reference alike.
//
// The other direction matters more. A tolerance wide enough to swallow rounding must not
// swallow curvature: an eigenvalue of -1e-3 relative is a non-convex objective, and it must
// still be refused, with a direction x that this file checks against Q itself, sharing no
// code with the factorization, to have x^T Q x < 0.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"

#include "qp/convexity.hpp"

namespace sankhya::qp {
namespace {

using Dense = std::vector<std::vector<double>>;

/// A model with nothing but a Hessian, given as a dense symmetric matrix.
Model hessian_only(const Dense& q) {
  const auto n = static_cast<Index>(q.size());
  Model model;
  model.col_cost.assign(q.size(), 0.0);
  model.col_lower.assign(q.size(), -10.0);
  model.col_upper.assign(q.size(), 10.0);
  model.col_type.assign(q.size(), VarType::kContinuous);
  model.matrix.reset(0, n);
  model.matrix.finalize();
  model.hessian.reset(n, n);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j <= i; ++j) {
      const double v = q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.hessian.add_entry(i, j, v);
    }
  }
  model.hessian.finalize();
  return model;
}

/// x^T Q x from the model's stored lower triangle, written here and not shared with src/.
double quadratic_form(const Model& model, const std::vector<double>& x) {
  double total = 0.0;
  const double sense = model.sense_multiplier();
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index i = column.rows[k];
      const double term = sense * column.values[k] * x[static_cast<std::size_t>(i)] *
                          x[static_cast<std::size_t>(j)];
      total += (i == j) ? term : 2.0 * term;
    }
  }
  return total;
}

/// A random orthogonal n x n matrix: modified Gram-Schmidt on Gaussian columns.
Dense random_orthogonal(std::size_t n, std::mt19937& rng) {
  std::normal_distribution<double> gauss(0.0, 1.0);
  Dense v(n, std::vector<double>(n));
  for (auto& column : v) {
    for (double& x : column) x = gauss(rng);
  }
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t p = 0; p < c; ++p) {
      double dot = 0.0;
      for (std::size_t r = 0; r < n; ++r) dot += v[c][r] * v[p][r];
      for (std::size_t r = 0; r < n; ++r) v[c][r] -= dot * v[p][r];
    }
    double norm = 0.0;
    for (const double x : v[c]) norm += x * x;
    norm = std::sqrt(norm);
    for (double& x : v[c]) x /= norm;
  }
  return v;  // v[c] is the c-th column
}

/// Q = V diag(eigenvalues) V^T, in floating point: the rounding is part of the test.
Dense with_spectrum(const std::vector<double>& eigenvalues, std::mt19937& rng) {
  const std::size_t n = eigenvalues.size();
  const Dense v = random_orthogonal(n, rng);
  Dense q(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double sum = 0.0;
      for (std::size_t c = 0; c < n; ++c) sum += v[c][i] * eigenvalues[c] * v[c][j];
      q[i][j] = sum;
      q[j][i] = sum;
    }
  }
  return q;
}

void expect_convex(const Model& model, const char* what) {
  const ConvexityResult sparse = check_convexity(model);
  const ConvexityResult dense = check_convexity_dense(model);
  EXPECT_EQ(sparse.verdict, Convexity::kConvex) << what << "\n  sparse: " << sparse.detail;
  EXPECT_EQ(dense.verdict, Convexity::kConvex) << what << "\n  dense: " << dense.detail;
}

/// Refused, with a witness that Q itself confirms, whose curvature is no lower than the
/// smallest eigenvalue (a Rayleigh quotient cannot be) and at least `at_most` below zero.
void expect_refused(const Model& model, double smallest_eigenvalue, double at_most,
                    const char* what) {
  for (const ConvexityResult& result : {check_convexity(model), check_convexity_dense(model)}) {
    ASSERT_EQ(result.verdict, Convexity::kIndefinite) << what << "\n  " << result.detail;
    ASSERT_EQ(result.witness.size(), static_cast<std::size_t>(model.num_cols()));
    double norm_squared = 0.0;
    for (const double x : result.witness) norm_squared += x * x;
    const double form = quadratic_form(model, result.witness);
    EXPECT_LT(form, 0.0) << what << ": the witness does not curve downward";
    EXPECT_NEAR(form / norm_squared, result.witness_curvature,
                1e-9 * std::fabs(result.witness_curvature))
        << what;
    EXPECT_GE(result.witness_curvature, smallest_eigenvalue * (1.0 + 1e-9)) << what;
    EXPECT_LE(result.witness_curvature, at_most) << what << "\n  " << result.detail;
  }
}

// =========================================================================================
// Accepted: semidefinite matrices whose rounding the old rule read as curvature
// =========================================================================================

TEST(ConvexityTolerance, ARankDeficientSemidefiniteMatrixWithRoundingNoiseIsConvex) {
  // QPLIB_10056's failure: a rank-deficient semidefinite matrix, formed in floating point,
  // whose nonzero part is itself ill conditioned. Here Q = u u^T + w w^T with w = u + 3e-5 z,
  // scaled to a largest diagonal of 6 as QPLIB_10056's is: rank 2, one eigenvalue of about
  // 1e-9 beside the large one, the rest zero up to rounding. The second pivot is then tiny
  // but real, and the old rule, which took a pivot within its slack (6e-10) as exactly zero
  // and refused unless the entries beside it were within the same slack, refused about one
  // draw in five; 40 draws, none chosen, must all be accepted. Two all-zero columns ride
  // along, as three of QPLIB_10056's 175 are.
  std::mt19937 rng(835);
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (int draw = 0; draw < 40; ++draw) {
    constexpr std::size_t n = 8;
    std::vector<double> u(n);
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
      u[i] = gauss(rng);
      w[i] = u[i] + 3e-5 * gauss(rng);
    }
    u[2] = w[2] = 0.0;
    u[5] = w[5] = 0.0;
    Dense q(n, std::vector<double>(n, 0.0));
    double largest = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) q[i][j] = u[i] * u[j] + w[i] * w[j];
      largest = std::max(largest, q[i][i]);
    }
    for (auto& row : q) {
      for (double& x : row) x *= 6.0 / largest;
    }
    expect_convex(hessian_only(q), "rank 2, nearly parallel");
  }
}

TEST(ConvexityTolerance, APositiveDefiniteMatrixWithAPivotBelowTheOldSlackIsConvex) {
  // QPLIB_10069's failure, in two columns: positive definite (determinant 5e-11 - 1e-12 > 0,
  // smallest eigenvalue 4.9e-11), whose first pivot 5e-11 is below the old slack of 1e-10 and
  // whose off-diagonal 1e-6 is above it. The old rule zeroed the pivot and then read the
  // 1e-6 beside it as "a nonzero beside a zero pivot". A pivot of 5e-11 carries entries up
  // to sqrt(5e-11 * 1) = 7e-6 beside it and stays semidefinite.
  expect_convex(hessian_only({{5e-11, 1e-6}, {1e-6, 1.0}}), "[[5e-11, 1e-6], [1e-6, 1]]");

  // And at size: positive definite, eigenvalues from 1 down to 1e-11.
  std::mt19937 rng(10069);
  for (int draw = 0; draw < 10; ++draw) {
    constexpr std::size_t n = 30;
    std::vector<double> eigenvalues(n);
    for (std::size_t c = 0; c < n; ++c) {
      eigenvalues[c] =
          std::pow(10.0, -11.0 * static_cast<double>(c) / static_cast<double>(n - 1));
    }
    expect_convex(hessian_only(with_spectrum(eigenvalues, rng)), "condition 1e11");
  }
}

TEST(ConvexityTolerance, AZeroDiagonalBesideATinyOffDiagonalIsRounding) {
  // QPLIB_8515's pattern at 1/40 of its size, with its numbers as the converter writes them:
  // a chain of odd columns with diagonal 4 (2 at the ends) and -1 between neighbours, each
  // with an even partner whose diagonal is ZERO and whose one entry is 7.8125e-9 beside it.
  // In exact arithmetic that is indefinite, by about -(7.8125e-9)^2 / 4 = -1.5e-17: rounding
  // by any measure, and QPLIB lists the instance as convex. The old rule refused it because
  // 7.8125e-9 exceeded its slack, an absolute 4e-10, beside the zero pivot.
  constexpr std::size_t m = 200;
  constexpr double coupling = 7.8125e-9;
  const std::size_t n = 2 * m + 2;
  Dense q(n, std::vector<double>(n, 0.0));
  for (std::size_t k = 0; k < m; ++k) {
    const std::size_t odd = 2 * k;
    q[odd][odd] = (k == 0 || k + 1 == m) ? 2.0 : 4.0;
    q[odd + 1][odd] = q[odd][odd + 1] = coupling;
    if (k + 1 < m) q[odd + 2][odd] = q[odd][odd + 2] = -1.0;
  }
  q[n - 2][n - 2] = 4.0;
  q[n - 1][n - 2] = q[n - 2][n - 1] = coupling;
  expect_convex(hessian_only(q), "QPLIB_8515's pattern");

  // The same pattern with a coupling that is curvature rather than rounding: beside a zero
  // diagonal, an entry b costs an eigenvalue of about -b^2 / 4, here -2.5e-3.
  for (std::size_t k = 0; k < m; ++k) q[2 * k + 1][2 * k] = q[2 * k][2 * k + 1] = 0.1;
  const Model coupled = hessian_only(q);
  const ConvexityResult refused = check_convexity(coupled);
  ASSERT_EQ(refused.verdict, Convexity::kIndefinite) << refused.detail;
  EXPECT_LT(quadratic_form(coupled, refused.witness), 0.0);
}

// =========================================================================================
// Refused: curvature, however small relative to the matrix, above the tolerance
// =========================================================================================

TEST(ConvexityTolerance, ASlightlyIndefiniteMatrixIsRefusedWithItsWitness) {
  // One eigenvalue of -1e-3 against a largest of 1: non-convex, and no amount of rounding
  // explains it. The rest of the spectrum is spread down to 1e-6 so the factorization is not
  // trivially well conditioned.
  std::mt19937 rng(20261003);
  for (int draw = 0; draw < 10; ++draw) {
    constexpr std::size_t n = 30;
    std::vector<double> eigenvalues(n);
    for (std::size_t c = 0; c + 1 < n; ++c) {
      eigenvalues[c] =
          std::pow(10.0, -6.0 * static_cast<double>(c) / static_cast<double>(n - 2));
    }
    eigenvalues[n - 1] = -1e-3;
    expect_refused(hessian_only(with_spectrum(eigenvalues, rng)), -1e-3, -1e-10,
                   "eigenvalue -1e-3");
  }
}

TEST(ConvexityTolerance, TheToleranceIsRelativeToTheLargestDiagonal) {
  // The same spectra at three scales. An eigenvalue of -1e-13 relative is accepted at every
  // scale, -1e-13 * 1e6 = -1e-7 included, which an absolute tolerance would refuse; -1e-5
  // relative is refused at every scale, -1e-5 * 1e-3 = -1e-8 included.
  std::mt19937 rng(1010);
  constexpr std::size_t n = 12;
  for (const double scale : {1e-3, 1.0, 1e6}) {
    for (const double negative : {-1e-13, -1e-5}) {
      std::vector<double> eigenvalues(n);
      for (std::size_t c = 0; c + 1 < n; ++c) {
        eigenvalues[c] = scale * (1.0 + static_cast<double>(c));
      }
      eigenvalues[n - 1] = scale * negative;
      const Model model = hessian_only(with_spectrum(eigenvalues, rng));
      if (negative == -1e-5) {
        expect_refused(model, scale * negative, 0.0, "relative -1e-5");
      } else {
        expect_convex(model, "relative -1e-13");
      }
    }
  }
}

TEST(ConvexityTolerance, ABilinearSaddleIsRefusedWithItsExactEigenvalue) {
  // [[0, 1], [1, 0]]: the shift makes the first pivot 1e-12 and the second about -1e12, and
  // the factorization's own direction, (-1e12, 1), has a Rayleigh quotient of only -2e-12. The
  // two-column direction the refusal reports instead is the eigenvector, curvature exactly -1.
  const Model model = hessian_only({{0.0, 1.0}, {1.0, 0.0}});
  expect_refused(model, -1.0, -1.0 + 1e-12, "x0 * x1");
}

}  // namespace
}  // namespace sankhya::qp
