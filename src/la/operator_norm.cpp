// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a certified bracket on ||A||_2 (#482). See operator_norm.hpp for the references
// ([HJ13], [GV13], [H02], [LPY25]); written from those texts.
//
// WHY EACH SIDE IS A BOUND.
//
// Lower. For any v != 0, ||A v|| / ||v|| <= max_{||u|| = 1} ||A u|| = ||A||_2 by the
// definition of the induced norm. Power iteration on A^T A only makes v a good candidate; the
// inequality holds for every iterate, so the largest value seen is kept.
//
// Upper, three ways, and the smallest is used:
//   1. ||A||_2 <= ||A||_F, since ||A||_2^2 = sigma_1^2 <= sum_i sigma_i^2 = ||A||_F^2
//      ([GV13] 2.3.2, [HJ13] 5.6).
//   2. ||A||_2^2 <= ||A||_1 ||A||_inf ([GV13] 2.3.2, Corollary 2.3.2): ||A||_2^2 = rho(A^T A)
//      <= ||A^T A||_1 <= ||A^T||_1 ||A||_1 = ||A||_inf ||A||_1, a spectral radius being at
//      most any induced norm.
//   3. Collatz-Wielandt on C = |A|^T |A| (entrywise absolute values). Entry by entry
//      |A^T A| <= |A|^T |A| = C (the triangle inequality on each inner product), so by
//      [HJ13] Theorem 8.1.18 rho(A^T A) <= rho(|A^T A|) <= rho(C). C is nonnegative, so for
//      ANY vector w with every entry positive, rho(C) <= max_j (C w)_j / w_j ([HJ13]
//      Corollary 8.1.29). Hence ||A||_2^2 = rho(A^T A) <= max_j (C w)_j / w_j for every
//      positive w. The w used are the power iterates of C from the all-ones vector, each
//      floored at the smallest normal double so none is zero; the bound holds for each of
//      them, so the smallest is kept. From w = 1 the ratio is max_j sum_i |a_ij| r_i with
//      r_i the absolute row sums, already at most ||A||_1 ||A||_inf; as w approaches the
//      Perron vector of C it falls towards || |A| ||_2^2.
//
// WHAT IS NOT USED, AND WHY. A Ritz value theta = v^T A^T A v of the power iteration with
// residual r = A^T A v - theta v certifies only that SOME eigenvalue of A^T A lies within
// ||r|| of theta (Bauer-Fike for a symmetric matrix, [GV13] 8.1.2); it says nothing about
// whether a LARGER eigenvalue exists that the start vector barely touched. theta + ||r|| is
// therefore not an upper bound on ||A||_2^2, and Lanczos error bounds have the same gap
// unless the start vector's component along the top singular vector is known, which makes
// them probabilistic (Kuczynski & Wozniakowski, SIAM J. Matrix Anal. Appl. 13(4), 1992).
// The power estimate is used here only as the certified LOWER side, so that the looseness of
// the upper side, upper / lower >= upper / ||A||_2, is itself a computed number.
//
// HOW TIGHT. 3 is exact for a matrix whose signs can be flipped to all nonnegative by row
// and column sign changes (bipartite sign pattern, network incidence matrices among them),
// since then || |A| ||_2 = ||A||_2. It is loosest on dense matrices of random sign, where
// || |A| ||_2 is of order n and ||A||_2 of order sqrt(n); tests/unit/test_operator_norm.cpp
// reports the ratio on both kinds. After the Pock-Chambolle alpha = 1 pass of build_scaling()
// the scaled matrix has || |A| ||_2 <= 1 (Pock & Chambolle, ICCV 2011, Lemma 2, whose proof
// uses only absolute values), so 3 falls towards a value of at most 1 there.
//
// ROUNDING. The three quantities are sums of nonnegative products, each of at most nnz + 2
// terms counting the two levels of the |A|^T |A| w product; by [H02] section 4.2 the computed
// value is within relative gamma_(nnz + 4) of the exact one in any summation order, and one
// division and one square root add a unit roundoff each. The squared bound is multiplied by
// 1 + 2 (nnz + 8) eps, eps = 2u, which exceeds that allowance, and its square root by
// 1 + 2 eps. The stored entries of A are exact as given, and this is a bound on exactly the
// matrix the iteration multiplies by.

#include "operator_norm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace sankhya {

OperatorNormBound bound_spectral_norm(const SparseMatrix& matrix, int power_iterations,
                                      int absolute_iterations, unsigned seed) {
  OperatorNormBound bound;
  const Index rows = matrix.num_rows();
  const Index cols = matrix.num_cols();
  if (rows == 0 || cols == 0 || matrix.num_nonzeros() == 0) return bound;
  const auto m = static_cast<std::size_t>(rows);
  const auto n = static_cast<std::size_t>(cols);

  // ---- Frobenius and Holder, one pass over the entries ------------------------------------
  std::vector<double> row_abs(m, 0.0);
  double frobenius_sq = 0.0;
  double max_column_abs = 0.0;
  for (Index j = 0; j < cols; ++j) {
    const ColumnView column = matrix.column(j);
    double column_abs = 0.0;
    for (Index k = 0; k < column.size; ++k) {
      const double a = std::fabs(column.values[k]);
      column_abs += a;
      frobenius_sq += a * a;
      row_abs[static_cast<std::size_t>(column.rows[k])] += a;
    }
    max_column_abs = std::max(max_column_abs, column_abs);
  }
  const double max_row_abs = *std::max_element(row_abs.begin(), row_abs.end());
  if (!(frobenius_sq > 0.0)) return bound;  // every stored entry is zero: ||A||_2 = 0
  const double holder_sq = max_column_abs * max_row_abs;

  // ---- Collatz-Wielandt on C = |A|^T |A| from w = 1 ---------------------------------------
  std::vector<double> w(n, 1.0);
  std::vector<double> z(m, 0.0);
  std::vector<double> cw(n, 0.0);
  double collatz_sq = std::numeric_limits<double>::infinity();
  const double floor_value = std::numeric_limits<double>::min();
  for (int step = 0; step < std::max(1, absolute_iterations); ++step) {
    std::fill(z.begin(), z.end(), 0.0);
    for (Index j = 0; j < cols; ++j) {  // z = |A| w
      const ColumnView column = matrix.column(j);
      const double wj = w[static_cast<std::size_t>(j)];
      for (Index k = 0; k < column.size; ++k) {
        z[static_cast<std::size_t>(column.rows[k])] += std::fabs(column.values[k]) * wj;
      }
    }
    double largest = 0.0;
    double ratio = 0.0;
    for (Index j = 0; j < cols; ++j) {  // cw = |A|^T z, and max_j cw_j / w_j
      const ColumnView column = matrix.column(j);
      double sum = 0.0;
      for (Index k = 0; k < column.size; ++k) {
        sum += std::fabs(column.values[k]) * z[static_cast<std::size_t>(column.rows[k])];
      }
      const auto u = static_cast<std::size_t>(j);
      cw[u] = sum;
      ratio = std::max(ratio, sum / w[u]);  // w[u] >= floor_value > 0 always
      largest = std::max(largest, sum);
    }
    if (std::isfinite(ratio)) collatz_sq = std::min(collatz_sq, ratio);
    if (!(largest > 0.0) || !std::isfinite(largest)) break;
    for (std::size_t j = 0; j < n; ++j) w[j] = std::max(cw[j] / largest, floor_value);
  }

  // ---- The lower side: power iteration on A^T A --------------------------------------------
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> spread(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& value : v) value = spread(rng);
  std::vector<double> av(m, 0.0);
  for (int step = 0; step < std::max(1, power_iterations); ++step) {
    double length = 0.0;
    for (const double value : v) length += value * value;
    length = std::sqrt(length);
    if (!(length > 0.0) || !std::isfinite(length)) break;
    for (double& value : v) value /= length;
    matrix.multiply(v.data(), av.data());
    double image = 0.0;
    for (const double value : av) image += value * value;
    bound.lower = std::max(bound.lower, std::sqrt(image));  // ||A v|| with ||v|| = 1
    matrix.transpose_multiply(av.data(), v.data());
  }

  // ---- The upper side, with the rounding allowance ----------------------------------------
  const double eps = std::numeric_limits<double>::epsilon();
  const double nnz = static_cast<double>(matrix.num_nonzeros());
  const double inflate_sq = 1.0 + 2.0 * (nnz + 8.0) * eps;
  const double inflate = 1.0 + 2.0 * eps;
  bound.frobenius = std::sqrt(frobenius_sq * inflate_sq) * inflate;
  bound.holder = std::sqrt(holder_sq * inflate_sq) * inflate;
  bound.collatz_wielandt = std::isfinite(collatz_sq)
                               ? std::sqrt(collatz_sq * inflate_sq) * inflate
                               : std::numeric_limits<double>::infinity();
  bound.upper = std::min({bound.frobenius, bound.holder, bound.collatz_wielandt});
  return bound;
}

}  // namespace sankhya
