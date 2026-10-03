// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a certified bracket on the spectral norm ||A||_2 of a sparse matrix (#482).
//
// estimate_spectral_norm() (scaling.hpp) is power iteration on A^T A: every value it produces
// is a LOWER bound on ||A||_2 (the Rayleigh quotient of any unit vector is at most the largest
// eigenvalue), rounded up 1% by hand. That is fine as the starting point of the adaptive step
// rule, which corrects it, and not fine as a CONSTANT step 1 / ||A||_2, which nothing
// corrects: PDHG converges for tau sigma ||A||_2^2 < 1 (Chambolle & Pock 2011, Theorem 1) and
// a step from an underestimate breaks that. This returns a bracket instead:
//
//   lower <= ||A||_2 <= upper,
//
// both sides proved, not estimated (up to the rounding of the arithmetic that computes them,
// which `upper` absorbs; see the .cpp). The constant-step option of the LP PDHG
// (pdhg_constant_step) runs at a fixed share of 1 / upper.
//
// References
//   [HJ13]  Horn & Johnson, *Matrix Analysis*, 2nd ed., Cambridge 2013: Theorem 8.1.18
//           (|B| <= C entrywise implies rho(B) <= rho(|B|) <= rho(C)), Corollary 8.1.29
//           (Collatz-Wielandt: for C >= 0 and w > 0, rho(C) <= max_j (C w)_j / w_j),
//           and section 5.6 (||A||_2 <= ||A||_F, ||A||_2^2 <= ||A||_1 ||A||_inf).
//   [GV13]  Golub & Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins 2013,
//           section 2.3.2 (the same two norm inequalities) and 8.2.1 (power iteration).
//   [H02]   Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002,
//           section 4.2 (a sum of k nonnegative terms carries relative error at most
//           gamma_(k-1) = (k-1) u / (1 - (k-1) u) in any order).
//   [LPY25] Lu, Peng & Yang, *cuPDLPx: a further enhanced GPU-based first-order solver for
//           linear programming*, arXiv:2507.14051: the constant step of about 0.998 / ||A||_2
//           under the reflected Halpern iteration that this bound makes safe.
// Written from these texts; no solver's source was consulted.
#pragma once

#include "sankhya/sparse.hpp"

namespace sankhya {

/// A proved bracket on ||A||_2, with the pieces the upper side is the minimum of, so a test
/// or a log can say which one was tight.
struct OperatorNormBound {
  /// max over the power iterates v of ||A v|| / ||v||: a lower bound on ||A||_2.
  double lower = 0.0;
  /// min(frobenius, holder, collatz_wielandt), inflated by the rounding allowance: an upper
  /// bound on ||A||_2. Zero only for a matrix with no nonzero entry, whose norm is zero.
  double upper = 0.0;
  /// ||A||_F.
  double frobenius = 0.0;
  /// sqrt(||A||_1 ||A||_inf).
  double holder = 0.0;
  /// sqrt of the smallest Collatz-Wielandt ratio max_j (|A|^T |A| w)_j / w_j over the
  /// positive power iterates w of |A|^T |A|: an upper bound on || |A| ||_2 >= ||A||_2.
  double collatz_wielandt = 0.0;
};

/// The bracket, from `power_iterations` steps of power iteration on A^T A from a random start
/// drawn from `seed` (the lower side) and `absolute_iterations` steps on |A|^T |A| from the
/// all-ones vector (the Collatz-Wielandt part of the upper side). Each step is one product
/// with A and one with A^T. Deterministic for a given seed.
[[nodiscard]] OperatorNormBound bound_spectral_norm(const SparseMatrix& matrix,
                                                    int power_iterations,
                                                    int absolute_iterations, unsigned seed);

}  // namespace sankhya
