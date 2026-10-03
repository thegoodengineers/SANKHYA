// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convexity test for a quadratic objective.
//
// A convex QP has a global minimum that first-order and interior-point methods actually
// converge to. A NON-convex one has local minima, saddle points and possibly no finite
// infimum at all, and every method here would still produce a point, print it, and call it
// optimal. That is the failure mode ENGINEERING_RULES.md opens with, so a Hessian that is not
// provably positive semidefinite is REFUSED rather than solved to whatever the iteration lands
// on.
//
// The test is an LDL^T factorization of Q + shift * I, shift = 1e-12 * max(1, largest |Q_ii|),
// with symmetric pivoting on the DIAGONAL only, and every pivot required to be positive
// (#835). 1e-12 is the threshold QPLIB counts a negative eigenvalue by; convexity.cpp records
// the instances on either side of it. A positive semidefinite Q makes Q + shift * I positive
// definite with its smallest eigenvalue at least shift, which a Cholesky-type factorization
// completes on without pivoting as long as its rounding stays below that eigenvalue; so a
// semidefinite Q passes whether it is singular, rank deficient or merely ill conditioned. A
// pivot that is not positive exhibits a direction x with x^T Q x <= -shift * x^T x < 0, which
// is returned and checked against Q itself, in arithmetic accurate enough for its sign to be
// trusted, before the model is called non-convex. That is a decision procedure with a stated
// tolerance - an eigenvalue above -shift is accepted as rounding - rather than a heuristic,
// which matters: guessing "convex" on a non-convex model is precisely the wrong error to make.
//
// The previous rule factorized Q itself and took a pivot within a slack of zero as exactly
// zero, requiring the entries beside it to vanish too. In floating point that refused three
// QPLIB instances QPLIB lists as convex (QPLIB_10056, rank 31 of 175; QPLIB_10069, positive
// definite with condition 1e11; QPLIB_8515, an eigenvalue of -3e-17 from a 7.8e-9 entry beside
// a zero diagonal); src/la/ldl.cpp records how each one failed.
//
// References: Golub & Van Loan, "Matrix Computations" (4th ed.), section 4.1, for LDL^T and
// its relationship to definiteness; Higham, "Analysis of the Cholesky decomposition of a
// semi-definite matrix" (1990), and "Accuracy and Stability of Numerical Algorithms" (2nd
// ed., 2002), ch. 10, for the semidefinite case and the backward error of Cholesky; Rump,
// "Verification of positive definiteness", BIT 46 (2006), for deciding definiteness by a
// Cholesky factorization of a shifted matrix; Ogita, Rump & Oishi, "Accurate sum and dot
// product", SIAM J. Sci. Comput. 26 (2005), for evaluating the witness.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {
class ResourceLimits;
class SolveControl;
class Timer;
}  // namespace sankhya

namespace sankhya::qp {

/// What the convexity test concluded.
enum class Convexity {
  kConvex,      ///< Q is positive semidefinite; the QP has a global minimum
  kIndefinite,  ///< Q has a provably negative eigenvalue; the model is non-convex
  kUnverified,  ///< the test could not decide; treated as a refusal
};

struct ConvexityResult {
  Convexity verdict = Convexity::kUnverified;
  /// Human-readable detail naming the column and the pivot that decided it.
  std::string detail;
  /// For kIndefinite, the certificate: a direction x over the model's columns with
  /// x^T Q x < 0 in minimization sense, checked against Q itself by more than its rounding
  /// error. Empty otherwise.
  std::vector<double> witness;
  /// x^T Q x / x^T x along `witness`: an upper bound on Q's smallest eigenvalue.
  double witness_curvature = 0.0;
  /// The caller's `should_stop` ended the factorization before it decided anything. The
  /// verdict is then kUnverified, never kConvex: running out of time proves nothing about
  /// Q, so the engine stops with the time limit rather than either solving or refusing.
  bool stopped = false;
};

/// Decide whether `model.hessian` is positive semidefinite.
///
/// The Hessian is stored as the lower triangle of a symmetric matrix and the objective term
/// is 0.5 x^T Q x, so this tests Q itself rather than the stored triangle.
///
/// Runs on the SPARSE factorization at every size (#303). The dense version below needs an
/// n x n working set - 20 GB at 50,000 columns - which is not a test a solver for sparse
/// models can afford to run, and refusing every QP above a few thousand columns to avoid it
/// meant a large sparse convex QP could not be solved at all.
///
/// `should_stop` is the caller's deadline (#835), asked inside the ordering, the symbolic
/// analysis and between the columns of the factorization, as SparseLdl asks it for the
/// interior point. The test is an engine's costliest step before its first iteration when
/// the factor is large, and an MIQP runs it before every node QP, so without the deadline it
/// is work no time limit reaches. A stop is reported as `stopped`, the verdict kUnverified.
[[nodiscard]] ConvexityResult check_convexity(const Model& model,
                                              const std::function<bool()>& should_stop = {});

/// The same decision, computed densely.
///
/// THE REFERENCE, not the production path: O(n^2) memory and O(n^3) time, and it reports
/// kUnverified above a couple of thousand columns rather than allocating. It stays in the
/// tree because two implementations of one decision, written from the same definition and
/// compared on instances nobody chose, is how this project checks its numerics - the same
/// role `DenseLu` plays for the sparse LU. `tests/unit/test_convexity_sparse.cpp` runs them
/// against each other on random matrices.
[[nodiscard]] ConvexityResult check_convexity_dense(const Model& model);

/// The deadline a QP engine hands check_convexity(): its time limit on its own clock, or the
/// user's interrupt. `limits`, `timer` and `control` must outlive the returned predicate.
[[nodiscard]] std::function<bool()> convexity_deadline(const ResourceLimits& limits,
                                                       const Timer& timer,
                                                       const SolveControl* control);

/// What a QP engine returns when that deadline stopped the test (`convexity.stopped`): the
/// time limit, or the interrupt if that is what fired, with the test's detail. Neither a
/// refusal nor a solve, since nothing about Q was decided.
void stopped_before_convexity(const ResourceLimits& limits, const Timer& timer,
                              const SolveControl* control, const ConvexityResult& convexity,
                              Solution* solution);

}  // namespace sankhya::qp
