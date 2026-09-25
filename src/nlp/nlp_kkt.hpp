// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the primal-dual system of the NLP interior point, and its inertia (NLP stage 2).
//
// THE SYSTEM. Each iteration solves (Wachter and Biegler, Math. Programming 106(1), 2006,
// eq. 13, with the inertia correction of eq. 26)
//
//     [ W + Sigma + delta_w I      J^T     ] [ dw ]      [ grad phi_mu + J^T lambda ]
//     [          J              -delta_c I ] [ dl ]  = - [            c             ]
//
// W the Hessian of the Lagrangian, Sigma the barrier diagonal, J the constraint Jacobian.
//
// THE INERTIA, CERTIFIED. The step is a descent direction for the barrier problem only when
// this matrix has n positive and m negative eigenvalues and no zero one (sec. 3.1). The
// paper reads the inertia off a Bunch-Kaufman factorization; this project's LDL^T
// (src/la/ldl.cpp) has no pivoting. It has a QUASIDEFINITE mode instead (Vanderbei, SIAM J.
// Optim. 5(1), 1995), which prescribes every pivot's sign - + for the n primal rows, - for
// the m dual rows - and counts the pivots that came out with the wrong sign or smaller than
// kNlpPivotFloor. When that count is ZERO the factors are exact, D has exactly n positive
// and m negative entries, and by Sylvester's law of inertia so does the matrix: the inertia
// is certified. When it is not zero the inertia is unknown - it may even be right, if the
// ordering met a negative curvature direction first - and the caller treats it as wrong and
// raises delta_w (Algorithm IC). That is CONSERVATIVE: it can regularize a matrix whose
// inertia was already correct, which costs iterations, never an uncertified step.
//
// THE REGULARIZATION, UNDONE. The (2,2) block is factorized with at least
// -kNlpDualRegularization on its diagonal so that a quasidefinite factorization exists under
// the fill-reducing ordering (a zero dual pivot met before its column's Schur update would
// otherwise be counted as wrong). The step is then refined against the matrix WITHOUT that
// term (iterative refinement), so the direction solves the system the paper states.
#pragma once

#include <vector>

#include "la/ldl.hpp"
#include "nlp/barrier_nlp.hpp"

namespace sankhya::nlp {

class NlpKkt {
 public:
  /// Build the pattern of the system for `p` and run the symbolic analysis once. False when
  /// the analysis could not be completed (deadline, or a factor too large to hold).
  bool analyze(const BarrierNlp& p, const SparseLdl::ShouldStop& should_stop);

  /// Assemble with W (in p's Hessian layout), J (in p's Jacobian layout), the diagonal
  /// `sigma` (n entries) and delta_w, delta_c; factorize. Returns true when the factorization
  /// is exact, so the inertia (n, m, 0) is certified; false otherwise (or on a deadline -
  /// see stopped_early()).
  ///
  /// A column marked in `fixed` (equal bounds: no barrier term, no step) has its row and
  /// column replaced by the identity's, so its step is exactly zero whatever the rest does.
  bool factorize(const Vec& hessian, const Vec& jacobian, const Vec& sigma,
                 const std::vector<char>& fixed, double delta_w, double delta_c,
                 const SparseLdl::ShouldStop& should_stop);

  /// Solve the system last factorized, WITHOUT the dual regularization, in place: `rhs` has
  /// n + m entries and becomes (dw, dlambda). Returns the final relative residual.
  double solve(Vec* rhs) const;

  [[nodiscard]] bool stopped_early() const noexcept { return ldl_.stopped_early(); }
  [[nodiscard]] Index wrong_pivots() const noexcept { return ldl_.regularized_pivots(); }

 private:
  /// y = K x for the matrix as stated (dual regularization excluded).
  void multiply(const Vec& x, Vec* y) const;

  Index n_ = 0;
  Index m_ = 0;
  std::vector<Index> starts_, rows_;
  std::vector<Index> hess_slot_;  ///< p's Hessian slot -> K slot
  std::vector<Index> jac_slot_;   ///< p's Jacobian slot -> K slot
  std::vector<Index> diag_slot_;  ///< the K slot of (j, j), all n + m of them
  std::vector<signed char> signs_;
  Vec values_;
  double delta_c_ = 0.0;  ///< as stated, before the regularization floor
  SparseLdl ldl_;
  SparseMatrix matrix_;
};

}  // namespace sankhya::nlp
