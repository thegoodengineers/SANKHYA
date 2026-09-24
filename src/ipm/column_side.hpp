// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's normal equations solved on the n x n side (#469).
//
// References:
//   Wright, "Primal-Dual Interior-Point Methods", SIAM (1997), ch. 11 - the normal
//     equations and their reduction from the augmented system.
//   Woodbury, "Inverting modified matrices", Statistical Research Group memo 42, Princeton
//     (1950); Golub & Van Loan, "Matrix Computations", 4th ed. (2013), sec. 2.1.4 - the
//     identity used below.
//   Zanetti & Gondzio, an interior point method choosing between the normal equations and
//     the augmented system by the predicted symbolic factor size, arXiv:2508.04370 (2025) -
//     the choice by factor size rather than by m against n.
//   Hestenes & Stiefel (1952); Higham, "Accuracy and Stability of Numerical Algorithms",
//     2nd ed. (2002), sec. 7.1 - conjugate gradients and the backward error they stop on.
//
// THE IDENTITY. The Newton system reduces to M dy = r with M = A Theta A^T + D, D the
// diagonal of the logicals' theta plus the dual regularization, over the m rows. When the
// rows far outnumber the columns, the n x n matrix
//     N = Theta^-1 + A^T D^-1 A
// is the smaller one, and by the Woodbury identity
//     M^-1 = D^-1 - D^-1 A N^-1 A^T D^-1,
// so dy = D^-1 (r - A u) with N u = A^T D^-1 r. N is symmetric positive definite, with the
// pattern of A^T A, and is factored by the same sparse LDL^T.
//
// WHY THE CONJUGATE GRADIENTS. D is tiny on an equality row (its logical is fixed, so only
// the regularization 1e-10 is left) and on an inequality row whose slack has vanished, and
// the product form then divides a difference by it. So, as for the dense columns of #467,
// the Woodbury form is the PRECONDITIONER of conjugate gradients on M itself, applied as
// products with A and never formed: exact where D is well scaled, repaired where it is not.
#pragma once

#include <vector>

#include "la/ldl.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya::ipm {

/// What the last ColumnSide::solve() did.
struct ColumnSideReport {
  int iterations = 0;
  double backward_error = 0.0;  ///< ||r - M x||_inf over ||r||_inf plus the terms of M x
  bool converged = false;
};

class ColumnSide {
 public:
  /// A^T, with the columns `fixed` marks left out (their theta is 0: they are constants and
  /// have no row in N beyond an identity diagonal). Built once.
  void set_matrix(const SparseMatrix& a, const std::vector<bool>& fixed);

  /// The lower triangle of N = Theta^-1 + A^T D^-1 A, D = row_shift + delta, for these
  /// theta (over the n columns of A). Records theta and D for solve(). The pattern is the
  /// same for every theta, which is what lets the analysis be reused.
  [[nodiscard]] bool assemble(const std::vector<double>& theta,
                              const std::vector<double>& row_shift, double delta,
                              SparseMatrix* lower, const SparseLdl::ShouldStop& should_stop);

  /// Solve M dy = rhs in place for the m-side M of the last assemble(), with `ldl` holding
  /// the factors of that N.
  ColumnSideReport solve(const SparseLdl& ldl, double* rhs) const;

 private:
  void multiply_m(const std::vector<double>& v, std::vector<double>* out) const;
  [[nodiscard]] double terms_of_m(const std::vector<double>& v) const;
  void precondition(const SparseLdl& ldl, const std::vector<double>& r,
                    std::vector<double>* z) const;

  const SparseMatrix* a_ = nullptr;
  SparseMatrix transpose_;  ///< A^T without the fixed columns, n x m
  std::vector<bool> fixed_;
  std::vector<double> theta_;  ///< over the n columns, 0 on a fixed one
  std::vector<double> d_;      ///< over the m rows: row_shift + delta
  /// d_ floored at kDiagonalFloor of each row's A Theta A^T diagonal: what N and the
  /// preconditioner are built with (column_side.cpp).
  std::vector<double> preconditioner_d_;
};

}  // namespace sankhya::ipm
