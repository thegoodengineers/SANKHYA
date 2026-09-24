// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dense columns in the interior point's normal equations (#467).
//
// References:
//   Andersen, Gondzio, Meszaros & Xu, "Implementation of interior point methods for large
//     scale linear programming", in Terlaky (ed.), Interior Point Methods of Mathematical
//     Programming, Kluwer (1996), sec. 5.2 - splitting off the dense columns, the
//     Sherman-Morrison-Woodbury correction, and why it needs a refinement to be stable.
//   Wright, "Primal-Dual Interior-Point Methods", SIAM (1997), ch. 11 - the same.
//   Hestenes & Stiefel, "Methods of conjugate gradients for solving linear systems", J. Res.
//     Nat. Bur. Standards 49 (1952); Golub & Van Loan, "Matrix Computations", 4th ed. (2013),
//     sec. 11.5 - preconditioned conjugate gradients.
//
// THE PROBLEM. M = A Theta A^T + diag(shift) + delta I. One column of A with c entries puts
// a c x c dense block into M: on bdry2 a column of 126,002 entries makes M about 7.9e9
// nonzeros, on Linf_520c two columns make it 4.7e8, where without them M holds 2.6e6 and
// 1.1e6. The columns are few; the fill is theirs.
//
// THE REMEDY. Write A = [A_s, A_d] with A_d the k dense columns, V = A_d Theta_d^(1/2), and
// M_s = A_s Theta_s A_s^T + diag(shift) + delta I, so M = M_s + V V^T. M_s is what the
// sparse LDL^T factorizes, and by Sherman-Morrison-Woodbury
//     M^-1 = M_s^-1 - M_s^-1 V (I + V^T M_s^-1 V)^-1 V^T M_s^-1,
// where the k x k Schur complement S = I + V^T M_s^-1 V is dense, symmetric positive
// definite with every eigenvalue at least 1, and costs k solves with M_s to build. The
// scaled form (I + ...) rather than Theta_d^-1 + A_d^T M_s^-1 A_d keeps the 1/theta of a
// column at its bound out of the arithmetic.
//
// WHY THE CONJUGATE GRADIENTS. The product form is unstable when M_s is nearly singular -
// a row met only by dense columns has nothing but delta on its diagonal in M_s - and the
// Woodbury answer then carries a large error in exactly the directions the dense columns
// span. So the Woodbury inverse is used as the PRECONDITIONER of conjugate gradients on
// M itself, applied as matrix-vector products with A and never formed: where the formula
// is exact CG converges in one or two steps, and where it is not CG repairs it.
#pragma once

#include <vector>

#include "la/ldl.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya::ipm {

/// The columns of `a` with more than `factor * sqrt(rows)` entries, among those `eligible`
/// marks (a fixed column contributes nothing to M and is never dense), densest first, at
/// most `cap` of them. Empty when there are none or factor <= 0.
[[nodiscard]] std::vector<Index> find_dense_columns(const SparseMatrix& a,
                                                    const std::vector<char>& eligible,
                                                    double factor, Index cap);

/// What the last solve() did.
struct PcgReport {
  enum class Preconditioner { kWoodbury, kSparseFactor };
  int iterations = 0;
  /// The normwise backward error at the end: ||r - M x||_inf over ||r||_inf plus the size
  /// of the terms of M x (see kPcgRelativeResidual in dense_columns.cpp).
  double relative_residual = 0.0;
  bool converged = false;
  bool broke_down = false;  ///< a step met non-positive curvature and the iteration stopped
  Preconditioner preconditioner = Preconditioner::kWoodbury;
};

class DenseColumnCorrection {
 public:
  /// The dense columns, fixed for the solve; `columns` from find_dense_columns().
  void set_columns(const SparseMatrix& a, std::vector<Index> columns);
  [[nodiscard]] bool active() const noexcept { return !columns_.empty(); }
  [[nodiscard]] const std::vector<Index>& columns() const noexcept { return columns_; }
  /// 1 on the dense columns, 0 elsewhere, over the columns of a.
  [[nodiscard]] const std::vector<char>& mask() const noexcept { return mask_; }

  /// Theta with the dense columns zeroed: what the sparse part M_s is assembled from.
  void sparse_theta(const std::vector<double>& theta, std::vector<double>* out) const;

  /// The row shift to assemble M_s with: row_shift, plus - on a row whose diagonal in M_s
  /// is below kSupportRatio of what the dense columns put there - the dense columns'
  /// diagonal. Such a row is held almost only by dense columns (in the extreme, an equality
  /// row met by nothing else has delta alone), M_s is nearly singular on it, and the
  /// Woodbury product form built on it loses every digit. With the dense diagonal standing
  /// in, the factor is of M_s + B for a diagonal B on those rows only: the preconditioner is
  /// then the exact inverse of M + B rather than of M, and conjugate gradients, which run on
  /// M itself, remove B's effect. Returns the number of rows so supported.
  Index preconditioner_shift(const SparseMatrix& a, const std::vector<double>& theta,
                             const std::vector<double>& row_shift, double delta,
                             std::vector<double>* shift) const;

  /// Build the Schur complement for this theta from `ldl`, which must hold the factors of
  /// M_s assembled from sparse_theta(theta) and preconditioner_shift(...) with this
  /// row_shift and delta; row_shift and delta here are the TRUE ones, which the conjugate
  /// gradients use to multiply by M. When the Schur complement is not numerically positive
  /// definite, solve() uses the sparse factor alone as its preconditioner; the return value
  /// is kept for a failure that leaves no preconditioner at all, which there is none of yet.
  [[nodiscard]] bool prepare(const SparseLdl& ldl, const SparseMatrix& a,
                             const std::vector<double>& theta,
                             const std::vector<double>& row_shift, double delta);

  /// Solve M x = rhs in place, M = A Theta A^T + diag(row_shift) + delta I with the theta
  /// given to prepare(), by conjugate gradients preconditioned with the Woodbury inverse,
  /// and with the sparse factor alone when that does not converge. When neither converges
  /// (`converged` false in the report) rhs holds the better of the two, which the caller
  /// must NOT use as a Newton direction: the interior point raises its regularization.
  PcgReport solve(double* rhs) const;
  /// Whether the last prepare() could factor the Schur complement.
  [[nodiscard]] bool woodbury_available() const noexcept { return woodbury_; }

  /// The Woodbury inverse alone, without the refinement (tests compare the two).
  void apply_preconditioner(const std::vector<double>& r, std::vector<double>* out) const;

 private:
  void multiply_full(const std::vector<double>& v, std::vector<double>* out) const;
  [[nodiscard]] double terms_of_product(const std::vector<double>& v) const;
  /// `patience`: steps before the stagnation test may stop the iteration. The sparse-factor
  /// pass has k eigenvalues away from one and needs about k + 1 steps whatever its residual
  /// does on the way, so it is not judged stalled before them.
  PcgReport conjugate_gradients(const std::vector<double>& b, bool woodbury, int max_iterations,
                                int patience, std::vector<double>* x) const;
  bool woodbury_ = false;

  std::vector<Index> columns_;
  std::vector<char> mask_;
  const SparseLdl* ldl_ = nullptr;
  const SparseMatrix* a_ = nullptr;
  std::vector<double> theta_;
  std::vector<double> row_shift_;
  double delta_ = 0.0;
  std::vector<double> sqrt_theta_d_;  ///< Theta_d^(1/2) over the k dense columns
  std::vector<double> schur_;         ///< Cholesky factor of S, k x k column-major (lower)
};

}  // namespace sankhya::ipm
