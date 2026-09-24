// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the proximal primal-dual regularized Newton system of the LP interior point (#473).
//
// References:
//   Altman & Gondzio, "Regularized symmetric indefinite systems in interior point methods for
//     linear and quadratic optimization", Optim. Methods Softw. 11 (1999) - primal and dual
//     regularization of the augmented system, and why it can be driven down as mu falls.
//   Friedlander & Orban, "A primal-dual regularized interior-point method for convex
//     quadratic programs", Math. Program. Comput. 4 (2012) - the regularization as proximal
//     terms centred at the current iterate: the matrix changes, the right-hand side does not.
//   Vanderbei, "Symmetric quasidefinite matrices", SIAM J. Optim. 5(1) (1995) - a
//     quasi-definite matrix has an LDL^T with D of known signs under every symmetric
//     permutation, so the AMD ordering needs no numerical pivoting.
//   Wilkinson, "Rounding Errors in Algebraic Processes" (1963); Higham, "Accuracy and
//     Stability of Numerical Algorithms", 2nd ed., SIAM (2002), ch. 12 - iterative refinement.
//
// THE SYSTEM. The interior point (ipm.cpp) works on [A | -I] [x; s] = 0 with every variable
// bounded. Its Newton system over all n + m variables and the m row multipliers is
//
//     [ -Theta^-1   Abar^T ] [dx]   [ g   ]          Theta^-1 = Z_l S_l^-1 + Z_u S_u^-1
//     [  Abar       0      ] [dy] = [ r_b ]
//
// (a fixed variable is pinned, dx = 0). The default path adds a fixed rho = 1e-8 to
// Theta^-1 and delta = 1e-10 to the normal equations Abar Theta Abar^T, raising delta only
// once a direction has already come back NaN. This path instead factors the regularized
// augmented system itself,
//
//     K_reg = [ -(Theta^-1 + rho I)   Abar^T  ]
//             [  Abar                 delta I ]
//
// which is quasi-definite for any rho, delta > 0 - however singular Theta^-1 becomes and
// whether or not a column is free - with rho and delta chosen by the caller from mu. The
// logicals' block of Theta is diagonal and is eliminated first, so what is factored is
//
//     [ -(Theta_x^-1 + rho I)   A^T                   ]     theta_s = (Theta_s^-1 + rho)^-1
//     [  A                      diag(theta_s) + delta ]
//
// of dimension n + m. It is factored by the project's own signed LDL^T,
// SparseLdl::factorize_quasidefinite - the kernel the QP interior point (#490) uses; nothing
// of it is duplicated here.
//
// THE UNREGULARIZED SYSTEM IS THE ONE SOLVED. The factors of K_reg are used as a
// preconditioner for the Newton system with rho = delta = 0, by iterative refinement: the
// residual is measured against the unregularized matrix and corrected through K_reg^-1. A
// correction is kept only while that residual falls, so the direction handed back never has
// a larger unregularized residual than the plain regularized solve. That is a statement
// about the residual, not about the distance to the exact Newton direction, and the residual
// can stall above what the method needs: the solve reports it RELATIVE to
// max(1, |g|, |r_b|), and the interior point counts every solve left above
// tol::kIpmProximalRefinementTarget, reports the count, and shrinks rho for the next
// factorization so that K_reg moves toward the unregularized matrix.
#pragma once

#include <vector>

#include "la/ldl.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya::ipm {

class ProximalSystem {
 public:
  /// `a` is the m x n constraint matrix, `fixed` flags each of the n + m variables that is
  /// pinned (a fixed column has no entries in the pattern). The pattern is fixed here.
  ProximalSystem(const SparseMatrix& a, const std::vector<bool>& fixed);

  /// The values of K_reg for the given Theta^-1 (over all n + m variables, WITHOUT any
  /// regularization; 0 for a free variable) and rho, delta > 0.
  void assemble(const std::vector<double>& theta_inverse, double rho, double delta);

  /// The regularization for the next iteration: max(floor, min(previous, share * mu)).
  /// When the last refinement missed its target, `previous` is first shrunk by
  /// tol::kIpmProximalRefinementShrink (never below the floor).
  [[nodiscard]] static double next_regularization(double previous, double mu, double floor,
                                                  bool refinement_missed = false);

  /// The rho floor after `raises` recoveries from a non-finite direction (#209):
  /// tol::kIpmProximalFloor * tol::kIpmProximalRecoveryRaise^raises, at most
  /// tol::kIpmProximalRecoveryCap.
  [[nodiscard]] static double recovery_floor(Count raises);

  /// Assemble with rho = delta = *reg and factor into `ldl`, which must already be analyzed
  /// on this pattern; while the factorization had to lift a pivot, raise *reg and factor
  /// again, at most tol::kIpmProximalAttempts factorizations in all (counted in
  /// *factorizations). A pivot still lifted after the last attempt is left to the caller,
  /// which sees it in ldl.regularized_pivots(). False only when the factorization itself
  /// failed (a deadline, or a pattern that does not fit).
  [[nodiscard]] bool factorize(SparseLdl& ldl, const std::vector<double>& theta_inverse,
                               double* reg, const SparseLdl::ShouldStop& should_stop,
                               int* factorizations);

  [[nodiscard]] const SparseMatrix& matrix() const noexcept { return k_; }
  /// -1 on the n structural positions, +1 on the m row positions.
  [[nodiscard]] const std::vector<signed char>& signs() const noexcept { return signs_; }

  struct Refinement {
    int steps = 0;                ///< corrections kept
    double first_residual = 0.0;  ///< unregularized residual of the plain K_reg solve
    double final_residual = 0.0;  ///< the same after the kept corrections
    double scale = 1.0;           ///< max(1, |g|, |r_b|), infinity norms
    /// final_residual / scale: what the target tol::kIpmProximalRefinementTarget is on.
    [[nodiscard]] double relative_residual() const noexcept { return final_residual / scale; }
  };

  /// Solve the UNREGULARIZED Newton system for dx (n + m) and dy (m), given g (n + m) and
  /// r_b (m), with the factors of the last assemble() in `ldl`, and up to `max_steps`
  /// refinement corrections. Entries of g at fixed variables are ignored, dx is 0 there.
  Refinement solve(const SparseLdl& ldl, const std::vector<double>& g,
                   const std::vector<double>& r_b, int max_steps, std::vector<double>* dx,
                   std::vector<double>* dy) const;

 private:
  /// (dx, dy) = K_reg^-1 (p, q), in the full n + m + m space.
  void apply_inverse(const SparseLdl& ldl, const std::vector<double>& p,
                     const std::vector<double>& q, std::vector<double>* dx,
                     std::vector<double>* dy) const;
  /// (p, q) = (g, r_b) - K_0 (dx, dy); returns the infinity norm of the residual.
  double residual(const std::vector<double>& g, const std::vector<double>& r_b,
                  const std::vector<double>& dx, const std::vector<double>& dy,
                  std::vector<double>* p, std::vector<double>* q) const;

  const SparseMatrix& a_;
  const std::vector<bool>& fixed_;
  Index n_ = 0;
  Index m_ = 0;
  std::vector<Index> starts_;
  std::vector<Index> rows_;
  std::vector<signed char> signs_;
  SparseMatrix k_;
  std::vector<double> theta_inverse_;  ///< unregularized, as last assembled
  std::vector<double> theta_s_;        ///< regularized logical weights, as last assembled
  mutable std::vector<double> work_;
};

}  // namespace sankhya::ipm
