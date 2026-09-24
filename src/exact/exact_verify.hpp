// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact rational verification of a reported optimal LP basis (#521).
//
// Gleixner, Steffy and Wolter, "Iterative refinement for linear programming", INFORMS J.
// Computing 28(3), 2016; Gleixner and Steffy, "Linear programming using limited-precision
// oracles", Math. Programming 183, 2020. The paper's full pipeline repeatedly refines a
// candidate point through scaled correction problems solved in double precision. This is
// the narrower piece of it that stands on its own: given the BASIS the double-precision
// engine already reports at `optimal`, rebuild it in exact rational arithmetic (every model
// coefficient converted bit-for-bit via Rational::from_double, not rounded) and check that
// it is exactly primal and dual feasible - which, for a nondegenerate basis, is exactly
// optimality. No iterative correction loop: this checks the ONE basis in hand rather than
// searching for a better one, which is what makes it small enough to implement and verify
// correctly in the time this issue actually has, at the cost of not repairing a basis that
// double precision got structurally wrong. See exact_verify.cpp's module comment for what
// happens then (declines, does not report a false proof).
#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::exact {

enum class ExactVerdict {
  kVerified,  ///< the basis is exactly primal and dual feasible; an exact optimum
  kDeclined,  ///< not attempted or not completed for a stated, non-adversarial reason
  kFailed,    ///< attempted and the basis is NOT exactly feasible - the double answer was wrong
};

struct ExactResult {
  ExactVerdict verdict = ExactVerdict::kDeclined;
  std::string message;  ///< why declined/failed; empty on kVerified

  /// Populated only on kVerified. Decimal "numerator/denominator" strings (exact, not
  /// rounded) - kept as strings here so this header does not expose __int128 across the
  /// module boundary; exact_verify.cpp does the Rational arithmetic internally.
  std::string exact_objective;
  std::vector<std::string> exact_col_value;
};

/// `solution` must be the Solution a plain LP solve() just returned with status kOptimal and
/// a basis (col_status/row_status sized to the model). Declines outright on a quadratic
/// objective (Hessian nonzero - #521 is the LP engines only, per its own title), on a MILP's
/// integer columns (a candidate basis over relaxed bounds proves nothing about integrality),
/// and on a basis larger than a row cap chosen so a dense exact Gaussian elimination stays
/// fast (see exact_verify.cpp).
[[nodiscard]] ExactResult verify_basis_exact(const Model& model, const Solution& solution);

}  // namespace sankhya::exact
