// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex quadratic programming.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::qp {

/// Solve a convex QP:
///
///     minimize    offset + c'x + 0.5 x' Q x
///     subject to  row_lower <= A x <= row_upper
///                 col_lower <=   x  <= col_upper
///
/// A non-convex Hessian is REFUSED (kModelError), never solved to whatever local point the
/// iteration happens to reach. Convexity is decided before any arithmetic starts; see
/// src/qp/convexity.hpp.
[[nodiscard]] Solution solve_convex_qp(const Model& model, const Options& options,
                                       Logger& logger,
                                       sankhya::SolveControl* control = nullptr);

/// The same problem by a proximal interior point (#490): Mehrotra predictor-corrector on the
/// regularized, quasi-definite augmented system, factored by the project's sparse LDL^T.
/// Chosen with qp_algorithm=ipm; the same convexity refusal comes first. See
/// src/qp/qp_ipm.cpp for the method, its references and what this first slice leaves out.
[[nodiscard]] Solution solve_convex_qp_ipm(const Model& model, const Options& options,
                                           Logger& logger,
                                           sankhya::SolveControl* control = nullptr);

}  // namespace sankhya::qp
