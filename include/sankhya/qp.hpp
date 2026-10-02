// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex quadratic programming.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::qp {

/// A starting point for solve_convex_qp_ipm, in the model's own columns and rows (#494, #893):
/// typically a parent branch-and-bound node's iterate, offered to a child so its interior point
/// does not start cold. Only `col_value` is required; `row_dual` seeds the equality multipliers
/// when given (model sense, matching Solution::row_dual) and is otherwise left at the engine's
/// usual zero start. A value that falls outside the child's bounds - branching only tightens
/// them, but a caller may pass anything - is pulled back to the interior by the same margin the
/// cold start uses, and the bound multipliers are always recomputed from the residual there, not
/// carried over: this is a PARTIAL warm start (the point, and optionally the duals on the rows),
/// not a continuation of the proximal path, its regularization, or the bound multipliers.
struct QpIpmWarmStart {
  std::vector<double> col_value;
  std::vector<double> row_dual;

  [[nodiscard]] bool empty() const noexcept { return col_value.empty(); }
};

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
                                           sankhya::SolveControl* control = nullptr,
                                           const QpIpmWarmStart* warm_start = nullptr);

}  // namespace sankhya::qp
