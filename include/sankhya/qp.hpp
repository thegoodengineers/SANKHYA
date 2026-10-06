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
/// does not start cold. Only `col_value` is required. Every other field is optional and is
/// used only when its size matches the model it is offered to (a cut row added or removed in
/// between leaves the row-sized ones out):
///
/// *   `row_dual`: the multipliers of the rows, in the model's sense (Solution::row_dual's
///     convention);
/// *   `col_z_lower`, `col_z_upper`, `row_z_lower`, `row_z_upper`: the bound multipliers of the
///     columns and of the rows' bounds, in MINIMISATION sense and non-negative, as the
///     iteration carries them; without them they are recomputed from the residual at the
///     shifted point, as #924's partial warm start did;
/// *   `rho`, `delta`: the proximal parameters the iterate was taken at; `mu`: its average
///     complementarity product (recorded; the shift recentres on the products at the
///     shifted point instead, see qp_ipm_warm.cpp); `iterations`: what the
///     solve that produced it had spent. Zero means unknown;
/// *   `retry_cold`: what to do when the warm run fails or stalls.
///
/// solve_convex_qp_ipm fills one in for the caller through QpIpmWarmResult::next: the iterate
/// at which its relative residuals and gap first fell below tol::kQpIpmWarmSaveLevel, an
/// advanced but still well-centred point, not the optimum (Gondzio 1998; see qp_ipm_warm.cpp).
struct QpIpmWarmStart {
  std::vector<double> col_value;
  std::vector<double> row_dual;
  std::vector<double> col_z_lower;
  std::vector<double> col_z_upper;
  std::vector<double> row_z_lower;
  std::vector<double> row_z_upper;
  double rho = 0.0;
  double delta = 0.0;
  double mu = 0.0;
  Count iterations = 0;
  /// When the warm run fails or stalls: solve again from the cold start (true), or hand the
  /// failure back as it is, for a caller with a cheaper test to run first (branch and bound
  /// decides the node by its LP, then runs the cold IPM itself).
  bool retry_cold = true;

  [[nodiscard]] bool empty() const noexcept { return col_value.empty(); }
};

/// What solve_convex_qp_ipm did with the warm start it was offered, and the iterate a later
/// solve can start from (#494, #893).
struct QpIpmWarmResult {
  /// The save point of this solve (see QpIpmWarmStart); empty when the solve never reached
  /// tol::kQpIpmWarmSaveLevel, which a solve that ends optimal always does.
  QpIpmWarmStart next;
  /// The offered start was usable and the iteration began from it.
  bool warm_used = false;
  /// The warm run failed (a numerical error, an iterate that stopped being finite) or
  /// stalled (tol::kQpIpmWarmStallWindow), and the model was solved again from the cold
  /// start; Solution::iterations counts both runs.
  bool fell_back = false;
  /// Iterations the abandoned warm run spent; zero when there was none.
  Count abandoned_iterations = 0;
  /// #981, qp_ipm_stall_handoff: the final status is kIterationLimit because the worst
  /// relative measure stalled (the stall test below, generalized to the cold run by that
  /// option), rather than because an explicit iteration_limit was reached.
  bool stalled = false;
  /// #981, qp_ipm_stall_handoff: the iterate at the point the solve stopped, on the model's
  /// own columns and rows, whatever that point's quality - unlike `next`, which is empty
  /// unless tol::kQpIpmWarmSaveLevel was reached. Filled whenever the solve ends with a
  /// point (every status but kInfeasible and kUnbounded), so a stall that never got close can
  /// still be handed to another engine with whatever it has.
  QpIpmWarmStart final_point;
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
/// A starting point for solve_convex_qp (#981): typically the interior point's iterate when
/// it has stalled short of the standard, in the model's own columns and rows, offered so the
/// first-order iteration does not start at the projection of zero. Only `col_value` is
/// required; `row_dual`, in the model's sense (Solution::row_dual's convention), is used when
/// its size matches the model's rows and left at zero otherwise. Honoured on the host
/// operator only (qp_gpu=false); the device operator declines it like a model it cannot take
/// and the iteration starts cold, which never costs an answer.
struct QpFirstOrderWarmStart {
  std::vector<double> col_value;
  std::vector<double> row_dual;

  [[nodiscard]] bool empty() const noexcept { return col_value.empty(); }
};

[[nodiscard]] Solution solve_convex_qp(const Model& model, const Options& options,
                                       Logger& logger, sankhya::SolveControl* control = nullptr,
                                       const QpFirstOrderWarmStart* warm_start = nullptr);

/// The same problem by a proximal interior point (#490): Mehrotra predictor-corrector on the
/// regularized, quasi-definite augmented system, factored by the project's sparse LDL^T.
/// Chosen with qp_algorithm=ipm; the same convexity refusal comes first. See
/// src/qp/qp_ipm.cpp for the method, its references and what this first slice leaves out.
/// `warm_start`, when usable, replaces the cold start (src/qp/qp_ipm_warm.cpp); `warm_result`,
/// when given, receives what happened to it and this solve's own save point.
[[nodiscard]] Solution solve_convex_qp_ipm(const Model& model, const Options& options,
                                           Logger& logger,
                                           sankhya::SolveControl* control = nullptr,
                                           const QpIpmWarmStart* warm_start = nullptr,
                                           QpIpmWarmResult* warm_result = nullptr);

}  // namespace sankhya::qp
