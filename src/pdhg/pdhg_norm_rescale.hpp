// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bound and objective rescaling, and the initial primal weight ||c|| / ||b||, for
// the LP PDHG (#482 item 3). Both off by default, each behind its own option:
//
//   pdhg_bound_objective_rescaling   after the Ruiz and Pock-Chambolle scaling, divide the
//                                    scaled objective by gamma = ||c||_2 + 1 and every scaled
//                                    bound, row and column, by beta = ||b||_2 + 1;
//   pdhg_initial_weight_from_norms   start the primal weight at ||c||_2 / ||b||_2 of the
//                                    problem the iteration runs on, instead of 1.
//
// b is the vector of row right-hand sides as the engine already measures it (problem.
// bound_norm): a row's lower bound when finite, else its upper bound when finite, else zero.
//
// WHY THE RESCALED PROBLEM HAS THE SAME SOLUTION. Write the scaled problem as
//     min c'x  s.t.  A x in C,  x in X
// and the rescaled one as min (c / gamma)'z  s.t.  A z in C / beta,  z in X / beta. With
// x = beta z its Lagrangian (c / gamma)'z + w'A z - sigma_(C / beta)(w) is
// (1 / (gamma beta)) [c'x + (gamma w)'A x - sigma_C(gamma w)], the support function being
// positively homogeneous, so (z, w) is a saddle point of the rescaled problem exactly when
// (beta z, gamma w) is one of the scaled problem. The matrix is untouched, so ||A||_2, the
// step condition and pdhg_constant_step's bound are unchanged. The rescaling is folded into
// the Scaling itself - column multipliers times beta, row multipliers times gamma, cost over
// gamma, bounds over beta - so every consumer that maps x = column * z and y = row * w
// (the unscaling, the warm start, the feasibility polisher, the certificate rays) stays
// right without knowing about it. Termination is still measured on the original model.
//
// References
//   [PDLP]  Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue & Schudy, "Practical large-scale
//           linear programming using primal-dual hybrid gradient", NeurIPS 2021, section 3.2:
//           the primal weight initialised to ||c||_2 / ||b||_2 when both are nonzero.
//   [LPY25] Lu, Peng & Yang, "cuPDLPx: a further enhanced GPU-based first-order solver for
//           linear programming", arXiv:2507.14051 (2025), which the issue cites for b and c
//           rescaled by their norms. The + 1 in each denominator is this project's choice, so
//           a zero or tiny vector is left as it is rather than blown up, and is not a
//           quotation of the paper.
// Written from the papers; no solver's source was consulted.
#pragma once

#include "../la/scaling.hpp"

namespace sankhya::pdhg {

/// ||b||_2 of a scaled problem, b as described above.
[[nodiscard]] double scaled_bound_norm(const Scaling& scaling);

/// ||c||_2 of a scaled problem's objective.
[[nodiscard]] double scaled_cost_norm(const Scaling& scaling);

/// What rescale_bounds_and_objective() divided by.
struct BoundObjectiveRescale {
  double bound_divisor = 1.0;      ///< beta = ||b||_2 + 1
  double objective_divisor = 1.0;  ///< gamma = ||c||_2 + 1
};

/// Applies the rescaling above to `scaling` in place.
BoundObjectiveRescale rescale_bounds_and_objective(Scaling* scaling);

/// ||c||_2 / ||b||_2 of the scaled problem, clamped to the engine's primal-weight range, or
/// 1 when either norm is zero.
[[nodiscard]] double initial_primal_weight_from_norms(const Scaling& scaling);

}  // namespace sankhya::pdhg
