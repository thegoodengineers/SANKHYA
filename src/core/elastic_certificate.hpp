// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a Farkas certificate from the elastic LP (#559).
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::detail {

/// A Farkas vector for `model` read off the row duals of its elastic LP, or an empty vector
/// when that LP does not give one.
///
/// The elastic LP keeps every column bound and gives each finite row side a non-negative
/// slack that absorbs its violation:
///
///     min  sum(e_lo) + sum(e_hi)   s.t.  l <= A x + e_lo - e_hi <= u,  lc <= x <= uc.
///
/// It is always feasible when the column box is, so it cannot fail the way the original
/// model's own phase 1 can. At its optimum the row duals y satisfy |y_i| <= 1, A'y leans
/// only on finite column bounds, and by strong duality y'(row bounds) minus what the box
/// allows equals the optimal total violation. When that is positive, y is exactly a Farkas
/// certificate of the original model (Chinneck, Feasibility and Infeasibility in
/// Optimization, Springer 2008, ch. 8, for the elastic form).
///
/// Multipliers on the side a row does not have (rounding noise in practice) are set to zero
/// before the vector is returned; the caller still checks it with farkas_proves_infeasible,
/// so nothing returned here is trusted on its own.
[[nodiscard]] std::vector<double> farkas_from_elastic(const Model& model,
                                                      const Options& options, double seconds);

/// THE EXACT PHASE 1 (option exact_phase1): asks rational arithmetic whether `model`, a plain
/// LP whose solve ended numerical_error (or infeasible by presolve, with no certificate), is
/// infeasible as read into doubles. Its elastic LP (above) is solved in floating point without
/// presolve and by the simplex, for a basis, and exact::prove_infeasible_exact
/// (src/exact/exact_repair.hpp) repairs that basis to an exactly optimal one; a positive exact
/// optimum is a proof, and its exact row duals the certificate. On a proof `solution` becomes
/// kInfeasible with farkas_dual (the multipliers rounded) and exact_repair_farkas (exact), its
/// point dropped and its message saying how, and true is returned; otherwise `solution` is
/// untouched. All within `seconds`. Written for Netlib cplex2, infeasible with a floating-point
/// phase 1 that stalls 4.1e-6 from feasibility and an elastic optimum below the 1e-7 tolerance,
/// so no floating certificate.
bool prove_infeasible_exactly(const Model& model, Solution* solution, const Options& options,
                              Logger& logger, double seconds);

}  // namespace sankhya::detail
