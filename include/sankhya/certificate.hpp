// SPDX-License-Identifier: Apache-2.0
// SANKHYA - certificates for the two verdicts that have no point to show (#191).
//
// `optimal` hands over a point and a basis, and anyone can check it. `infeasible` and
// `unbounded` hand over nothing, and until now the solver's own independent checker
// answered a correct infeasible verdict with REJECTED, because it was given an all-zero
// point and asked whether it satisfied the rows. That is the wrong question: there is no
// point, and the claim being made is about the whole feasible region.
//
// Both claims do have short proofs, and both are checkable in one pass over the matrix.
//
// FARKAS, for infeasibility. Take one multiplier y_i per row. With the model written as
//
//     row_lower <= A x <= row_upper,   col_lower <= x <= col_upper
//
// a multiplier y_i > 0 uses that row's LOWER bound and y_i < 0 its UPPER bound, so every
// feasible x satisfies the single aggregated inequality
//
//     d . x >= S,   where d = A'y and S = sum_i y_i * (the bound its sign selected).
//
// The largest d.x can be over the column box is M, each column taken to whichever of its own
// bounds the sign of d_j prefers. If M < S then no x in the box satisfies the aggregate, so
// the model has no feasible point at all. That is a complete proof, and it is the one
// Farkas's lemma guarantees exists whenever the model really is infeasible.
//
// A RAY, for unboundedness. A direction d that no bound blocks and along which the objective
// improves forever, checked together with a feasible point: x + t d is then feasible for all
// t >= 0 and the objective runs to minus infinity.
//
// Reference: Farkas, J., "Theorie der einfachen Ungleichungen", Journal fuer die reine und
// angewandte Mathematik 124 (1902); Schrijver, "Theory of Linear and Integer Programming"
// (1986), section 7.3, for the bounded-variable form used here.
//
// WHY THE SOLVER CHECKS ITS OWN CERTIFICATE. The engines compute these vectors on a SCALED
// model, under perturbed bounds, from a factorization that may have drifted. Any of those
// can turn a valid ray into a not-quite-valid one, and an invalid proof published as a proof
// is worse than no proof. So every certificate is validated against the ORIGINAL model
// before it is written, and dropped if it does not hold. tools/verify_solution.py then
// checks it again, sharing no code with this.
#pragma once

#include <string>
#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

namespace sankhya {

/// Does `y` prove `model` has no feasible point?
///
/// `y` must have one entry per row. Returns false, and fills `why` when it is not null, if
/// the multipliers lean on a bound the row does not have, if the aggregate is unbounded above
/// over the column box, or if the contradiction is not strict beyond rounding.
[[nodiscard]] bool farkas_proves_infeasible(const Model& model, const std::vector<double>& y,
                                            std::string* why = nullptr);

/// Does `d` prove `model`'s objective is unbounded, given that a feasible point exists?
///
/// `d` must have one entry per column. Feasibility of the starting point is NOT checked here
/// - it is the caller's other half of the claim, and tools/verify_solution.py checks it from
/// the written point. Returns false, and fills `why`, if any bound blocks the direction or
/// the objective does not strictly improve along it.
[[nodiscard]] bool ray_proves_unbounded(const Model& model, const std::vector<double>& d,
                                        std::string* why = nullptr);

/// Keep `solution`'s certificate only if it proves what its status claims, against the
/// ORIGINAL `model` - the check solve() has always applied (#191), exported (#297 review) so
/// a SolverEngine wrapper that produces a certificate (the simplex family) can apply the
/// identical check without solve()'s dispatcher.
///
/// The engines compute these on a scaled model, under perturbed bounds, from factors that may
/// have drifted, and a Farkas vector's SIGN depends on which bound the leaving variable
/// crossed. Rather than derive the convention and hope, both signs are tried and the proof is
/// checked here; a candidate that does not prove the claim is dropped and the message says
/// so. An unproven certificate published as a proof would be worse than the empty field this
/// project already uses to mean "no proof was produced".
void verify_and_keep_certificate(Solution* solution, const Model& model, Logger& logger);

}  // namespace sankhya
