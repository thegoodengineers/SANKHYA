// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a returned solution checked against the model it was asked about. TESTS ONLY.
//
// Both checks recompute everything from the model and the solution's vectors; nothing the
// solver measured about itself is read. They are what the general-form oracle fuzz
// (test_fuzz_general_form.cpp) holds every optimal answer to, after presolve and postsolve.
#pragma once

#include <string>

#include "sankhya/model.hpp"

namespace sankhya::oracle {

/// Empty when the solution's point satisfies the model at the solver's tolerances and is
/// worth the objective reported. The row test is the verifier's (tools/verify_solution.py):
/// a violation is measured against the largest term of its row.
[[nodiscard]] std::string point_violation(const Model& model, const Solution& s);

/// Empty when the solution's row duals certify `expected` on the model. In minimization
/// space, with d = c - A'y, every feasible x has
///     c'x  =  y'(Ax) + d'x  >=  sum_i min(y_i L_i, y_i U_i) + sum_j min(d_j l_j, d_j u_j),
/// and the right-hand side reaches the optimum exactly when y is an optimal dual vector
/// (Chvatal, "Linear Programming", 1983, ch. 5). A multiplier pointing at a side with no
/// bound makes the bound minus infinity; one within the dual feasibility tolerance is taken
/// as zero there, which is the solver's own promise. The reduced costs are recomputed; the
/// reported ones only have to agree with them.
[[nodiscard]] std::string dual_violation(const Model& model, const Solution& s,
                                         double expected);

}  // namespace sankhya::oracle
