// SPDX-License-Identifier: Apache-2.0
// SANKHYA - certified shadow prices and sensitivity ranges (#757).
//
// The floating-point report (src/simplex/ranging.cpp) is re-derived from the reported basis
// in exact rational arithmetic: the duals y (B^T y = c_B), the reduced costs, the cost
// ranges (Chvatal, "Linear Programming", ch. 10, 1983) and the right-hand-side ranges, each
// compared with the float value and marked certified or corrected. A fixed variable never
// enters a basis, so it never limits a range: exact arithmetic applies that by the bounds,
// where the float code applies it by the status label.
//
// THE SHADOW PRICE INTERVAL. At a degenerate optimum the dual is not unique, and the optimal
// value v(t) of the model with row i's bounds shifted by t has a kink at t = 0. Its left and
// right derivatives are the correct shadow prices (Jansen, de Jong, Roos and Terlaky,
// "Sensitivity analysis in linear programming: just be careful!", EJOR 101, 1997). Where
// the RHS range extends past 0 on a side the basis stays optimal there and the derivative on
// that side is y_i. Where it does not, the lexicographic dual simplex runs in exact
// arithmetic on the infinitesimally shifted bounds (Gal, "Postoptimal Analyses, Parametric
// Programming, and Related Topics", 2nd ed., 1995, ch. 3): every pivot is degenerate at
// t = 0, so the point never moves, only the basis, until one is optimal for small t on that
// side; its y_i is the one-sided derivative. No such basis means no feasible point on that
// side and an infinite derivative.
//
// tools/verify_solution_sensitivity.py derives the same quantities with its own code from
// the .sol file alone, and tools/verify_solution.py rejects any mismatch.
#pragma once

#include <string>
#include <vector>

#include "exact/exact_verify.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

namespace sankhya::exact {

struct SensitivityResult {
  ExactVerdict verdict = ExactVerdict::kDeclined;
  std::string message;  ///< why declined or failed
  std::vector<Solution::ExactSensitivityEntry> columns;
  std::vector<Solution::ExactSensitivityEntry> rows;
};

/// `solution` must carry an optimal basis and the float ranging (option "ranging"). Declines
/// on what verify_basis_exact declines, and fails when the basis is not exactly optimal.
[[nodiscard]] SensitivityResult certify_sensitivity(const Model& model,
                                                    const Solution& solution,
                                                    double seconds = kInfinity);

/// certify_sensitivity, stored into `solution` (sensitivity_status and the exact entries)
/// and summarised in the log. solve() calls it under options "exact" and "ranging".
void apply_certified_sensitivity(const Model& model, Solution* solution, Logger& logger,
                                 double seconds);

}  // namespace sankhya::exact
