// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the general NLP engine's entry point (NLP stage 2).
//
// THE SEAM. `Model` and `solve()` are frozen and describe linear rows; a nonlinear model is an
// nlp::NonlinearModel, and this is its solve() - the same shape (model, options, control) ->
// Solution, beside the frozen one as solve_global() is for quadratic rows. The CLI reaches it
// for a .nl file and the C API for a handle with nonlinear parts.
//
// THE ANSWER'S VECTORS are in the model's terms: col_value and col_dual per column,
// row_activity and row_dual per ROW OF THE NLP FORM - the model's linear rows first, then
// its nonlinear constraints in their order (nlp_problem.hpp). Multipliers use the LP sign
// convention of the rest of the project (a positive multiplier prices a lower bound, in the
// model's own sense), so tools/verify_solution.py reads them as it reads an LP's.
//
// THE STATUS IS HONEST ABOUT LOCALITY. `optimal` only when the KKT check at the project
// tolerances passed (nlp_kkt_check.hpp) AND the continuous problem is proved convex by the
// composition rules (NonlinearModel::convexity()), which makes a KKT point global;
// `locally_optimal` when the check passed and convexity is not proved; `locally_infeasible`
// when the method found a local minimizer of the violation. Nothing is called optimal on
// the method's word.
#pragma once

#include <vector>

#include "nlp/nonlinear_model.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::nlp {

/// Solve a nonlinear model whose columns are all continuous. A model with integer columns is
/// refused here (kNotSolved, with the reason); a MINLP goes to the branch and bound.
[[nodiscard]] Solution solve_nlp(const NonlinearModel& model, const Options& options,
                                 SolveControl* control = nullptr);

/// The continuous relaxation - integrality IGNORED - from `start` (empty: the model's own
/// start, else zeros). The node solver of the NLP-based branch and bound.
[[nodiscard]] Solution solve_nlp_relaxation(const NonlinearModel& model, const Options& options,
                                            const std::vector<double>& start,
                                            SolveControl* control, Logger& logger);

}  // namespace sankhya::nlp
