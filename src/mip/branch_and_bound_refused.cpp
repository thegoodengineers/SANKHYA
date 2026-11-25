// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: an integral node LP point the incumbent test refuses, tried
// once more in the model's own units (#1348).
//
// A node LP is solved on the scaled model, and its feasibility tolerance is the scaled
// model's: a row short by 6e-7 that scaling multiplies by 0.14 is short by 8e-8 there, and
// the LP is optimal with it. The incumbent test (offer_incumbent) measures the point against
// the model as given, at an absolute tolerance, and refuses it. When the relaxation is
// integral and no integer column was moved by the rounding, #1318 has nothing to branch on
// and sets the node aside, and a model whose answer is one pivot away ends numerical_error:
//
//     min x1:  3 x0 - 2 x1 >= 2,  3 x1 + x2 >= 8,
//              x0 in [2 - 2e-7, 7/3],  x1 integer in [2, 3],  x2 >= 2 - 2e-7
//
// has its optimum at (2, 2, 2), and the root LP stops at x0 = 2 - 2e-7.
//
// What the node is missing is a point with the same integer values that is feasible in the
// units the incumbent test uses. That is an LP: the model with every integer column fixed
// at its value in the refused point, solved WITHOUT scaling, so the simplex's tolerance is
// the model's. Fixing the integer columns and re-solving the rest is how a mixed-integer
// point is completed (the diving and rounding literature throughout; Berthold, "Primal
// heuristics for mixed integer programs", 2006, sec. 3.1). The result goes through
// offer_incumbent like every other candidate, so nothing is accepted that the test would
// not have accepted from anywhere else.

#include <algorithm>
#include <cmath>
#include <vector>

#include "branch_and_bound_internal.hpp"

namespace sankhya::mip {

bool BranchAndBound::offer_in_model_units(const std::vector<double>& x) {
  // The completion is an LP; an MIQP's would be a QP, and its node points are the
  // first-order engine's, refused or not for reasons of their own.
  if (quadratic_ || pool_complete_) return false;
  Model lp = original_;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    double v = std::round(x[u]);
    if (is_finite_bound(lp.col_lower[u])) v = std::max(v, lp.col_lower[u]);
    if (is_finite_bound(lp.col_upper[u])) v = std::min(v, lp.col_upper[u]);
    lp.col_lower[u] = v;
    lp.col_upper[u] = v;
  }
  std::fill(lp.col_type.begin(), lp.col_type.end(), VarType::kContinuous);
  Options options = node_options_;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_bool("scaling", false);
  if (limits_.has_time_limit()) {
    options.set_double("time_limit", limits_.remaining_seconds(timer_.elapsed_seconds()));
  }
  const Solution completed = solve(lp, options);
  if (completed.status != SolveStatus::kOptimal ||
      completed.col_value.size() != static_cast<std::size_t>(original_.num_cols())) {
    return false;
  }
  bool feasible = false;
  offer_incumbent(completed.col_value, &feasible);
  return feasible;
}

}  // namespace sankhya::mip
