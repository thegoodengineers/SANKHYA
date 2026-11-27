// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one round of primal iterative refinement (#1082). See lp_primal_refinement.hpp;
// the method is Gleixner, Steffy and Wolter, INFORMS J. Computing 28(3), 2016, sec. 3.

#include "core/lp_primal_refinement.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/lp_exact_dyadic.hpp"
#include "sankhya/types.hpp"

namespace sankhya {

using namespace dyadic_arith;  // Dyadic, dyadic(), scaled() (lp_exact_dyadic.hpp)

PrimalRefinement refine_primal(const Model& model, const std::vector<double>& x,
                               int max_scale_bits) {
  const auto n = static_cast<Sz>(model.num_cols());
  const auto m = static_cast<Sz>(model.num_rows());
  PrimalRefinement out;
  out.model = model;
  out.shift = x;
  // A x*, exactly.
  std::vector<Dyadic> activity(m);
  for (Sz j = 0; j < n; ++j) {
    if (x[j] == 0.0) continue;
    const Dyadic xj = dyadic(x[j]);
    const ColumnView col = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < col.size; ++k) {
      Dyadic& a = activity[static_cast<Sz>(col.rows[k])];
      a = a + dyadic(col.values[k]) * xj;
    }
  }
  // Every finite bound minus x* (or A x*), exactly, then rounded once: b - v, unscaled.
  std::vector<Dyadic> col_lower(n), col_upper(n), row_lower(m), row_upper(m);
  const auto measure = [&](double bound, const Dyadic& v, Dyadic* gap) {
    if (!is_finite_bound(bound)) return;
    *gap = dyadic(bound) - v;
  };
  double violation = 0.0;
  for (Sz j = 0; j < n; ++j) {
    measure(model.col_lower[j], dyadic(x[j]), &col_lower[j]);
    measure(model.col_upper[j], dyadic(x[j]), &col_upper[j]);
    violation = std::max({violation, scaled(col_lower[j], 0), -scaled(col_upper[j], 0)});
  }
  for (Sz i = 0; i < m; ++i) {
    measure(model.row_lower[i], activity[i], &row_lower[i]);
    measure(model.row_upper[i], activity[i], &row_upper[i]);
    violation = std::max({violation, scaled(row_lower[i], 0), -scaled(row_upper[i], 0)});
  }
  out.max_violation = violation;
  int bits = max_scale_bits;
  if (violation > 0.0) bits = std::clamp(-std::ilogb(violation), 0, max_scale_bits);
  out.scale_bits = bits;
  const auto refined = [&](double bound, const Dyadic& gap) {
    return is_finite_bound(bound) ? scaled(gap, -bits) : bound;
  };
  for (Sz j = 0; j < n; ++j) {
    out.model.col_lower[j] = refined(model.col_lower[j], col_lower[j]);
    out.model.col_upper[j] = refined(model.col_upper[j], col_upper[j]);
  }
  for (Sz i = 0; i < m; ++i) {
    out.model.row_lower[i] = refined(model.row_lower[i], row_lower[i]);
    out.model.row_upper[i] = refined(model.row_upper[i], row_upper[i]);
  }
  return out;
}

Solution unrefine_primal(const Model& model, const PrimalRefinement& refinement,
                         Solution refined) {
  if (refined.col_value.size() != refinement.shift.size()) return refined;
  for (Sz j = 0; j < refinement.shift.size(); ++j) {
    refined.col_value[j] =
        refinement.shift[j] + std::ldexp(refined.col_value[j], -refinement.scale_bits);
  }
  refined.recompute_quality(model);  // the row activities A x and the measures
  return refined;
}

}  // namespace sankhya
