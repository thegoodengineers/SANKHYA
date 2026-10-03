// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bound and objective rescaling, and the initial primal weight ||c|| / ||b||
// (#482 item 3). See pdhg_norm_rescale.hpp for the argument and the references.

#include "pdhg_norm_rescale.hpp"

#include <algorithm>
#include <cmath>

#include "sankhya/tolerances.hpp"

namespace sankhya::pdhg {

double scaled_bound_norm(const Scaling& scaling) {
  double sum = 0.0;
  for (std::size_t i = 0; i < scaling.row_lower.size(); ++i) {
    const double lower = scaling.row_lower[i];
    const double upper = scaling.row_upper[i];
    const double b = is_finite_bound(lower) ? lower : (is_finite_bound(upper) ? upper : 0.0);
    sum += b * b;
  }
  return std::sqrt(sum);
}

double scaled_cost_norm(const Scaling& scaling) {
  double sum = 0.0;
  for (const double c : scaling.cost) sum += c * c;
  return std::sqrt(sum);
}

BoundObjectiveRescale rescale_bounds_and_objective(Scaling* scaling) {
  BoundObjectiveRescale r;
  r.bound_divisor = scaled_bound_norm(*scaling) + 1.0;
  r.objective_divisor = scaled_cost_norm(*scaling) + 1.0;
  const double beta = r.bound_divisor;
  const double gamma = r.objective_divisor;
  // An infinite bound stays infinite: dividing it would give the same infinity, and leaving
  // it alone keeps a sentinel value, if one is ever used, intact.
  const auto shrink = [beta](double bound) {
    return is_finite_bound(bound) ? bound / beta : bound;
  };
  for (std::size_t j = 0; j < scaling->cost.size(); ++j) {
    scaling->cost[j] /= gamma;
    scaling->col_lower[j] = shrink(scaling->col_lower[j]);
    scaling->col_upper[j] = shrink(scaling->col_upper[j]);
    scaling->column[j] *= beta;  // x = column * z, with x_scaled = beta z
  }
  for (std::size_t i = 0; i < scaling->row_lower.size(); ++i) {
    scaling->row_lower[i] = shrink(scaling->row_lower[i]);
    scaling->row_upper[i] = shrink(scaling->row_upper[i]);
    scaling->row[i] *= gamma;  // y = row * w, with y_scaled = gamma w
  }
  return r;
}

double initial_primal_weight_from_norms(const Scaling& scaling) {
  const double c = scaled_cost_norm(scaling);
  const double b = scaled_bound_norm(scaling);
  if (!(c > 0.0) || !(b > 0.0) || !std::isfinite(c) || !std::isfinite(b)) return 1.0;
  return std::clamp(c / b, tol::kQpPrimalWeightMin, tol::kQpPrimalWeightMax);
}

}  // namespace sankhya::pdhg
