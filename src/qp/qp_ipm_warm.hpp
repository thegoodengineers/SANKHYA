// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the warm start of the proximal QP interior point (#494, #893): a saved iterate
// shifted into the interior of the model it is offered to. See qp_ipm_warm.cpp.
#pragma once

#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/qp.hpp"

#include "qp_ipm_system.hpp"

namespace sankhya::qp::ipm_detail {

/// What warm_start_iterate did.
struct WarmShift {
  double rho = 0.0;        ///< the proximal parameters to start from
  double delta = 0.0;      ///< (never below the configured floor)
  double mu = 0.0;         ///< the centring target the multipliers were moved towards
  Index moved_inside = 0;  ///< slacks to a bound raised to the shift distance
  Index recentred = 0;     ///< multipliers moved into the centrality box
};

/// Seed the iterate (v, y, z_l, z_u) over `s`'s columns and rows from `warm`, which is over
/// `model`'s own columns and rows and whose col_value has the model's size. Every vector is
/// resized and overwritten.
[[nodiscard]] WarmShift warm_start_iterate(const Model& model, const Standard& s,
                                           const QpIpmWarmStart& warm,
                                           double regularization_floor, std::vector<double>* v,
                                           std::vector<double>* y, std::vector<double>* zl,
                                           std::vector<double>* zu);

/// The iterate (v, y, z_l, z_u) of `s`, on `model`'s own columns and rows, as a warm start
/// for a later solve.
[[nodiscard]] QpIpmWarmStart save_iterate(const Model& model, const Standard& s,
                                          const std::vector<double>& v,
                                          const std::vector<double>& y,
                                          const std::vector<double>& zl,
                                          const std::vector<double>& zu, double rho,
                                          double delta, double mu, Count iterations);

}  // namespace sankhya::qp::ipm_detail
