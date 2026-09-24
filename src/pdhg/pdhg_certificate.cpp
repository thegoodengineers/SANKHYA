// SPDX-License-Identifier: Apache-2.0
// SANKHYA - infeasibility and unboundedness from PDHG restart differences (#484). See
// pdhg_certificate.hpp for the reference and the contract.

#include "pdhg_certificate.hpp"

#include <cstddef>

#include <fmt/format.h>

#include "sankhya/certificate.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::pdhg {

bool detect_certificate_from_restart(const Model& model, const Scaling& scaling,
                                     const std::vector<double>& dx, double dx_norm,
                                     const std::vector<double>& dy, double dy_norm,
                                     SolveStatus* status, std::vector<double>* certificate,
                                     std::string* message) {
  // Infeasibility first: it is a complete claim on its own, where a ray is only half of one.
  if (dy_norm > tol::kPdhgDetectionMinNorm) {
    const auto m = dy.size();
    std::vector<double> candidate(m);
    // row_dual's convention is sense * (-y); the sign Farkas' lemma needs is a property of
    // the proof, not of this engine's Lagrangian, so both signs are offered to the checker.
    for (std::size_t u = 0; u < m; ++u) candidate[u] = -dy[u] * scaling.row[u];
    for (int attempt = 0; attempt < 2; ++attempt) {
      std::string why;
      if (farkas_proves_infeasible(model, candidate, &why)) {
        *status = SolveStatus::kInfeasible;
        *certificate = std::move(candidate);
        *message =
            fmt::format("a restart's iterate difference is a certified Farkas ray; {}", why);
        return true;
      }
      for (double& v : candidate) v = -v;
    }
  }
  if (dx_norm > tol::kPdhgDetectionMinNorm) {
    const auto n = dx.size();
    std::vector<double> ray(n);
    for (std::size_t u = 0; u < n; ++u) ray[u] = dx[u] * scaling.column[u];
    std::string why;
    if (ray_proves_unbounded(model, ray, &why)) {
      // A ray proves the dual infeasible; whether the primal has a feasible point to move
      // along it from is the other half, which no restart difference establishes. An
      // infeasible model with an improving direction (min -x1; x2 + x3 <= 1; 2 x2 + 3 x3 >= 4;
      // x >= 0) was reported unbounded at iteration 1 before this (review of #652).
      *status = SolveStatus::kInfeasibleOrUnbounded;
      *certificate = std::move(ray);
      *message = fmt::format(
          "a restart's iterate difference is a certified improving ray, so the dual is "
          "infeasible; whether the primal has a feasible point is not established, so the "
          "model is infeasible or unbounded; {}",
          why);
      return true;
    }
  }
  return false;
}

}  // namespace sankhya::pdhg
