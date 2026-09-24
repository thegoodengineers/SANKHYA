// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Gondzio's multiple centrality correctors (#472). See centrality.hpp for the
// method and its references.

#include "ipm/centrality.hpp"

#include <algorithm>
#include <cmath>

#include "sankhya/tolerances.hpp"

namespace sankhya::ipm {

double aspiration_step(double alpha) {
  return std::min(1.5 * alpha + 0.3, 1.0);
}

Index add_centrality_term(const std::vector<double>& s, const std::vector<double>& ds,
                          const std::vector<double>& z, const std::vector<double>& dz,
                          const std::vector<bool>& present, double alpha_p, double alpha_d,
                          double mu_target, std::vector<double>* r_mu) {
  const double low = tol::kIpmCentralityBetaMin * mu_target;
  const double high = tol::kIpmCentralityBetaMax * mu_target;
  Index corrected = 0;
  for (std::size_t k = 0; k < s.size(); ++k) {
    if (!present[k]) continue;
    const double v = (s[k] + alpha_p * ds[k]) * (z[k] + alpha_d * dz[k]);
    double t = 0.0;
    if (v < low) {
      t = low - v;
    } else if (v > high) {
      t = std::max(high - v, -high);
    } else {
      continue;
    }
    (*r_mu)[k] += t;
    ++corrected;
  }
  return corrected;
}

int corrector_budget(double factor_nonzeros, double dimension, int cap) {
  if (cap <= 0 || dimension <= 0.0) return 0;
  const double ratio = (factor_nonzeros / dimension + 1.0) / 4.0;
  int budget = 1;
  for (double r = ratio; r > 2.0 && budget < cap; r *= 0.5) ++budget;
  return std::min(budget, cap);
}

}  // namespace sankhya::ipm
