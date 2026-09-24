// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Gondzio's multiple centrality correctors for the LP interior point (#472).
//
// References:
//   Gondzio, "Multiple centrality corrections in a primal-dual method for linear
//     programming", Computational Optimization and Applications 6 (1996) - the corrector:
//     from a trial point a little beyond the step the direction allows, the complementarity
//     products that fall outside [beta_min mu, beta_max mu] are pushed back to that interval,
//     through the factors already computed, and the corrected direction is kept only if it
//     lengthens the step.
//   Colombo & Gondzio, "Further development of multiple centrality correctors for interior
//     point methods", Computational Optimization and Applications 41 (2008) - the aspiration
//     step min(1.5 alpha + 0.3, 1) and the cap on how far a large product is pulled down.
//
// WHY IT PAYS. Each corrector is one more back-solve with the factorization the iteration
// already paid for. Where the factorization is most of an iteration's cost - a dense factor
// - trading solves for iterations is a good exchange, and the number of correctors allowed
// grows with the ratio of factorization work to solve work. That ratio is estimated from the
// factor's shape, never measured by a clock, so the same model takes the same path on every
// machine.
#pragma once

#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::ipm {

/// The aspiration step of Colombo & Gondzio (2008): min(1.5 alpha + 0.3, 1).
[[nodiscard]] double aspiration_step(double alpha);

/// Add Gondzio's centrality term to r_mu for one family of complementarity pairs (the lower
/// or the upper bounds): at the trial point (s + alpha_p ds, z + alpha_d dz) every product v
/// below beta_min * mu_target is raised to it, every product above beta_max * mu_target is
/// lowered to it but by no more than beta_max * mu_target, and the rest are left alone. Only
/// entries with present[k] are touched. Returns how many products were corrected.
Index add_centrality_term(const std::vector<double>& s, const std::vector<double>& ds,
                          const std::vector<double>& z, const std::vector<double>& dz,
                          const std::vector<bool>& present, double alpha_p, double alpha_d,
                          double mu_target, std::vector<double>* r_mu);

/// How many correctors an iteration may try, at most `cap`, from the factor's shape: the
/// factorization costs about sum_j c_j^2 multiply-adds for column counts c_j and a solve
/// about 4 sum_j c_j, so with the average column count c their ratio is near c / 4. One
/// corrector while the ratio is at most 2, one more for each doubling beyond (the constants
/// are tol::kIpmCentralitySolveWork, kIpmCentralityBudgetStart, kIpmCentralityBudgetGrowth).
/// 0 when `cap` is 0 (the option's default: off). The caller lowers it further where a solve
/// is not one back-solve (ipm.cpp, centrality_correctors).
[[nodiscard]] int corrector_budget(double factor_nonzeros, double dimension, int cap);

}  // namespace sankhya::ipm
