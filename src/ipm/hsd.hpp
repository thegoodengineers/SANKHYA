// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the homogeneous self-dual embedding of the LP interior point (#475, ipm_hsd).
//
// The interior point in ipm.cpp solves the LP as given and, on a model with no optimum,
// stalls or stops on a limit with nothing it can prove. This path solves the simplified
// homogeneous self-dual embedding of the same LP instead (Xu, Hung and Ye 1996; Andersen and
// Andersen 2000; see hsd.cpp), which always has a solution: tau > 0 at the limit gives the
// optimum scaled by tau, kappa > 0 gives a ray, either a Farkas certificate of primal
// infeasibility or a primal ray of unboundedness. The certificate is reported in the
// Solution fields the simplex uses (farkas_dual, primal_ray) and only once this project's
// own checker (src/core/certificate.cpp) accepts it.
#pragma once

#include "la/scaling.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya::ipm {

/// Solve the continuous LP `model` through its homogeneous self-dual embedding. `model` is
/// the one the method iterates on: the Ruiz-scaled copy solve_scaled() builds, with `scaling`
/// its factors (used only to measure the iterate in the original units, #582), or the model
/// itself with `scaling` null. `original`, when not null, is the model `model` was scaled
/// from: a certificate is then reported only when it also holds there, mapped back. The time
/// limit is measured on `clock`.
///
/// The answer is in `model`'s units: optimal with a point and duals; infeasible with a Farkas
/// vector farkas_proves_infeasible() accepted against `model`; unbounded with a ray
/// ray_proves_unbounded() accepted and a primal feasible point from a second, zero-cost solve
/// of the embedding; otherwise a limit, a feasible point, or a numerical error, never a
/// verdict without its proof. No basis is produced.
[[nodiscard]] Solution solve_homogeneous(const Model& model, const Options& options,
                                         Logger& logger, SolveControl* control,
                                         const Timer& clock, const Scaling* scaling,
                                         const Model* original = nullptr);

/// Say once which of the default path's options do not apply to the embedding, which
/// factors the plain normal equations on the CPU.
void announce_homogeneous(const Options& options, Logger& logger);

/// The certificate in `solution`, mapped back from the scaled model, checked once more
/// against `model` in the caller's units. A verdict whose certificate does not hold there is
/// withdrawn: numerical_error, the vector cleared, and the message says why.
void keep_only_a_proved_certificate(const Model& model, Solution* solution, Logger& logger);

}  // namespace sankhya::ipm
