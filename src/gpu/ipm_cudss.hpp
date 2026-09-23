// SPDX-License-Identifier: Apache-2.0
// GPU interior-point method via NVIDIA cuDSS (#489).
// cuDSS is a sparse direct linear-solver library (same class as cuSOLVER),
// not an optimization solver. Licence to be verified before this merges.
//
// Reference: Shin et al., MadIPM GPU interior point, arXiv:2508.16094
#pragma once

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::gpu {

/// Solve the LP/QP with the interior-point method using cuDSS on the device
/// to factor the normal equations or augmented system.
/// Newton iteration logic stays on the host; only the factorization moves.
///
/// Returns an empty Solution (status=unknown) when:
///   - gpu_cudss_ipm=false (default), OR
///   - SANKHYA_ENABLE_CUDA is off at build time, OR
///   - cuDSS is unavailable at runtime.
/// The caller (ipm.cpp) falls back to the CPU LDL^T path transparently.
[[nodiscard]] Solution solve_ipm_cudss(const Model& model, const Options& options);

}  // namespace sankhya::gpu
