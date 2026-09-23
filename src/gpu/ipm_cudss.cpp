// SPDX-License-Identifier: Apache-2.0
// GPU IPM via cuDSS (#489) — CPU-side stub.
//
// Architecture (Shin et al., arXiv:2508.16094):
//   The Mehrotra predictor-corrector loop (src/ipm/ipm.cpp) is unchanged on the host.
//   Only the normal-equations factorization step moves to the device:
//     host assembles A D A^T + epsilon I  (or the augmented KKT system)
//     cudssMatrixCreateCsr(...) wraps it as a cuDSS sparse handle
//     cudssAnalysis + cudssFactorize run once per outer iteration
//     cudssRefactorize used for warm-started inner iterations
//     cudssTriangularSolve replaces the CPU LDL^T solve
//     result copied back to host; Newton step logic unchanged
//   Dense-column handling follows #467 (dense columns split out before
//   cuDSS sees the matrix to avoid fill-in blow-up).
//
// Current state: stub returning an empty Solution. No cuDSS header is
// included until the licence check is recorded in PROVENANCE.md.

#include "ipm_cudss.hpp"

namespace sankhya::gpu {

Solution solve_ipm_cudss(const Model& /*model*/, const Options& options) {
  if (!options.get_bool("gpu_cudss_ipm")) {
    return {};
  }
  // TODO(#489): check cuDSS licence; add SANKHYA_ENABLE_CUDSS CMake option;
  // include <cudss.h> and implement the factorization loop.
  return {};
}

}  // namespace sankhya::gpu
