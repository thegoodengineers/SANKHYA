// SPDX-License-Identifier: Apache-2.0
// GPU PDHG on-device iteration loop via CUDA Graphs (#478).
//
// Current iteration path (pdhg_gpu.cu):
//   For each iteration:
//     cudaMemset 3 scalars  → device-side
//     launch primal/dual/interaction kernels
//     blocking dh_copy of 3 scalars  ← host round-trip
//     host evaluates adaptive step-size rule
//   Every 40 iterations: full residual/restart copy to host
//
// Target path (this file, when gpu_on_device_loop=true):
//   1. Capture a block of K=64 iterations as a CUDA Graph:
//        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)
//        loop K times: memset + primal + dual + interaction kernels
//                      step-size kernel (evaluates adaptive rule on device)
//                      KKT residual kernel in scaled space (scalings applied on device)
//        cudaStreamEndCapture → graph
//        cudaGraphInstantiate → exec
//   2. Replay: cudaGraphLaunch(exec, stream) — one launch per K iterations,
//      zero host synchronisation inside the block.
//   3. After each block: one small D→H copy of ~8 scalars for convergence/restart.
//   4. A^T stored explicitly in CSR (separate from the CSC of A) so both
//      SpMVs are row-parallel without atomics; deterministic reduction preserved.
//
// Current state: stub returning 0 (gpu_on_device_loop=false by default).
// The existing per-iteration path in pdhg_gpu.cu is unchanged.

#include "device.hpp"
#include "pdhg_graph.hpp"

namespace sankhya::gpu {

int pdhg_graph_block(int /*K*/, const Options& options) {
  if (!options.get_bool("gpu_on_device_loop") || !device_available(nullptr)) {
    return 0;
  }
  // TODO(#478): capture and replay the K-iteration CUDA Graph.
  return 0;
}

}  // namespace sankhya::gpu
