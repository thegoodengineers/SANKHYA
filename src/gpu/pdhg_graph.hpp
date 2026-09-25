// SPDX-License-Identifier: Apache-2.0
// GPU PDHG on-device iteration loop via CUDA Graphs (#478).
//
// References:
//   Lu & Yang, cuPDLP.jl, arXiv:2311.12180 (the host/device split: iterate on the device,
//     check termination on a stride)
//   Applegate et al., PDLP, NeurIPS 2021, section 3.1 (the adaptive step-size rule this
//     moves onto the device unchanged)
//   NVIDIA, CUDA C++ Programming Guide, "CUDA Graphs" (stream capture, instantiate, launch)
//
// The per-iteration path in pdhg_gpu.cu copies three scalars to the host every iteration so
// the host can apply the adaptive step rule; that copy synchronises the device and is the
// dominant cost on small and medium models. With gpu_on_device_loop the rule runs in a
// one-thread kernel, an accepted step is committed by a copy kernel instead of a host-side
// pointer swap (a captured graph needs fixed addresses), and one iteration's launches are
// captured once as a CUDA graph and replayed K times per host synchronisation.
#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

namespace sankhya::gpu {

/// Device buffers and handles the loop runs on, owned by the caller (pdhg_gpu.cu).
struct DeviceLoopBuffers {
  // Iterates and their scratch, at fixed addresses for the life of the loop.
  double *x{}, *xn{}, *ext{}, *dx{}, *aty{}, *xsum{};
  double *y{}, *yn{}, *dy{}, *ax{}, *adx{}, *ysum{};
  double *axc{}, *axn{};  ///< two-mat-vec cache (#479); null when it is off
  const double *cost{}, *col_lo{}, *col_hi{}, *row_lo{}, *row_hi{};
  /// One slot per block of the fused kernels: [0, bn) movement x, [bn, bn + bm) movement y,
  /// [bn + bm, bn + 2 bm) interaction, summed in a fixed order by the step kernel (#478).
  double* partials{};
  // Step state, on the device.
  double* eta{};          ///< step size
  double* omega{};        ///< primal weight (written by the host at a restart)
  long long* accepted{};  ///< accepted iterations so far
  int* accept_flag{};     ///< the last step's verdict, read by the commit kernel
  double eta_ceil = 0.0;  ///< the host path's clamp, 1e3 / ||A||
  int n = 0, m = 0, nnz = 0;
  bool two_matvec = false;
  cusparseHandle_t cusparse{};
  cusparseSpMatDescr_t matrix{};
  /// A^T held explicitly in CSR (deterministic mode, #478); A^T y then runs as a
  /// non-transpose product on it. Null: cuSPARSE's transpose product of `matrix`.
  cusparseSpMatDescr_t matrix_t{};
  cusparseSpMVAlg_t spmv_alg = CUSPARSE_SPMV_ALG_DEFAULT;
  cusparseDnVecDescr_t vec_n{}, vec_m{};
  void* spmv_buffer{};
};

/// One captured iteration replayed `block` times per run_block().
class DeviceLoop {
 public:
  DeviceLoop() = default;
  DeviceLoop(const DeviceLoop&) = delete;
  DeviceLoop& operator=(const DeviceLoop&) = delete;
  ~DeviceLoop();

  /// Captures the iteration. False when capture or instantiation fails; the caller then keeps
  /// the per-iteration path.
  bool init(const DeviceLoopBuffers& buffers, int block);
  /// Replays the block and waits for it. False on any CUDA error.
  bool run_block();
  [[nodiscard]] cudaStream_t stream() const { return stream_; }

 private:
  bool record_iteration();
  DeviceLoopBuffers b_{};
  int block_ = 0;
  cudaStream_t stream_{};
  cudaGraph_t graph_{};
  cudaGraphExec_t exec_{};
};

}  // namespace sankhya::gpu
