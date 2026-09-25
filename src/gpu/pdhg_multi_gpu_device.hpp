// SPDX-License-Identifier: Apache-2.0
// SANKHYA - per-device state and kernels of the row-partitioned multi-GPU PDHG (#295).
//
// Included only by CUDA translation units (SANKHYA_ENABLE_CUDA builds).
//
// Every reduction on this path is reproducible bit for bit, run to run and card to card of
// one architecture, because nothing in it depends on the order threads or cards finish in:
//   - movement and interaction sums: CUB BlockReduce per block into a per-block slot, then
//     one block sums the slots in index order (k_finalize) - no atomicAdd, unlike the
//     single-card engine (#451);
//   - both matrix products are cuSPARSE CSR SpMV with CUSPARSE_SPMV_CSR_ALG2, which the
//     cuSPARSE documentation states gives deterministic (bit-wise) results for each run, on
//     explicitly stored A_k and A_k^T: the transpose operation on A_k is not covered by that
//     guarantee, so it is never used;
//   - the cross-card sum of the A_k^T y_k partials is formed on every card in slot order
//     0, 1, ..., K-1 (multi_gpu_exchange.hpp).
// Algorithm: restarted PDHG, Chambolle & Pock (JMIV 40(1) 2011) with the adaptive step and
// restarts of Applegate et al. (PDLP, NeurIPS 2021) and the GPU layout of Lu & Yang
// (cuPDLP, arXiv:2311.12180); the row-block distribution is the standard one for a
// distributed SpMV (Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., SIAM 2003,
// ch. 11, matrix-by-vector products).
#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstddef>
#include <vector>

#include "../la/scaling.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya::gpu::multi {

inline constexpr int kBlockSize = 256;
// The fixed-order sum kernel takes the slot pointers by value; this bounds the device list.
inline constexpr int kMaxDevices = 16;

struct DeviceState {
  int device_id = -1;
  int slot = 0;  // position in the device list = this card's slot in the exchange
  int num_slots = 0;
  int n = 0, local_m = 0, local_nnz = 0, row_start = 0;
  int blocks_n = 0, blocks_m = 0, block_stride = 0;
  cudaStream_t stream{};

  // n-vectors, replicated and bit-identical on every card.
  double *d_x{}, *d_xn{}, *d_ext{}, *d_dx{}, *d_aty{}, *d_xsum{};
  double *d_cost{}, *d_clo{}, *d_chi{};
  double* d_partial{};  // A_k^T y_k, this card's share of A^T y
  double* d_recv{};     // num_slots * n: slot k receives card k's partial (own slot unused)

  // local_m-vectors, this card's rows only.
  double *d_y{}, *d_yn{}, *d_dy{}, *d_ax{}, *d_adx{}, *d_ysum{}, *d_rlo{}, *d_rhi{};

  // Reductions: 3 * block_stride per-block sums, then [mv_x, mv_y, interaction].
  double* d_blocks{};
  double* d_scalars{};

  // A_k (local_m x n) and A_k^T (n x local_m), both CSR, 0-based.
  int *d_rp{}, *d_ci{}, *d_trp{}, *d_tci{};
  double *d_v{}, *d_tv{};
  void* d_spmv{};
  cusparseHandle_t cs{};
  cusparseSpMatDescr_t mat{}, mat_t{};
  cusparseDnVecDescr_t vn{}, vm{};

  DeviceState() = default;
  DeviceState(const DeviceState&) = delete;
  DeviceState& operator=(const DeviceState&) = delete;
  ~DeviceState();
};

/// Allocate and upload rows [row_start, row_end) of the scaled matrix, the replicated
/// problem vectors and the starting point; create the card's stream and cuSPARSE handle.
/// d.device_id, d.slot and d.num_slots must be set. Returns false on any CUDA failure.
[[nodiscard]] bool setup_device(DeviceState& d, const CsrView& full_csr, const Scaling& scaling,
                                const std::vector<double>& x0_scaled, int row_start,
                                int row_end, int n_cols);

// Everything below enqueues on d.stream and returns false on a launch or cuSPARSE error.
// The caller has made d.device_id current.

/// d_partial = A_k^T y_k.
[[nodiscard]] bool spmv_aty(DeviceState& d);
/// out (local_m) = A_k in (n).
[[nodiscard]] bool spmv_ax(DeviceState& d, double* in_n, double* out_m);
/// x_next, extrapolation and dx from d_aty; per-block movement sums into slot 0.
[[nodiscard]] bool launch_primal(DeviceState& d, double tau, double omega);
/// y_next and dy from d_ax; per-block movement sums into slot 1.
[[nodiscard]] bool launch_dual(DeviceState& d, double sigma, double omega);
/// Per-block sums of dy . adx into slot 2, then the fixed-order finalize of all three slots
/// into d_scalars, then the scalars' download to host_scalars (pinned, 3 doubles).
[[nodiscard]] bool launch_interaction_and_scalars(DeviceState& d, double* host_scalars);
/// Running sums for the average: xsum += x, ysum += y.
[[nodiscard]] bool launch_accumulate(DeviceState& d);
/// Synchronous download of count doubles on d.stream.
[[nodiscard]] bool download(DeviceState& d, const double* src, double* dst, std::size_t count);

}  // namespace sankhya::gpu::multi
