// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU-accelerated restarted PDHG for LP.
//
// References (written from the papers; per ENGINEERING_RULES.md no solver source was
// consulted):
//   [CP11]  Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//           applications to imaging", JMIV 40(1), 2011. Algorithm 1 is the iteration below.
//   [PDLP]  Applegate et al., "Practical Large-Scale LP using PDHG", NeurIPS 2021.
//           Sections 3.1 (adaptive step size), 3.2 (primal weight), 4.3 (restarts).
//   [cuPDLP] Lu & Yang, "cuPDLP.jl: A GPU Implementation of Restarted PDHG for LP",
//           arXiv:2311.12180. GPU design reference.
//   [LPY25] Lu, Peng & Yang, "cuPDLPx", arXiv:2507.14051: the constant step 0.998 / ||A||_2
//           (pdhg_constant_step, #482, here on the proved upper bound of
//           src/la/operator_norm.hpp) and the PID primal weight (pdhg_primal_weight_pid,
//           #482, the QP engine's controller). Both host-side decisions, as in
//           src/pdhg/pdhg.cpp, and both off by default; see that file for the argument.
//
// CPU/GPU split:
//   GPU  SpMVs (cuSPARSE), primal/dual coordinate updates, running-sum accumulation,
//        movement and interaction reductions (CUB BlockReduce per block, then one block
//        summing the block totals in a fixed order, pdhg_reduce.cuh, #478; no atomics).
//   CPU  Preconditioning (one-off), the restart and stopping decisions on the scalars the
//        convergence evaluation returns, and scalar step-size arithmetic.
//   The convergence evaluation itself (unscaled KKT residuals, gap, restart distances of the
//   current and the average iterate, every 40 accepted steps) runs on the device with
//   gpu_device_evaluation (the default, #478 item 3, pdhg_device_eval.cuh) and on the host
//   (pdhg::evaluate, the reference) without it.
//
// The CSR matrix is built on CPU from CsrView(scaling.matrix) and uploaded once.
// The GPU VRAM ceiling on the primary test machine (RTX 5050) is 6 GB.
//
// DETERMINISM (#383, #478). With deterministic=true the same input gives the same bits every
// run. Two things are needed and both are here: the step-rule scalars are summed in a fixed
// order (pdhg_reduce.cuh; always on: one single-block kernel takes the place of the memset
// that zeroed the atomic targets every iteration), and both sparse products run
// as NON-TRANSPOSE products with CUSPARSE_SPMV_CSR_ALG2, which the cuSPARSE documentation
// (cusparseSpMV, "Algorithms") states gives bit-wise identical results run to run for CSR
// with opA = NON_TRANSPOSE only; a TRANSPOSE product is not covered. A^T y therefore runs on
// A^T stored explicitly in CSR, which is the CSC form of A the scaling already holds, at the
// cost of a second copy of the matrix on the device. Without deterministic=true the default
// algorithm and cuSPARSE's transpose product are kept, as before.

#include "pdhg_gpu.hpp"

#include "../pdhg/pdhg_evaluate.hpp"
#include "../pdhg/pdhg_refinement.hpp"
#include "../pdhg/pdhg_trace.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// fmt/format.h uses Unicode string literals that nvcc cannot parse; use snprintf instead.
#include <cstdio>
#include <string>

#include "../core/resource_limits.hpp"
#include "../core/stop_controller.hpp"
#include "../la/operator_norm.hpp"
#include "../la/scaling.hpp"
#include "../qp/qp_first_order_accel.hpp"
#include "device.hpp"
#include "pdhg_device_eval.cuh"
#include "pdhg_graph.hpp"
#include "pdhg_reduce.cuh"
#include "sankhya/pdhg.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya::gpu {
namespace {

constexpr Count kEvaluationInterval = 40;
constexpr int kBlockSize = 256;

// ---- Error-check macros -------------------------------------------------
// Each returns false from the enclosing function, which the caller treats as a GPU failure
// and either logs + falls back, or frees resources before returning.

#define CUDA_CHECK(expr)                     \
  do {                                       \
    if ((expr) != cudaSuccess) return false; \
  } while (0)

#define CS_CHECK(expr)                                   \
  do {                                                   \
    if ((expr) != CUSPARSE_STATUS_SUCCESS) return false; \
  } while (0)

// A kernel launch reports its failure (bad configuration, out of resources) through
// cudaPeekAtLastError, not through the launch statement, and not reliably through a later
// copy. Checked after every launch below and mapped to the same fallback as any other
// device failure.
static bool launch_ok() {
  return cudaPeekAtLastError() == cudaSuccess;
}

// ---- cuSPARSE scalar constants (host pointers, valid as alpha/beta) -----
static const double kOne = 1.0;
static const double kZero = 0.0;
// Single-precision counterparts for the mixed-precision sparse products (#982, see
// pdhg_precision below). The adaptive step rule, the restart logic and the projections stay
// in double; only A x and A^T y run through these when pdhg_precision=mixed.
static const float kOneF = 1.0f;
static const float kZeroF = 0.0f;

// ---- CUDA kernels -------------------------------------------------------

// [CP11] Algorithm 1, primal half-step, fused with [PDLP] §3.1 movement_x reduction (#280).
// Combining the update with the reduction eliminates one kernel launch and one global-memory
// round-trip (write dx to d_tmpn, read it back for CUB DeviceReduce) per iteration.
//   x_next = proj_[col_lo, col_hi](x - tau*(cost + at_y))
//   extrapolated = 2*x_next - x    ([CP11] over-relaxation)
//   dx = x_next - x
//   partials[block] = sum over the block of 0.5 * omega * dx[j]^2   (fixed order, #478)
__global__ void k_primal_fused(const double* __restrict__ x, const double* __restrict__ at_y,
                               const double* __restrict__ cost, const double* __restrict__ col_lo,
                               const double* __restrict__ col_hi, double* __restrict__ x_next,
                               double* __restrict__ extrapolated, double* __restrict__ dx,
                               double* partials, double tau, double omega, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  double mv_thread = 0.0;
  if (j < n) {
    double xnj = x[j] - tau * (cost[j] + at_y[j]);
    // project onto [col_lo[j], col_hi[j]]; infinite sides (IEEE) are treated as absent
    if (!isinf(col_lo[j]) && xnj < col_lo[j]) xnj = col_lo[j];
    if (!isinf(col_hi[j]) && xnj > col_hi[j]) xnj = col_hi[j];
    x_next[j] = xnj;
    const double dxj = xnj - x[j];
    extrapolated[j] = 2.0 * xnj - x[j];
    dx[j] = dxj;
    mv_thread = 0.5 * omega * dxj * dxj;
  }
  detail::write_block_partial<kBlockSize>(mv_thread, partials);
}

// [CP11] Algorithm 1, dual half-step, fused with [PDLP] §3.1 movement_y reduction (#280).
//   v = y + sigma * a_x
//   y_next = v - sigma * proj_[row_lo, row_hi](v / sigma)   (Moreau identity)
//   dy = y_next - y
//   partials[block] = sum over the block of 0.5 * dy[i]^2 / omega   (fixed order, #478)
__global__ void k_dual_fused(const double* __restrict__ y, const double* __restrict__ a_x,
                             const double* __restrict__ row_lo, const double* __restrict__ row_hi,
                             double* __restrict__ y_next, double* __restrict__ dy,
                             double* partials, double sigma, double omega, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double mv_thread = 0.0;
  if (i < m) {
    const double v = y[i] + sigma * a_x[i];
    double pv = v / sigma;
    if (!isinf(row_lo[i]) && pv < row_lo[i]) pv = row_lo[i];
    if (!isinf(row_hi[i]) && pv > row_hi[i]) pv = row_hi[i];
    const double yni = v - sigma * pv;
    y_next[i] = yni;
    const double dyi = yni - y[i];
    dy[i] = dyi;
    mv_thread = 0.5 * dyi * dyi / omega;
  }
  detail::write_block_partial<kBlockSize>(mv_thread, partials);
}

// [PDLP] §3.1 interaction: partials[block] = sum over the block of dy[i] * adx[i]
// Replaces the previous k_pointwise_mul + CUB DeviceReduce pair (#280).
__global__ void k_interaction_fused(const double* __restrict__ dy,
                                    const double* __restrict__ adx, double* partials, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double val = (i < m) ? dy[i] * adx[i] : 0.0;
  detail::write_block_partial<kBlockSize>(val, partials);
}

// The three block-partial arrays summed in a fixed order by one block (pdhg_reduce.cuh,
// #478): scalars[0] movement x, [1] movement y, [2] interaction. Launched <<<1, kBlockSize>>>.
__global__ void k_finish_scalars(const double* __restrict__ partials, int bn, int bm,
                                 double* __restrict__ scalars) {
  double totals[3] = {0.0, 0.0, 0.0};
  detail::sum_step_partials<kBlockSize>(partials, bn, bm, totals);
  if (threadIdx.x == 0) {
    scalars[0] = totals[0];
    scalars[1] = totals[1];
    scalars[2] = totals[2];
  }
}

// Two-mat-vec (#479): from A x_{k+1} and the cached A x_k derive both products the step
// needs, A xbar = 2 A x_{k+1} - A x_k for the dual update and A dx = A x_{k+1} - A x_k
// for the interaction term, by linearity, instead of two more sparse products.
__global__ void k_derive_products(const double* __restrict__ ax_next,
                                  const double* __restrict__ ax_cached,
                                  double* __restrict__ ax_bar, double* __restrict__ adx, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const double next = ax_next[i];
  const double cached = ax_cached[i];
  ax_bar[i] = 2.0 * next - cached;
  adx[i] = next - cached;
}

// Accumulate into running sums (for the average iterate)
__global__ void k_accum_n(const double* __restrict__ v, double* __restrict__ sum, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  sum[j] += v[j];
}

__global__ void k_accum_m(const double* __restrict__ v, double* __restrict__ sum, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  sum[i] += v[i];
}

// ---- Mixed precision (#982): cast kernels around the single-precision sparse products ----
// Single precision alone cannot reach the project's tolerances (1e-8 relative is below the
// ~6e-8 unit roundoff of float), so the matrix values and the vector operand of each product
// are cast down to float immediately before cusparseSpMV and the product's result is cast
// back up to double immediately after. Every other array - the iterate, the residual and
// restart state, the step-size scalars - stays double, as it already is in the code above;
// only the two sparse products' memory traffic (the dominant per-iteration cost PDLP cites,
// issue #982 "Why") is halved on the vector side and by a third on the matrix side. See
// pdhg_precision in src/util/options.cpp for the full scope of what is and is not covered.
__global__ void k_cast_d2f(const double* __restrict__ in, float* __restrict__ out, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) out[j] = static_cast<float>(in[j]);
}

__global__ void k_cast_f2d(const float* __restrict__ in, double* __restrict__ out, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) out[j] = static_cast<double>(in[j]);
}

// ---- Device-memory helpers ----------------------------------------------

static double* dev_zeros(std::size_t count) {
  if (count == 0) return nullptr;
  double* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(double)) != cudaSuccess) return nullptr;
  if (cudaMemset(p, 0, count * sizeof(double)) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

// Mixed precision (#982): the float counterpart of dev_zeros, for the matrix mirror and the
// per-call cast scratch the single-precision products use.
static float* dev_zeros_f(std::size_t count) {
  if (count == 0) return nullptr;
  float* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(float)) != cudaSuccess) return nullptr;
  if (cudaMemset(p, 0, count * sizeof(float)) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

static int* dev_int(std::size_t count) {
  if (count == 0) return nullptr;
  int* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(int)) != cudaSuccess) return nullptr;
  return p;
}

template <typename T>
static bool hd_copy(const T* host, T* dev, std::size_t count) {
  if (count == 0) return true;
  return cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

static bool dh_copy(const double* dev, double* host, std::size_t count) {
  if (count == 0) return true;
  return cudaMemcpy(host, dev, count * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
}

// ---- RAII wrapper for all GPU resources ---------------------------------

struct GpuState {
  // Iterates (scaled space)
  double *d_x{}, *d_xn{}, *d_ext{}, *d_dx{}, *d_aty{}, *d_xsum{};
  double *d_y{}, *d_yn{}, *d_dy{}, *d_ax{}, *d_adx{}, *d_ysum{};
  // Two-mat-vec (#479): A x_k and A x_{k+1}; allocated only when the option is on.
  double *d_axc{}, *d_axn{};
  // Device loop (#478): the step state the adaptive rule reads and writes on the device.
  double *d_eta{}, *d_omega{};
  long long* d_accepted{};
  int* d_accept_flag{};
  // Per-iteration scalars: [0]=movement_x, [1]=movement_y, [2]=interaction, written by
  // k_finish_scalars from the block partials (bn + 2 bm of them, one per block of the fused
  // kernels, #478) and downloaded in one transfer to avoid per-scalar sync stalls.
  double* d_scalars{};
  double* d_partials{};
  // Problem constants (device)
  double *d_cost{}, *d_clo{}, *d_chi{}, *d_rlo{}, *d_rhi{};
  // CSR matrix (device)
  int *d_rowptr{}, *d_colidx{};
  double* d_vals{};
  // A^T in CSR (the CSC arrays of A), deterministic mode only (#478): n + 1 row offsets.
  int *d_trowptr{}, *d_tcolidx{};
  double* d_tvals{};
  // Mixed precision (#982): the matrix values mirrored in float (same d_rowptr/d_colidx,
  // index arrays are unchanged by #982 point 2), plus n- and m-sized float scratch the cast
  // kernels cast into and out of around each single-precision product. Allocated only when
  // pdhg_precision=mixed; never combined with deterministic=true or the device loop in this
  // change (see the mixed_precision gating in solve_pdhg_gpu, #982).
  float* d_vals_f{};
  float *d_mix_n{}, *d_mix_m{};
  void* d_spmv_f{};
  cusparseSpMatDescr_t mat_f{};
  cusparseDnVecDescr_t vn_f{};
  cusparseDnVecDescr_t vm_f{};
  // cuSPARSE SpMV workspaces, ONE PER MATRIX DESCRIPTOR, each passed to every product on
  // its descriptor and to no other. A probe against the cuSPARSE of CUDA 12.4 on the L4
  // (#478) found that a descriptor keeps state in the workspace of its first product: with
  // the default algorithm as with CSR_ALG2, a later product on the same descriptor through a
  // different workspace, or a second descriptor's product through the same workspace, came
  // back wrong. d_spmv serves `mat` (A x, and A^T y as its transpose when A^T is not held);
  // d_spmv_t serves `mat_t` (deterministic mode only).
  void* d_spmv{};
  void* d_spmv_t{};
  // cuSPARSE handles
  cusparseHandle_t cs{};
  cusparseSpMatDescr_t mat{};
  cusparseSpMatDescr_t mat_t{};  // A^T, n x m; null unless deterministic
  cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
  cusparseDnVecDescr_t vn{};  // n-element dense vector
  cusparseDnVecDescr_t vm{};  // m-element dense vector

  int n{}, m{}, nnz{};

  ~GpuState() {
    if (vm_f) cusparseDestroyDnVec(vm_f);
    if (vn_f) cusparseDestroyDnVec(vn_f);
    if (mat_f) cusparseDestroySpMat(mat_f);
    if (vm) cusparseDestroyDnVec(vm);
    if (vn) cusparseDestroyDnVec(vn);
    if (mat_t) cusparseDestroySpMat(mat_t);
    if (mat) cusparseDestroySpMat(mat);
    if (cs) cusparseDestroy(cs);
    cudaFree(d_x);
    cudaFree(d_xn);
    cudaFree(d_ext);
    cudaFree(d_dx);
    cudaFree(d_aty);
    cudaFree(d_xsum);
    cudaFree(d_y);
    cudaFree(d_yn);
    cudaFree(d_dy);
    cudaFree(d_ax);
    cudaFree(d_adx);
    cudaFree(d_ysum);
    cudaFree(d_axc);
    cudaFree(d_axn);
    cudaFree(d_eta);
    cudaFree(d_omega);
    cudaFree(d_accepted);
    cudaFree(d_accept_flag);
    cudaFree(d_scalars);
    cudaFree(d_partials);
    cudaFree(d_cost);
    cudaFree(d_clo);
    cudaFree(d_chi);
    cudaFree(d_rlo);
    cudaFree(d_rhi);
    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_vals);
    cudaFree(d_trowptr);
    cudaFree(d_tcolidx);
    cudaFree(d_tvals);
    cudaFree(d_spmv);
    cudaFree(d_spmv_t);
    cudaFree(d_vals_f);
    cudaFree(d_mix_n);
    cudaFree(d_mix_m);
    cudaFree(d_spmv_f);
  }

  [[nodiscard]] bool alloc_ok() const {
    return d_x && d_xn && d_ext && d_dx && d_aty && d_xsum && d_y && d_yn && d_dy && d_ax &&
           d_adx && d_ysum && d_scalars && d_partials && d_cost && d_clo && d_chi;
  }
};

// ---- cuSPARSE SpMV helpers ----------------------------------------------

// Non-transpose SpMV: d_out_m = A * d_in_n
static bool spmv_nt(GpuState& g, double* d_in_n, double* d_out_m) {
  if (g.m == 0 || g.nnz == 0) {
    if (g.m > 0) cudaMemset(d_out_m, 0, static_cast<std::size_t>(g.m) * sizeof(double));
    return true;
  }
  CS_CHECK(cusparseDnVecSetValues(g.vn, d_in_n));
  CS_CHECK(cusparseDnVecSetValues(g.vm, d_out_m));
  CS_CHECK(cusparseSpMV(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, g.mat, g.vn, &kZero,
                        g.vm, CUDA_R_64F, g.alg, g.d_spmv));
  return true;
}

// Transpose SpMV: d_out_n = A^T * d_in_m. On the explicit A^T when it is held (deterministic
// mode, #478), as a NON-TRANSPOSE product; otherwise cuSPARSE's transpose of A.
static bool spmv_t(GpuState& g, double* d_in_m, double* d_out_n) {
  if (g.m == 0 || g.nnz == 0) {
    if (g.n > 0) cudaMemset(d_out_n, 0, static_cast<std::size_t>(g.n) * sizeof(double));
    return true;
  }
  CS_CHECK(cusparseDnVecSetValues(g.vm, d_in_m));
  CS_CHECK(cusparseDnVecSetValues(g.vn, d_out_n));
  if (g.mat_t != nullptr) {
    CS_CHECK(cusparseSpMV(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, g.mat_t, g.vm, &kZero,
                          g.vn, CUDA_R_64F, g.alg, g.d_spmv_t));  // mat_t's own workspace
    return true;
  }
  CS_CHECK(cusparseSpMV(g.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOne, g.mat, g.vm, &kZero, g.vn,
                        CUDA_R_64F, g.alg, g.d_spmv));  // mat's workspace, as for A x
  return true;
}

// Mixed precision (#982): non-transpose SpMV d_out_m = A * d_in_n with the product itself run
// in float. d_in_n is cast down to g.d_mix_n, the product computed on mat_f into g.d_mix_m,
// and the result cast back up into d_out_m - the same pattern [CH18] and [GSW16] use for a
// low-precision inner solve inside a higher-precision outer loop (see pdhg_refinement.hpp
// for the round-level decision this is meant to feed). Never called when g.mat_f is null
// (mixed_precision is false for this solve).
static bool spmv_nt_mixed(GpuState& g, double* d_in_n, double* d_out_m) {
  if (g.m == 0 || g.nnz == 0) {
    if (g.m > 0) cudaMemset(d_out_m, 0, static_cast<std::size_t>(g.m) * sizeof(double));
    return true;
  }
  const int bn = (g.n + kBlockSize - 1) / kBlockSize;
  const int bm = (g.m + kBlockSize - 1) / kBlockSize;
  k_cast_d2f<<<bn, kBlockSize>>>(d_in_n, g.d_mix_n, g.n);
  if (!launch_ok()) return false;
  CS_CHECK(cusparseDnVecSetValues(g.vn_f, g.d_mix_n));
  CS_CHECK(cusparseDnVecSetValues(g.vm_f, g.d_mix_m));
  CS_CHECK(cusparseSpMV(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOneF, g.mat_f, g.vn_f,
                        &kZeroF, g.vm_f, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, g.d_spmv_f));
  k_cast_f2d<<<bm, kBlockSize>>>(g.d_mix_m, d_out_m, g.m);
  if (!launch_ok()) return false;
  return true;
}

// Mixed precision (#982): transpose SpMV d_out_n = A^T * d_in_m, float product on mat_f (the
// explicit-A^T determinism path of #478 is not covered - mixed_precision and deterministic
// are mutually exclusive, enforced in solve_pdhg_gpu).
static bool spmv_t_mixed(GpuState& g, double* d_in_m, double* d_out_n) {
  if (g.m == 0 || g.nnz == 0) {
    if (g.n > 0) cudaMemset(d_out_n, 0, static_cast<std::size_t>(g.n) * sizeof(double));
    return true;
  }
  const int bn = (g.n + kBlockSize - 1) / kBlockSize;
  const int bm = (g.m + kBlockSize - 1) / kBlockSize;
  k_cast_d2f<<<bm, kBlockSize>>>(d_in_m, g.d_mix_m, g.m);
  if (!launch_ok()) return false;
  CS_CHECK(cusparseDnVecSetValues(g.vm_f, g.d_mix_m));
  CS_CHECK(cusparseDnVecSetValues(g.vn_f, g.d_mix_n));
  CS_CHECK(cusparseSpMV(g.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOneF, g.mat_f, g.vm_f, &kZeroF,
                        g.vn_f, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, g.d_spmv_f));
  k_cast_f2d<<<bn, kBlockSize>>>(g.d_mix_n, d_out_n, g.n);
  if (!launch_ok()) return false;
  return true;
}

using pdhg::euclidean_norm;
using pdhg::evaluate;
using pdhg::Problem;
using pdhg::Residuals;

}  // anonymous namespace

// ---- Main solver --------------------------------------------------------

Solution solve_pdhg_gpu(const Model& model, const Options& options, Logger& logger,
                        SolveControl* control, const pdhg::PdhgWarmStart* warm_start) {
  std::string device_desc;
  if (!device_available(&device_desc)) {
    logger.warning("GPU PDHG: no CUDA device ({}); falling back to CPU solver", device_desc);
    return pdhg::solve_pdhg(model, options, logger, control, warm_start);
  }

  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "pdhg-cuda";

  const std::string validation = model.validate();
  if (!validation.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = validation;
    return solution;
  }

  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  const double sense = model.sense_multiplier();
  const auto n = static_cast<std::size_t>(cols);
  const auto m = static_cast<std::size_t>(rows);
  const int ni = static_cast<int>(n), mi = static_cast<int>(m);

  // ---- Unscaled minimise-space problem ------------------------------------
  Problem prob;
  prob.model = &model;
  prob.cost.resize(n);
  for (Index j = 0; j < cols; ++j)
    prob.cost[static_cast<std::size_t>(j)] =
        sense * model.col_cost[static_cast<std::size_t>(j)];
  prob.cost_norm = euclidean_norm(prob.cost);
  {
    double bsq = 0.0;
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const double b = is_finite_bound(model.row_lower[u])
                           ? model.row_lower[u]
                           : (is_finite_bound(model.row_upper[u]) ? model.row_upper[u] : 0.0);
      bsq += b * b;
    }
    prob.bound_norm = std::sqrt(bsq);
  }

  // ---- Preconditioning (CPU, one-off) ------------------------------------
  constexpr int kRuizIter = 10;
  constexpr int kPowerIter = 30;
  const Scaling scaling = build_scaling(model, prob.cost, kRuizIter);
  const double spectral_norm = estimate_spectral_norm(
      scaling.matrix, kPowerIter, static_cast<unsigned>(options.get_int("random_seed")) + 1u);

  // ---- Build CSR on CPU via CsrView --------------------------------------
  const CsrView csr(scaling.matrix);
  const int nnz = static_cast<int>(csr.values().size());

  // ---- Solver parameters -------------------------------------------------
  const double tolerance = options.get_double("pdhg_tolerance");
  const ResourceLimits limits(options, logger);
  const Count iteration_limit =
      limits.iteration_limit() < 0 ? 1000000 : static_cast<Count>(limits.iteration_limit());
  const bool use_restarts = options.get_bool("pdhg_restart");
  const bool stop_at_request = options.get_bool("pdhg_stop_at_request");
  // pdhg_two_matvec (#479): the default "cpu" keeps three products on the device, where the
  // A/B on main 58a8374 was mixed (pdhg-two-matvec-58a8374.csv); only "true" takes two here.
  const bool two_matvec = options.get_string("pdhg_two_matvec") == "true";
  bool device_loop = options.get_bool("gpu_on_device_loop");
  // Bit-for-bit repeatable run to run (#383, #478): explicit A^T, CSR_ALG2 for both products.
  const bool deterministic = options.get_bool("deterministic");
  // pdhg_precision (#982): single-precision A x and A^T y, every decision still in double.
  // Scoped out, in this change, from the deterministic path (whose bit-for-bit guarantee is
  // documented by cuSPARSE for the double-precision CSR_ALG2 product only) and from the
  // device loop (#478's CUDA-graph capture, which this change does not touch). Both
  // combinations fall back to the ordinary double run rather than silently do something
  // 982 did not ask for.
  bool mixed_precision = options.get_string("pdhg_precision") == "mixed";
  if (mixed_precision && deterministic) {
    logger.warning(
        "GPU PDHG: pdhg_precision=mixed is not combined with deterministic=true (#982); "
        "running this solve in double precision");
    mixed_precision = false;
  }
  if (mixed_precision && device_loop) {
    logger.warning(
        "GPU PDHG: pdhg_precision=mixed is not wired into the device loop (#478) yet; "
        "overriding gpu_on_device_loop to false for this solve so the per-iteration path "
        "runs instead (#982)");
    device_loop = false;
  }

  logger.info("Solving LP with CUDA restarted PDHG on {}: {} rows, {} columns, {} nonzeros",
              device_desc, rows, cols, model.num_nonzeros());
  logger.info("Scaled matrix entries in [{:.3e}, {:.3e}], estimated ||A||_2 = {:.4e}",
              scaling.min_abs, scaling.max_abs, spectral_norm);
  logger.info("Target relative tolerance {:.1e}, restarts {}", tolerance,
              use_restarts ? "on" : "off");
  // #482: the constant step on a proved bound and the PID primal weight, both off by default
  // and both decided on the host (src/pdhg/pdhg.cpp has the argument for each).
  const bool constant_step = options.get_bool("pdhg_constant_step");
  OperatorNormBound norm_bound;
  if (constant_step) {
    norm_bound = bound_spectral_norm(scaling.matrix, tol::kPdhgConstantStepPowerIterations,
                                     tol::kPdhgConstantStepPowerIterations,
                                     static_cast<unsigned>(options.get_int("random_seed")) + 1u);
    logger.info("Constant step (#482): ||A||_2 in [{:.6e}, {:.6e}] proved, eta = {} / upper",
                norm_bound.lower, norm_bound.upper, tol::kPdhgConstantStepShare);
  }
  const bool use_pid = options.get_bool("pdhg_primal_weight_pid");
  const qp::PidGains pid_gains{options.get_double("pdhg_pid_kp"),
                               options.get_double("pdhg_pid_ki"),
                               options.get_double("pdhg_pid_kd")};
  qp::PidState pid_state;
  // #482 item 3 is CPU only: the device evaluation unscales A x through the row and column
  // multipliers (A x = Dr^-1 Ahat xhat), which the rescaling's fold into the Scaling would
  // break, so this engine does not take it and says so.
  if (options.get_bool("pdhg_bound_objective_rescaling") ||
      options.get_bool("pdhg_initial_weight_from_norms")) {
    logger.warning(
        "pdhg_bound_objective_rescaling and pdhg_initial_weight_from_norms are CPU-engine "
        "options (#482); the CUDA engine runs without them");
  }

  // ---- GPU resource allocation -------------------------------------------
  GpuState g;
  g.n = ni;
  g.m = mi;
  g.nnz = nnz;

  g.d_x = dev_zeros(n);
  g.d_xn = dev_zeros(n);
  g.d_ext = dev_zeros(n);
  g.d_dx = dev_zeros(n);
  g.d_aty = dev_zeros(n);
  g.d_xsum = dev_zeros(n);
  g.d_y = dev_zeros(m);
  g.d_yn = dev_zeros(m);
  g.d_dy = dev_zeros(m);
  g.d_ax = dev_zeros(m);
  g.d_adx = dev_zeros(m);
  if (two_matvec) {
    g.d_axc = dev_zeros(m);
    g.d_axn = dev_zeros(m);
  }
  g.d_ysum = dev_zeros(m);
  g.d_scalars = dev_zeros(3);  // [0]=mv_x, [1]=mv_y, [2]=interaction
  const int part_n = (ni + kBlockSize - 1) / kBlockSize;
  const int part_m = (mi + kBlockSize - 1) / kBlockSize;
  g.d_partials = dev_zeros(static_cast<std::size_t>(std::max(1, part_n + 2 * part_m)));
  g.d_cost = dev_zeros(n);
  g.d_clo = dev_zeros(n);
  g.d_chi = dev_zeros(n);
  if (m > 0) {
    g.d_rlo = dev_zeros(m);
    g.d_rhi = dev_zeros(m);
  }
  g.d_rowptr = dev_int(m + 1);
  g.d_colidx = nnz > 0 ? dev_int(static_cast<std::size_t>(nnz)) : nullptr;
  g.d_vals = nnz > 0 ? dev_zeros(static_cast<std::size_t>(nnz)) : nullptr;

  if (deterministic) {
    g.d_trowptr = dev_int(n + 1);
    g.d_tcolidx = nnz > 0 ? dev_int(static_cast<std::size_t>(nnz)) : nullptr;
    g.d_tvals = nnz > 0 ? dev_zeros(static_cast<std::size_t>(nnz)) : nullptr;
  }

  if (mixed_precision) {
    g.d_vals_f = nnz > 0 ? dev_zeros_f(static_cast<std::size_t>(nnz)) : nullptr;
    g.d_mix_n = dev_zeros_f(n);
    g.d_mix_m = dev_zeros_f(m);
  }

  if (!g.alloc_ok() || !g.d_rowptr || (m > 0 && (!g.d_rlo || !g.d_rhi)) ||
      (nnz > 0 && (!g.d_colidx || !g.d_vals)) ||
      (deterministic && (!g.d_trowptr || (nnz > 0 && (!g.d_tcolidx || !g.d_tvals)))) ||
      (mixed_precision && ((nnz > 0 && !g.d_vals_f) || !g.d_mix_n || (m > 0 && !g.d_mix_m)))) {
    logger.warning("GPU PDHG: device allocation failed; falling back to CPU solver");
    return pdhg::solve_pdhg(model, options, logger, control, warm_start);
  }

  // #913 part 2: usable only when offered and sized for THIS model, as pdhg::solve_pdhg
  // checks it. Needed again below, at the primal weight's seed, so it is function-scoped.
  const bool warm_usable = warm_start != nullptr && !warm_start->empty() &&
                           warm_start->x.size() == static_cast<std::size_t>(cols);

  // ---- Upload matrix and problem data ------------------------------------
  {
    const auto& rs = csr.row_starts();
    const auto& ci = csr.column_indices();
    const auto& cv = csr.values();
    // Index = int32_t matches CUSPARSE_INDEX_32I; both vectors hold plain ints on the wire
    if (!hd_copy(rs.data(), g.d_rowptr, rs.size()) ||
        (nnz > 0 && (!hd_copy(ci.data(), g.d_colidx, ci.size()) ||
                     !hd_copy(cv.data(), g.d_vals, cv.size()))) ||
        !hd_copy(scaling.cost.data(), g.d_cost, n) ||
        !hd_copy(scaling.col_lower.data(), g.d_clo, n) ||
        !hd_copy(scaling.col_upper.data(), g.d_chi, n) ||
        (m > 0 && (!hd_copy(scaling.row_lower.data(), g.d_rlo, m) ||
                   !hd_copy(scaling.row_upper.data(), g.d_rhi, m)))) {
      logger.warning("GPU PDHG: data upload failed; falling back to CPU solver");
      return pdhg::solve_pdhg(model, options, logger, control, warm_start);
    }
    // Mixed precision (#982): the same scaled values, rounded to float once here rather than
    // re-cast from the double copy on every product - a constant cost paid once per solve,
    // not per iteration.
    if (mixed_precision && nnz > 0) {
      std::vector<float> cv_f(cv.size());
      for (std::size_t k = 0; k < cv.size(); ++k) cv_f[k] = static_cast<float>(cv[k]);
      if (!hd_copy(cv_f.data(), g.d_vals_f, cv_f.size())) {
        logger.warning("GPU PDHG: mixed-precision matrix upload failed; falling back to CPU "
                       "solver");
        return pdhg::solve_pdhg(model, options, logger, control, warm_start);
      }
    }
    // A^T in CSR is A in CSC: the scaled matrix's own column starts, row indices and values,
    // uploaded as they are (#478). Index is int32, as for the CSR of A above.
    if (deterministic) {
      const auto& ts = scaling.matrix.column_starts();
      const auto& ti = scaling.matrix.row_indices();
      const auto& tv = scaling.matrix.values();
      if (ts.size() != n + 1 || ti.size() != static_cast<std::size_t>(nnz) ||
          !hd_copy(ts.data(), g.d_trowptr, ts.size()) ||
          (nnz > 0 && (!hd_copy(ti.data(), g.d_tcolidx, ti.size()) ||
                       !hd_copy(tv.data(), g.d_tvals, tv.size())))) {
        logger.warning("GPU PDHG: transpose upload failed; falling back to CPU solver");
        return pdhg::solve_pdhg(model, options, logger, control, warm_start);
      }
    }

    // Initial x: the caller's warm iterate, divided by this solve's own scaling and
    // projected into its own bounds (#913 part 2, same mapping as pdhg::solve_pdhg), when
    // one is offered and sized for this model; otherwise the projection of 0 onto column
    // bounds (the cold start, same as CPU PDHG).
    std::vector<double> x0(n);
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      double v = warm_usable ? warm_start->x[u] / scaling.column[u] : 0.0;
      if (!std::isinf(scaling.col_lower[u]) && v < scaling.col_lower[u])
        v = scaling.col_lower[u];
      if (!std::isinf(scaling.col_upper[u]) && v > scaling.col_upper[u])
        v = scaling.col_upper[u];
      x0[u] = v;
    }
    if (!hd_copy(x0.data(), g.d_x, n)) {
      logger.warning("GPU PDHG: initial iterate upload failed; falling back to CPU solver");
      return pdhg::solve_pdhg(model, options, logger, control, warm_start);
    }
    if (warm_usable && warm_start->y.size() == static_cast<std::size_t>(rows) && m > 0) {
      std::vector<double> y0(m);
      for (Index i = 0; i < rows; ++i) {
        const auto u = static_cast<std::size_t>(i);
        y0[u] = warm_start->y[u] / scaling.row[u];
      }
      if (!hd_copy(y0.data(), g.d_y, m)) {
        logger.warning("GPU PDHG: initial dual iterate upload failed; falling back to CPU solver");
        return pdhg::solve_pdhg(model, options, logger, control, warm_start);
      }
    }
  }

  // ---- cuSPARSE setup ----------------------------------------------------
  if (cusparseCreate(&g.cs) != CUSPARSE_STATUS_SUCCESS) {
    logger.warning("GPU PDHG: cusparseCreate failed; falling back to CPU solver");
    return pdhg::solve_pdhg(model, options, logger, control, warm_start);
  }

  auto cs_failed = [&]() -> Solution {
    logger.warning("GPU PDHG: cuSPARSE setup failed; falling back to CPU solver");
    return pdhg::solve_pdhg(model, options, logger, control, warm_start);
  };

  // Create CSR matrix descriptor.  When nnz=0, d_colidx/d_vals are nullptr;
  // cusparseCreateCsr accepts null pointers when nnz=0.
  if (cusparseCreateCsr(&g.mat, static_cast<int64_t>(mi), static_cast<int64_t>(ni),
                        static_cast<int64_t>(nnz), g.d_rowptr, g.d_colidx, g.d_vals,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
    return cs_failed();
  if (deterministic) {
    g.alg = CUSPARSE_SPMV_CSR_ALG2;
    if (cusparseCreateCsr(&g.mat_t, static_cast<int64_t>(ni), static_cast<int64_t>(mi),
                          static_cast<int64_t>(nnz), g.d_trowptr, g.d_tcolidx, g.d_tvals,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
  }

  // Create dense vector descriptors; dimension is pinned at construction, data updated later.
  // vn: n-element (input to NT SpMV, output of T SpMV)
  // vm: m-element (output of NT SpMV, input to T SpMV)
  void* vn_init = (g.d_x ? g.d_x : g.d_scalars);  // non-null placeholder
  void* vm_init = (g.d_y ? g.d_y : g.d_scalars);
  if (cusparseCreateDnVec(&g.vn, static_cast<int64_t>(ni), vn_init, CUDA_R_64F) !=
      CUSPARSE_STATUS_SUCCESS)
    return cs_failed();
  if (mi > 0) {
    if (cusparseCreateDnVec(&g.vm, static_cast<int64_t>(mi), vm_init, CUDA_R_64F) !=
        CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
  }

  // Query SpMV buffer sizes and allocate one workspace per descriptor (see GpuState::d_spmv).
  if (m > 0 && nnz > 0) {
    std::size_t bytes_nt = 0, bytes_t = 0;
    cusparseDnVecSetValues(g.vn, g.d_ext);
    cusparseDnVecSetValues(g.vm, g.d_ax);
    if (cusparseSpMV_bufferSize(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, g.mat, g.vn,
                                &kZero, g.vm, CUDA_R_64F, g.alg,
                                &bytes_nt) != CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
    cusparseDnVecSetValues(g.vm, g.d_y);
    cusparseDnVecSetValues(g.vn, g.d_aty);
    const cusparseStatus_t sized_t =
        g.mat_t != nullptr
            ? cusparseSpMV_bufferSize(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, g.mat_t,
                                      g.vm, &kZero, g.vn, CUDA_R_64F, g.alg, &bytes_t)
            : cusparseSpMV_bufferSize(g.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOne, g.mat, g.vm,
                                      &kZero, g.vn, CUDA_R_64F, g.alg, &bytes_t);
    if (sized_t != CUSPARSE_STATUS_SUCCESS) return cs_failed();
    // Without A^T held, `mat` serves both products and its workspace is sized for both.
    const std::size_t bytes_mat = g.mat_t != nullptr ? bytes_nt : std::max(bytes_nt, bytes_t);
    if (bytes_mat > 0 && cudaMalloc(&g.d_spmv, bytes_mat) != cudaSuccess) return cs_failed();
    if (g.mat_t != nullptr && bytes_t > 0 && cudaMalloc(&g.d_spmv_t, bytes_t) != cudaSuccess)
      return cs_failed();
  }

  // Mixed precision (#982): the float matrix descriptor and dense vectors the two
  // single-precision products (spmv_nt_mixed/spmv_t_mixed) use, built the same way as the
  // double descriptors above but on g.d_vals_f and CUDA_R_32F. Always the default algorithm
  // (CSR_ALG2's bit-for-bit guarantee is documented for CUDA_R_64F only); mixed_precision and
  // deterministic are mutually exclusive, enforced earlier in this function.
  if (mixed_precision) {
    if (cusparseCreateCsr(&g.mat_f, static_cast<int64_t>(mi), static_cast<int64_t>(ni),
                          static_cast<int64_t>(nnz), g.d_rowptr, g.d_colidx, g.d_vals_f,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
    void* mix_n_init = g.d_mix_n ? g.d_mix_n : vn_init;
    void* mix_m_init = g.d_mix_m ? g.d_mix_m : vm_init;
    if (cusparseCreateDnVec(&g.vn_f, static_cast<int64_t>(ni), mix_n_init, CUDA_R_32F) !=
        CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
    if (mi > 0 && cusparseCreateDnVec(&g.vm_f, static_cast<int64_t>(mi), mix_m_init,
                                      CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
    if (m > 0 && nnz > 0) {
      std::size_t bytes_nt_f = 0, bytes_t_f = 0;
      cusparseDnVecSetValues(g.vn_f, g.d_mix_n);
      cusparseDnVecSetValues(g.vm_f, g.d_mix_m);
      if (cusparseSpMV_bufferSize(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOneF, g.mat_f,
                                  g.vn_f, &kZeroF, g.vm_f, CUDA_R_32F,
                                  CUSPARSE_SPMV_ALG_DEFAULT,
                                  &bytes_nt_f) != CUSPARSE_STATUS_SUCCESS)
        return cs_failed();
      if (cusparseSpMV_bufferSize(g.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOneF, g.mat_f, g.vm_f,
                                  &kZeroF, g.vn_f, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                  &bytes_t_f) != CUSPARSE_STATUS_SUCCESS)
        return cs_failed();
      const std::size_t bytes_mat_f = std::max(bytes_nt_f, bytes_t_f);
      if (bytes_mat_f > 0 && cudaMalloc(&g.d_spmv_f, bytes_mat_f) != cudaSuccess)
        return cs_failed();
    }
  }

  // ---- Main iteration loop -----------------------------------------------
  const int bn = (ni + kBlockSize - 1) / kBlockSize;
  const int bm = (mi + kBlockSize - 1) / kBlockSize;

  // Mixed precision (#982): the two sparse products the per-iteration path below takes,
  // dispatched to the single-precision pair when pdhg_precision=mixed and to the ordinary
  // double pair otherwise. Every other call site in this function - the convergence
  // evaluation (evaluate_on_device, above) and the host evaluation path it falls back to -
  // stays on spmv_nt/spmv_t directly, in double, by construction: #982 point 3 ("decisions
  // in double") is satisfied by NOT routing those call sites through this dispatch, not by a
  // runtime check.
  auto product_nt = [&](double* in, double* out) {
    return mixed_precision ? spmv_nt_mixed(g, in, out) : spmv_nt(g, in, out);
  };
  auto product_t = [&](double* in, double* out) {
    return mixed_precision ? spmv_t_mixed(g, in, out) : spmv_t(g, in, out);
  };

  // CPU-side vectors for convergence evaluation
  std::vector<double> h_x(n), h_y(m), h_xsum(n), h_ysum(m);
  std::vector<double> x_unscaled(n), y_unscaled(m);
  std::vector<double> activity(m), reduced_costs(n);
  // Restart reference point (scaled): the same initial iterate x0 was set to above - the
  // warm one when #913 part 2 offered one, the cold projection of 0 otherwise.
  std::vector<double> x_restart(n, 0.0), y_restart(m, 0.0);
  {
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      double v = warm_usable ? warm_start->x[u] / scaling.column[u] : 0.0;
      if (!std::isinf(scaling.col_lower[u]) && v < scaling.col_lower[u])
        v = scaling.col_lower[u];
      if (!std::isinf(scaling.col_upper[u]) && v > scaling.col_upper[u])
        v = scaling.col_upper[u];
      x_restart[u] = v;
    }
    if (warm_usable && warm_start->y.size() == static_cast<std::size_t>(rows)) {
      for (Index i = 0; i < rows; ++i) {
        const auto u = static_cast<std::size_t>(i);
        y_restart[u] = warm_start->y[u] / scaling.row[u];
      }
    }
  }

  auto unscale = [&](const std::vector<double>& xs, const std::vector<double>& ys) {
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      x_unscaled[u] = xs[u] * scaling.column[u];
    }
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      y_unscaled[u] = ys[u] * scaling.row[u];
    }
  };

  double eta = spectral_norm > 0.0 ? 1.0 / spectral_norm : 1.0;
  if (constant_step) {
    eta = norm_bound.upper > 0.0 ? tol::kPdhgConstantStepShare / norm_bound.upper : 1.0;
  }
  // #913 part 2: the caller's primal weight carried over, when offered and positive and
  // finite; every restart below still re-derives omega from the iterates, so a bad carried
  // value only costs the first step.
  double omega = (warm_usable && std::isfinite(warm_start->omega) && warm_start->omega > 0.0)
                     ? warm_start->omega
                     : 1.0;
  Count iteration = 0, restarts = 0, last_restart = 0, averaged = 0;
  double restart_kkt = std::numeric_limits<double>::infinity();
  // Mixed precision (#982 point 4): the refinement round this run is on is read off the
  // restart cadence already above - each restart's double-precision KKT residual against the
  // previous restart's is exactly the round history pdhg_refinement.hpp's decision rule wants
  // (compute the residual in double, run more work at the lower precision, repeat). When a
  // restart stops shrinking that residual usefully, this run falls back to full double
  // precision for its remaining iterations rather than keep paying for single precision
  // products that are not buying accuracy.
  pdhg::RefinementState refine_state;
  Count refinement_fallback_at_restart = -1;
  // First crossings of the relative KKT error (#486), recorded as the CPU engine does and
  // on the same measure (pdhg::evaluate), so a CUDA row in a runner CSV is not "never
  // reached" beside a verified optimum.
  double kkt_seconds[3] = {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::quiet_NaN()};
  Count kkt_iterations[3] = {-1, -1, -1};

  Residuals best;
  best.primal = best.dual = best.gap = std::numeric_limits<double>::infinity();
  std::vector<double> best_x(n, 0.0), best_y(m, 0.0);

  bool converged = false, gpu_error = false, logged_table = false;
  // The evaluation's share of the run (#478 item 3), timed on either path so the log shows
  // what it costs: on the host it copies x, y and both running sums off the device and
  // evaluates them on the unscaled model; on the device only the sums come back.
  double evaluation_seconds = 0.0;
  Count evaluations = 0;
  StopController stop(control, timer, limits);
  SolveStatus stop_status = SolveStatus::kIterationLimit;
  const pdhg::IterateTraceHook trace = pdhg::iterate_trace_for_testing();
  std::vector<double> trace_x(trace.callback != nullptr ? n : 0);
  std::vector<double> trace_y(trace.callback != nullptr ? m : 0);
  std::vector<double> trace_axc(trace.callback != nullptr ? m : 0);
  std::vector<double> trace_ax(trace.callback != nullptr ? m : 0);

  // Two-mat-vec (#479): the cache starts as A x0, filled once the SpMV buffers exist.
  if (two_matvec) {
    if ((mi > 0 && (g.d_axc == nullptr || g.d_axn == nullptr)) || !product_nt(g.d_x, g.d_axc)) {
      gpu_error = true;  // the CPU fallback below takes over
    }
  }
  // THE LOOP ON THE DEVICE (#478). One iteration captured as a CUDA graph, replayed
  // kPdhgDeviceLoopBlock times per host synchronisation; the adaptive step rule and the
  // commit of an accepted step run on the device. If the capture fails the per-iteration
  // path below runs instead, and the log says so.
  std::unique_ptr<DeviceLoop> loop;
  Count accepted_at_restart = 0;
  const double eta_ceil_device = 1.0e3 / std::max(spectral_norm, 1e-12);
  if (device_loop && !gpu_error) {
    bool ready = cudaMalloc(&g.d_eta, sizeof(double)) == cudaSuccess &&
                 cudaMalloc(&g.d_omega, sizeof(double)) == cudaSuccess &&
                 cudaMalloc(&g.d_accepted, sizeof(long long)) == cudaSuccess &&
                 cudaMalloc(&g.d_accept_flag, sizeof(int)) == cudaSuccess;
    const long long zero_count = 0;
    ready = ready && hd_copy(&eta, g.d_eta, 1) && hd_copy(&omega, g.d_omega, 1) &&
            hd_copy(&zero_count, g.d_accepted, 1);
    if (ready) {
      DeviceLoopBuffers b;
      b.x = g.d_x; b.xn = g.d_xn; b.ext = g.d_ext; b.dx = g.d_dx; b.aty = g.d_aty;
      b.xsum = g.d_xsum; b.y = g.d_y; b.yn = g.d_yn; b.dy = g.d_dy; b.ax = g.d_ax;
      b.adx = g.d_adx; b.ysum = g.d_ysum; b.axc = g.d_axc; b.axn = g.d_axn;
      b.cost = g.d_cost; b.col_lo = g.d_clo; b.col_hi = g.d_chi; b.row_lo = g.d_rlo;
      b.row_hi = g.d_rhi; b.partials = g.d_partials; b.eta = g.d_eta; b.omega = g.d_omega;
      b.accepted = g.d_accepted; b.accept_flag = g.d_accept_flag;
      b.eta_ceil = eta_ceil_device; b.constant_step = constant_step;
      b.n = ni; b.m = mi; b.nnz = nnz;
      b.two_matvec = two_matvec; b.cusparse = g.cs; b.matrix = g.mat; b.vec_n = g.vn;
      b.vec_m = g.vm; b.spmv_buffer = g.d_spmv; b.spmv_buffer_t = g.d_spmv_t;
      b.matrix_t = g.mat_t; b.spmv_alg = g.alg;
      loop = std::make_unique<DeviceLoop>();
      ready = loop->init(b, static_cast<int>(tol::kPdhgDeviceLoopBlock));
    }
    if (!ready) {
      loop.reset();
      cudaGetLastError();  // clear a failed capture's error before the host path runs
      device_loop = false;
      logger.warning("GPU PDHG: the device loop could not be captured; running the "
                     "per-iteration path");
    } else {
      logger.info("GPU PDHG: device loop on, {} iterations per host synchronisation (#478)",
                  tol::kPdhgDeviceLoopBlock);
    }
  }
  if (deterministic && !gpu_error) {
    logger.info("GPU PDHG: deterministic, fixed-order reductions and both products on CSR "
                "with CUSPARSE_SPMV_CSR_ALG2 on an explicit A^T (#478)");
  }
  if (mixed_precision && !gpu_error) {
    logger.info("GPU PDHG: pdhg_precision=mixed, A x and A^T y run in single precision; "
                "residuals, restarts and the step rule stay in double (#982). UNVERIFIED ON "
                "HARDWARE as of this change - no device micro-benchmark has been run; see "
                "bench/runners/gpu_precision_microbench.py and #982");
  }
  // THE EVALUATION ON THE DEVICE (#478 item 3). On the stream the products run on: the
  // device loop's while it holds the cuSPARSE handle, else the legacy default stream.
  std::unique_ptr<eval::DeviceEvaluator> evaluator;
  const cudaStream_t eval_stream = loop ? loop->stream() : cudaStream_t{};
  if (options.get_bool("gpu_device_evaluation") && !gpu_error) {
    evaluator = std::make_unique<eval::DeviceEvaluator>();
    if (!evaluator->init(prob, scaling, 0, mi, true, g.d_x)) {
      evaluator.reset();
      cudaGetLastError();
      logger.warning("GPU PDHG: the device evaluation could not be set up; evaluating on "
                     "the host");
    } else {
      logger.info("GPU PDHG: convergence evaluated on the device, scalars to the host (#478)");
    }
  }
  // Residuals of the current iterate (point 0) and the average (point 1) from the device,
  // and the current iterate's scaled distances to the restart point.
  auto evaluate_on_device = [&](bool with_average, Residuals* cur, Residuals* avg,
                                double* restart_dx, double* restart_dy) -> bool {
    eval::DeviceEvaluator& ev = *evaluator;
    const cudaStream_t s = eval_stream;
    if (!spmv_nt(g, g.d_x, ev.ax()) || !ev.rows(s, 0, g.d_y, ev.ax(), true) ||
        !spmv_t(g, g.d_y, ev.aty()) || !ev.columns(s, 0, g.d_x, ev.aty(), true))
      return false;
    if (with_average &&
        (!ev.average(s, g.d_xsum, g.d_ysum, static_cast<double>(averaged)) ||
         !spmv_nt(g, ev.x_average(), ev.ax()) || !ev.rows(s, 1, ev.y_average(), ev.ax(), false) ||
         !spmv_t(g, ev.y_average(), ev.aty()) ||
         !ev.columns(s, 1, ev.x_average(), ev.aty(), false)))
      return false;
    if (!ev.finish(s, with_average ? 2 : 1) || cudaStreamSynchronize(s) != cudaSuccess)
      return false;
    const eval::RowSums r0 = eval::row_sums(ev.host_sums(), 0);
    const eval::ColumnSums c0 = eval::column_sums(ev.host_sums(), 0);
    *cur = eval::assemble(prob, r0, c0);
    *restart_dx = std::sqrt(c0.restart_sq);
    *restart_dy = std::sqrt(r0.restart_sq);
    if (with_average) {
      *avg = eval::assemble(prob, eval::row_sums(ev.host_sums(), 1),
                            eval::column_sums(ev.host_sums(), 1));
    }
    return true;
  };
  while (!gpu_error) {
    if (iteration >= iteration_limit) break;
    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kLp;
              p.iterations = iteration;
              p.objective = best.primal;
              p.best_bound = sense * best.dual_objective + model.objective_offset;
              return p;
            },
            &stop_status))
      break;

    if (device_loop) {
      if (!loop->run_block()) {
        gpu_error = true;
        break;
      }
      long long accepted_now = 0;
      if (cudaMemcpy(&accepted_now, g.d_accepted, sizeof(long long),
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          !dh_copy(g.d_eta, &eta, 1)) {
        gpu_error = true;
        break;
      }
      const Count before = iteration;
      iteration = static_cast<Count>(accepted_now);
      averaged = iteration - accepted_at_restart;
      // Evaluate when the accepted count crosses a multiple of the interval: a block can
      // cover several accepted steps, so the host sees the count on a stride.
      if (iteration == 0 || iteration / kEvaluationInterval == before / kEvaluationInterval) {
        continue;
      }
    } else {
    const double tau = eta / omega;
    const double sigma = eta * omega;

    // 1. A^T y  [cuPDLP §3, transpose SpMV]
    if (!product_t(g.d_y, g.d_aty)) {
      gpu_error = true;
      break;
    }

    // 2. Primal update + movement_x reduction  [CP11 Alg.1, PDLP §3.1]
    if (ni > 0)
      k_primal_fused<<<bn, kBlockSize>>>(g.d_x, g.d_aty, g.d_cost, g.d_clo, g.d_chi, g.d_xn,
                                         g.d_ext, g.d_dx, g.d_partials, tau, omega, ni);
    if (!launch_ok()) {
      gpu_error = true;
      break;
    }

    // 3. A * extrapolated  [CP11 Alg.1 dual step]. With two_matvec (#479): A x_{k+1} once,
    // and A xbar and A dx derived from it and the cached A x_k.
    if (two_matvec) {
      if (!product_nt(g.d_xn, g.d_axn)) {
        gpu_error = true;
        break;
      }
      if (mi > 0) k_derive_products<<<bm, kBlockSize>>>(g.d_axn, g.d_axc, g.d_ax, g.d_adx, mi);
      if (!launch_ok()) {
        gpu_error = true;
        break;
      }
    } else if (!product_nt(g.d_ext, g.d_ax)) {
      gpu_error = true;
      break;
    }

    // 4. Dual update + movement_y reduction  [CP11 Alg.1, PDLP §3.1]
    if (mi > 0)
      k_dual_fused<<<bm, kBlockSize>>>(g.d_y, g.d_ax, g.d_rlo, g.d_rhi, g.d_yn, g.d_dy,
                                       g.d_partials + bn, sigma, omega, mi);
    if (!launch_ok()) {
      gpu_error = true;
      break;
    }

    // 5. A * dx  [PDLP §3.1 interaction term]; already derived with two_matvec.
    if (!two_matvec && !product_nt(g.d_dx, g.d_adx)) {
      gpu_error = true;
      break;
    }

    // 6. Interaction, then all three scalars from the block partials in a fixed order
    // (#478), downloaded in one transfer  [PDLP §3.1]. Steps 2 and 4 wrote the movement
    // partials; every slot is rewritten each iteration, so nothing needs zeroing.
    if (mi > 0) {
      k_interaction_fused<<<bm, kBlockSize>>>(g.d_dy, g.d_adx, g.d_partials + bn + bm, mi);
      if (!launch_ok()) {
        gpu_error = true;
        break;
      }
    }
    k_finish_scalars<<<1, kBlockSize>>>(g.d_partials, bn, bm, g.d_scalars);
    if (!launch_ok()) {
      gpu_error = true;
      break;
    }
    double h_scalars[3] = {};
    if (!dh_copy(g.d_scalars, h_scalars, 3)) {
      gpu_error = true;
      break;
    }
    const double movement = h_scalars[0] + h_scalars[1];
    const double interaction = std::fabs(h_scalars[2]);

    // 7. Adaptive step size  [PDLP §3.1]
    const bool no_info = (interaction <= 0.0);
    const double limit =
        no_info ? std::numeric_limits<double>::infinity() : movement / interaction;
    // exponent floor of 2 avoids the shrink=0 collapse on the first iteration  [PDLP §3.1]
    const double exp = static_cast<double>(std::max<Count>(2, iteration + 1));
    const double shrink = 1.0 - std::pow(exp, -0.3);
    const double grow = 1.0 + std::pow(exp, -0.6);
    const double proposed = std::min(shrink * limit, grow * eta);

    if (constant_step || eta <= limit) {  // #482: a constant step is admissible by proof
      // Accept: swap iterates via pointer swap
      std::swap(g.d_x, g.d_xn);
      std::swap(g.d_y, g.d_yn);
      if (two_matvec) std::swap(g.d_axc, g.d_axn);  // A x_{k+1} is A x of the new iterate
      // Accumulate running sums
      if (ni > 0) k_accum_n<<<bn, kBlockSize>>>(g.d_x, g.d_xsum, ni);
      if (mi > 0) k_accum_m<<<bm, kBlockSize>>>(g.d_y, g.d_ysum, mi);
      if (!launch_ok()) {
        gpu_error = true;
        break;
      }
      ++averaged;
      ++iteration;
      if (trace.callback != nullptr) {  // a test seam, null outside tests (#479)
        // With two_matvec, d_ax is free until the next step derives into it, so A x_k is
        // recomputed there for the test to hold against the cache.
        const bool cached = two_matvec && mi > 0;
        if (!dh_copy(g.d_x, trace_x.data(), n) || !dh_copy(g.d_y, trace_y.data(), m) ||
            (cached && (!spmv_nt(g, g.d_x, g.d_ax) || !dh_copy(g.d_axc, trace_axc.data(), m) ||
                        !dh_copy(g.d_ax, trace_ax.data(), m)))) {
          gpu_error = true;
          break;
        }
        trace.callback(trace.context, iteration, trace_x.data(), n, trace_y.data(), m,
                       cached ? trace_axc.data() : nullptr, cached ? trace_ax.data() : nullptr);
      }
    }
    const double eta_ceil = 1.0e3 / std::max(spectral_norm, 1e-12);
    if (!no_info && !constant_step) eta = std::clamp(proposed, 1e-12, eta_ceil);

    // 8. Convergence and restart check (CPU, every kEvaluationInterval accepted steps)
    if (iteration == 0) continue;
    if (iteration % kEvaluationInterval != 0 && !no_info) continue;
    }  // per-iteration path

    const double evaluation_start = timer.elapsed_seconds();
    ++evaluations;
    // The current iterate and the running average [PDLP §4.3], on the device or, as the
    // reference, on the host.
    const bool have_average = averaged > 0;
    Residuals cur, avg;
    double restart_dx = 0.0, restart_dy = 0.0;  // device path: formed with point 0
    std::vector<double> cur_x, cur_y, avg_x, avg_y;  // host path only
    if (evaluator) {
      if (!evaluate_on_device(have_average, &cur, &avg, &restart_dx, &restart_dy)) {
        gpu_error = true;
        break;
      }
    } else {
      if (!dh_copy(g.d_x, h_x.data(), n) || !dh_copy(g.d_y, h_y.data(), m)) {
        gpu_error = true;
        break;
      }
      unscale(h_x, h_y);
      cur_x = x_unscaled;
      cur_y = y_unscaled;
      cur = evaluate(prob, cur_x, cur_y, activity, reduced_costs);
      if (have_average) {
        if (!dh_copy(g.d_xsum, h_xsum.data(), n) || !dh_copy(g.d_ysum, h_ysum.data(), m)) {
          gpu_error = true;
          break;
        }
        const double cnt = static_cast<double>(averaged);
        std::vector<double> xav(n), yav(m);
        for (std::size_t j = 0; j < n; ++j) xav[j] = h_xsum[j] / cnt;
        for (std::size_t i = 0; i < m; ++i) yav[i] = h_ysum[i] / cnt;
        unscale(xav, yav);
        avg_x = x_unscaled;
        avg_y = y_unscaled;
        avg = evaluate(prob, avg_x, avg_y, activity, reduced_costs);
      }
    }
    const bool use_average = have_average && avg.worst() < cur.worst();
    // The chosen point becomes the best so far: a copy on the device, or of the host vectors.
    auto keep_chosen = [&]() -> bool {
      if (evaluator) {
        return evaluator->set_best(eval_stream, use_average ? evaluator->x_average() : g.d_x,
                                   use_average ? evaluator->y_average() : g.d_y);
      }
      best_x = use_average ? avg_x : cur_x;
      best_y = use_average ? avg_y : cur_y;
      return true;
    };

    evaluation_seconds += timer.elapsed_seconds() - evaluation_start;
    const Residuals& better = use_average ? avg : cur;
    for (int level = 0; level < 3; ++level) {
      if (kkt_iterations[level] < 0 && better.worst() <= pdhg::kKktCrossingLevels[level]) {
        kkt_iterations[level] = iteration;
        kkt_seconds[level] = timer.elapsed_seconds();
      }
    }
    if (better.worst() < best.worst()) {
      best = better;
      if (!keep_chosen()) {
        gpu_error = true;
        break;
      }
    }

    if (!logged_table) {
      logger.begin_iteration_table();
      logged_table = true;
    }
    logger.iteration(iteration, sense * better.primal_objective + model.objective_offset,
                     better.primal, better.dual, timer.elapsed_seconds());

    const bool stop_here =
        better.meets_request(tolerance) && (stop_at_request || better.meets_project_standard());
    if (stop_here) {
      best = better;
      gpu_error = !keep_chosen();
      converged = !gpu_error;
      break;
    }

    // Restart logic  [PDLP §4.3]
    if (use_restarts) {
      const double kkt = better.worst();
      const Count since = iteration - last_restart;
      const bool sufficient = kkt <= 0.2 * restart_kkt;
      const bool artificial =
          since >= std::max<Count>(kEvaluationInterval,
                                   static_cast<Count>(0.36 * static_cast<double>(iteration)));

      if (sufficient || artificial) {
        // Primal weight update towards observed ratio of dual/primal movement  [PDLP §3.2]
        double dxn = restart_dx, dyn = restart_dy;
        if (!evaluator) {
          std::vector<double> dx_rs(n), dy_rs(m);
          for (std::size_t j = 0; j < n; ++j) dx_rs[j] = h_x[j] - x_restart[j];
          for (std::size_t i = 0; i < m; ++i) dy_rs[i] = h_y[i] - y_restart[i];
          dxn = euclidean_norm(dx_rs);
          dyn = euclidean_norm(dy_rs);
        }
        if (use_pid) {
          omega = qp::pid_primal_weight(omega, dxn, dyn, pid_gains, &pid_state);  // #482
        } else if (dxn > 1e-12 && dyn > 1e-12) {
          constexpr double theta = 0.5;
          omega = std::exp(theta * std::log(dyn / dxn) + (1.0 - theta) * std::log(omega));
          omega = std::clamp(omega, 1e-6, 1e6);
        }

        cudaMemset(g.d_xsum, 0, n * sizeof(double));
        cudaMemset(g.d_ysum, 0, m * sizeof(double));
        if (device_loop) {
          // The device reads omega in its kernels; the average restarts from here.
          if (!hd_copy(&omega, g.d_omega, 1)) {
            gpu_error = true;
            break;
          }
          accepted_at_restart = iteration;
        }
        averaged = 0;
        if (evaluator) {
          if (!evaluator->set_restart(eval_stream, g.d_x, g.d_y)) {
            gpu_error = true;
            break;
          }
        } else {
          x_restart = h_x;
          y_restart = h_y;
        }
        if (mixed_precision && std::isfinite(restart_kkt)) {
          refine_state.previous_residual = restart_kkt;
          refine_state.current_residual = kkt;
          const pdhg::RefinementAction action =
              pdhg::decide_refinement_action(refine_state, tolerance, iteration_limit);
          ++refine_state.rounds_run;
          if (action == pdhg::RefinementAction::kFallBackToDouble) {
            mixed_precision = false;
            refinement_fallback_at_restart = restarts;
            logger.info(
                "GPU PDHG: single precision stalled at restart {} (KKT {:.3e} -> {:.3e}, "
                "shrink {:.3f}); falling back to double precision for the rest of this "
                "solve (#982)",
                restarts, refine_state.previous_residual, refine_state.current_residual,
                refine_state.current_residual / refine_state.previous_residual);
          }
        }
        restart_kkt = kkt;
        last_restart = iteration;
        ++restarts;
        // The cache is A x carried forward by derivation; at each restart it is recomputed
        // from x so rounding cannot accumulate across restart periods (#479).
        if (two_matvec && !product_nt(g.d_x, g.d_axc)) {
          gpu_error = true;
          break;
        }
        logger.verbose("restart {} at iteration {}: KKT {:.3e}, primal weight {:.3e}", restarts,
                       iteration, kkt, omega);
      }
    }
  }  // end while

  // With the evaluation on the device the best point stayed there, scaled: bring it back and
  // unscale it as the host path does (x = Dc xhat, y = Dr yhat, the same multiplications).
  if (evaluator && !gpu_error) {
    std::vector<double> xs(n), ys(m);
    if (!evaluator->download_best(eval_stream, xs.data(), ys.data())) {
      gpu_error = true;
    } else {
      unscale(xs, ys);
      best_x = x_unscaled;
      best_y = y_unscaled;
    }
  }

  if (gpu_error) {
    // The CPU engine takes over on what the budget has left (#289): the seconds already
    // spent on the device are not handed out a second time.
    Options remaining = options;
    if (limits.has_time_limit()) {
      remaining.set_double("time_limit", limits.remaining_seconds(timer.elapsed_seconds()));
    }
    logger.warning(
        "GPU PDHG: CUDA error during solve after {:.2f}s; falling back to the CPU "
        "solver on the remaining budget",
        timer.elapsed_seconds());
    return pdhg::solve_pdhg(model, remaining, logger, control, warm_start);
  }

  // ---- Extract solution --------------------------------------------------
  for (Index j = 0; j < cols; ++j)
    solution.col_value[static_cast<std::size_t>(j)] =
        best_x.empty() ? 0.0 : best_x[static_cast<std::size_t>(j)];

  const Residuals final_r = evaluate(prob, best_x, best_y, activity, reduced_costs);
  for (Index j = 0; j < cols; ++j)
    solution.col_dual[static_cast<std::size_t>(j)] =
        sense * reduced_costs[static_cast<std::size_t>(j)];
  for (Index i = 0; i < rows; ++i)
    solution.row_dual[static_cast<std::size_t>(i)] =
        sense * (-best_y[static_cast<std::size_t>(i)]);

  solution.iterations = iteration;
  solution.kkt_1e4_seconds = kkt_seconds[0];
  solution.kkt_1e6_seconds = kkt_seconds[1];
  solution.kkt_1e8_seconds = kkt_seconds[2];
  solution.kkt_1e4_iterations = kkt_iterations[0];
  solution.kkt_1e6_iterations = kkt_iterations[1];
  solution.kkt_1e8_iterations = kkt_iterations[2];
  solution.solve_seconds = timer.elapsed_seconds();

  const bool verifiable = converged && final_r.meets_project_standard();
  if (verifiable) {
    solution.status = SolveStatus::kOptimal;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA PDHG converged after %lld iterations and %lld restarts; absolute primal %.3e, "
        "dual %.3e, relative gap %.3e",
        (long long)iteration, (long long)restarts, final_r.absolute_primal, final_r.absolute_dual,
        final_r.gap_as_verified);
    solution.message = buf;
  } else if (converged) {
    solution.status = SolveStatus::kFeasible;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA PDHG met requested tolerance %.1e after %lld iterations but NOT project standard "
        "(primal %.3e vs %.1e, dual %.3e vs %.1e, gap %.3e vs %.1e)",
        tolerance, (long long)iteration, final_r.absolute_primal, tol::kPrimalFeasibility,
        final_r.absolute_dual, tol::kDualFeasibility, final_r.gap_as_verified,
        tol::kDualityGap);
    solution.message = buf;
  } else {
    solution.status =
        (stop_status == SolveStatus::kTimeLimit || stop_status == SolveStatus::kInterrupted)
            ? stop_status
            : SolveStatus::kIterationLimit;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA PDHG stopped at relative primal %.3e, dual %.3e, gap %.3e after %lld "
        "iterations and %lld restarts (target %.1e)",
        final_r.primal, final_r.dual, final_r.gap, (long long)iteration, (long long)restarts, tolerance);
    solution.message = buf;
  }

  if (verifiable) {
    solution.dual_bound = sense * final_r.dual_objective + model.objective_offset;
  } else {
    solution.dual_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model);

  logger.info("");
  logger.info("Status: {}   objective {:.10e}   iterations {}   restarts {}   time {:.3f}s",
              to_string(solution.status), solution.objective, solution.iterations, restarts,
              solution.solve_seconds);
  logger.info("Evaluation on the {}: {} evaluations, {:.3f}s of {:.3f}s",
              evaluator ? "device" : "host", evaluations, evaluation_seconds,
              solution.solve_seconds);
  logger.info("Relative residuals: primal {:.3e}, dual {:.3e}, gap {:.3e}", final_r.primal,
              final_r.dual, final_r.gap);
  if (refinement_fallback_at_restart >= 0) {
    logger.info("GPU PDHG: pdhg_precision=mixed fell back to double precision at restart {} "
                "of {} (#982)",
                refinement_fallback_at_restart, restarts);
  }
  if (!solution.message.empty()) logger.info("{}", solution.message);
  return solution;
}

}  // namespace sankhya::gpu
