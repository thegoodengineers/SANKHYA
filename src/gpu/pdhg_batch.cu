// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Batched GPU PDHG for B&B node bounding and strong branching (#520).
//
// References (written from the papers; no solver source consulted):
//   [BPDHG] Applegate et al., "Practical Large-Scale LP via PDHG", NeurIPS 2021.
//           Section 3 extended to n x K / m x K block iterates: one cuSPARSE SpMM
//           replaces K separate SpMV calls; per-column step sizes from pdhg_gpu.cu.
//   [SBB]   Huang et al., "Batched First-Order Strong Branching on the GPU",
//           arXiv:2601.21990. Use-2 (strong branching): 2K children of K candidates
//           solved as one batch; each child's safe bound is its score.
//
// Design:
//   - K node LPs share the constraint matrix A (uploaded once per call, CSR).
//   - Primal iterates: X ∈ R^{n x K} (n = num cols, K = batch size).
//   - Dual iterates:   Y ∈ R^{m x K} (m = num rows).
//   - Per iteration:
//       1. Y_new_k = clip(Y_k + σ*(A*X_k - b_k), bounds)  for all k simultaneously
//          via cusparseSpMM(A, X, Y_scratch)
//       2. X_new_k = clip(X_k - τ*(A^T*Y_new_k - c), [lb_k, ub_k]) for all k
//          via cusparseSpMM(A^T, Y_new, X_scratch)
//       3. Every kCheckEvery iterations: extract dual iterate, compute safe bound per k
//          (Neumaier-Shcherbina #519), mark pruned when bound > incumbent.
//   - Fixed step sizes σ = τ = 1/||A||_F per column (simple; the per-column schedule
//     from pdhg_gpu.cu is a future improvement).
//   - Stops when all K nodes are either pruned or the iteration limit is reached.

#include "pdhg_batch.hpp"
#include "device.hpp"

#include "../core/safe_bound.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>

namespace sankhya::gpu {
namespace {

static const double kOne = 1.0;
static const double kZero = 0.0;
static const double kNegOne = -1.0;

/// Row-major-indexed dense matrix on device: m rows, n cols, stored column-major
/// (cuSPARSE CUSPARSE_ORDER_COL for the dense operand).
struct DeviceMat {
  double* data = nullptr;
  int rows = 0;
  int cols = 0;
  bool alloc(int r, int c) {
    rows = r; cols = c;
    return cudaMalloc(reinterpret_cast<void**>(&data),
                      (r == 0 || c == 0 ? 1 : static_cast<std::size_t>(r) * static_cast<std::size_t>(c)) *
                          sizeof(double)) == cudaSuccess;
  }
  ~DeviceMat() { if (data) cudaFree(data); }
};

/// Simple wrapper for a scalar device array.
template<typename T>
bool upload_vec(T** dst, const T* src, std::size_t n) {
  if (cudaMalloc(reinterpret_cast<void**>(dst), (n == 0 ? 1 : n) * sizeof(T)) != cudaSuccess) return false;
  if (n > 0 && cudaMemcpy(*dst, src, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return false;
  return true;
}

/// Clip values in [lo, hi] elementwise on GPU (one kernel).
__global__ void k_clip(double* x, const double* lo, const double* hi, int n, int K) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * K) return;
  const int j = idx % n;  // column index within one LP
  x[idx] = fmax(lo[j + static_cast<int>(blockIdx.y) * n],
                fmin(hi[j + static_cast<int>(blockIdx.y) * n], x[idx]));
}

/// Add σ*(Ax - b) to dual in-place: Y += σ*(AX - B) then clip to row sense.
/// After SpMM computes AX into Y_scratch (m x K), we add –b per row and clip to dual cone.
__global__ void k_dual_update(double* Y, const double* AX, const double* row_lo,
                               const double* row_hi, int m, int K, double sigma) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y;
  if (i >= m || k >= K) return;
  const int idx = i + k * m;
  // Y_new = Y + sigma * (A*X_k - b) then project onto dual cone.
  // For constraint l <= Ax <= u, dual y is free; the update is y += sigma*(Ax - proj).
  // Here we just do the gradient step; the dual cone for LP is R (no projection needed
  // unless we use a penalty form — simple PDHG for LP has unconstrained dual).
  Y[idx] = Y[idx] + sigma * AX[idx];
  // Clip to [0, +inf) for <= rows (row_lo = -inf), (-inf, 0] for >= rows (row_hi = +inf),
  // and free for equality rows (both finite and equal).
  if (!isinf(row_hi[i]) && isinf(row_lo[i])) {
    // <= constraint: y >= 0 (standard form dual)
    Y[idx] = fmax(0.0, Y[idx]);
  } else if (!isinf(row_lo[i]) && isinf(row_hi[i])) {
    // >= constraint: y <= 0
    Y[idx] = fmin(0.0, Y[idx]);
  }
  // equality: y free, no clipping
}

/// X -= tau*(A^T*Y - c) then clip to [lb_k, ub_k].
__global__ void k_primal_update(double* X, const double* ATY, const double* col_cost,
                                 const double* lb_block, const double* ub_block,
                                 int n, int K, double tau) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y;
  if (j >= n || k >= K) return;
  const int idx = j + k * n;
  X[idx] = X[idx] - tau * (ATY[idx] - col_cost[j]);
  X[idx] = fmax(lb_block[idx], fmin(ub_block[idx], X[idx]));
}

/// Subtract b_k from AX: AX[:,k] -= b for each k (b is the same across k in standard PDHG).
__global__ void k_subtract_rhs(double* AX, const double* rhs, int m, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y;
  if (i >= m || k >= K) return;
  // rhs is the row centre: (row_lo + row_hi)/2 for ranged rows, the bound for one-sided.
  AX[i + k * m] -= rhs[i];
}

constexpr int kBlock = 256;
constexpr int kCheckEvery = 80;
constexpr double kDualTolerance = 1e-4;

}  // namespace

std::vector<BatchNodeResult>
solve_batch_nodes(const Model& model,
                  int K,
                  const std::vector<double>& col_lb,
                  const std::vector<double>& col_ub,
                  double incumbent,
                  const Options& options) {
  if (!options.get_bool("gpu_batch_nodes")) return {};
  if (!device_available(nullptr)) return {};
  if (K <= 0) return {};

  const int m = static_cast<int>(model.num_rows());
  const int n = static_cast<int>(model.num_cols());
  const int nnz = static_cast<int>(model.matrix.num_entries());

  // Build CSR in int (cuSPARSE wants int32 row pointers for csrSetPointers).
  std::vector<int> row_ptr(static_cast<std::size_t>(m + 1));
  std::vector<int> col_idx(static_cast<std::size_t>(nnz));
  std::vector<double> val(static_cast<std::size_t>(nnz));
  {
    const CsrView csr(model.matrix);
    for (int i = 0; i <= m; ++i) row_ptr[static_cast<std::size_t>(i)] = static_cast<int>(csr.row_start()[static_cast<std::size_t>(i)]);
    for (int p = 0; p < nnz; ++p) {
      col_idx[static_cast<std::size_t>(p)] = static_cast<int>(csr.column()[static_cast<std::size_t>(p)]);
      val[static_cast<std::size_t>(p)] = csr.value()[static_cast<std::size_t>(p)];
    }
  }

  // Fixed step size: σ = τ = 1 / (||A||_F + 1).
  double norm_sq = 0.0;
  for (double v : val) norm_sq += v * v;
  const double step = 1.0 / (std::sqrt(norm_sq) + 1.0);
  const double sigma = step;
  const double tau = step;

  // RHS: for each row, centre of [row_lo, row_hi] (used in dual update).
  std::vector<double> rhs(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    const double lo = model.row_lower[static_cast<std::size_t>(i)];
    const double hi = model.row_upper[static_cast<std::size_t>(i)];
    if (std::isinf(lo)) rhs[static_cast<std::size_t>(i)] = hi;
    else if (std::isinf(hi)) rhs[static_cast<std::size_t>(i)] = lo;
    else rhs[static_cast<std::size_t>(i)] = 0.5 * (lo + hi);
  }

  // Device allocations.
  int* d_row_ptr = nullptr; int* d_col_idx = nullptr; double* d_val = nullptr;
  double* d_cost = nullptr; double* d_row_lo = nullptr; double* d_row_hi = nullptr;
  double* d_rhs = nullptr;
  if (!upload_vec(&d_row_ptr, row_ptr.data(), row_ptr.size()) ||
      !upload_vec(&d_col_idx, col_idx.data(), col_idx.size()) ||
      !upload_vec(&d_val, val.data(), val.size()) ||
      !upload_vec(&d_cost, model.col_cost.data(), static_cast<std::size_t>(n)) ||
      !upload_vec(&d_row_lo, model.row_lower.data(), static_cast<std::size_t>(m)) ||
      !upload_vec(&d_row_hi, model.row_upper.data(), static_cast<std::size_t>(m)) ||
      !upload_vec(&d_rhs, rhs.data(), static_cast<std::size_t>(m))) {
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
    cudaFree(d_cost); cudaFree(d_row_lo); cudaFree(d_row_hi); cudaFree(d_rhs);
    return {};
  }

  // Block matrices: column-major, each block stores K LP solutions side by side.
  DeviceMat X, Y, AX_scratch, ATY_scratch;
  DeviceMat d_lb, d_ub;
  if (!X.alloc(n, K) || !Y.alloc(m, K) ||
      !AX_scratch.alloc(m, K) || !ATY_scratch.alloc(n, K) ||
      !d_lb.alloc(n, K) || !d_ub.alloc(n, K)) {
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
    cudaFree(d_cost); cudaFree(d_row_lo); cudaFree(d_row_hi); cudaFree(d_rhs);
    return {};
  }

  // Initialize X to midpoint of column bounds, Y to zero.
  cudaMemset(Y.data, 0, static_cast<std::size_t>(m) * static_cast<std::size_t>(K) * sizeof(double));
  {
    std::vector<double> x_init(static_cast<std::size_t>(n) * static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < n; ++j) {
        const double lo = col_lb[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
        const double hi = col_ub[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
        const double mid = (std::isinf(lo) ? 0.0 : (std::isinf(hi) ? lo : 0.5*(lo+hi)));
        x_init[static_cast<std::size_t>(j) + static_cast<std::size_t>(k) * static_cast<std::size_t>(n)] = mid;
      }
    }
    if (cudaMemcpy(X.data, x_init.data(), x_init.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_lb.data, col_lb.data(), col_lb.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_ub.data, col_ub.data(), col_ub.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
      cudaFree(d_cost); cudaFree(d_row_lo); cudaFree(d_row_hi); cudaFree(d_rhs);
      return {};
    }
  }

  // cuSPARSE setup.
  cusparseHandle_t cs{};
  if (cusparseCreate(&cs) != CUSPARSE_STATUS_SUCCESS) {
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
    cudaFree(d_cost); cudaFree(d_row_lo); cudaFree(d_row_hi); cudaFree(d_rhs);
    return {};
  }
  cusparseSpMatDescr_t mat{};
  cusparseCreateCsr(&mat, static_cast<int64_t>(m), static_cast<int64_t>(n),
                    static_cast<int64_t>(nnz), d_row_ptr, d_col_idx, d_val,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseDnMatDescr_t dm_X{}, dm_AX{}, dm_Y{}, dm_ATY{};
  cusparseCreateDnMat(&dm_X,   static_cast<int64_t>(n), static_cast<int64_t>(K),
                      static_cast<int64_t>(n), X.data,         CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&dm_AX,  static_cast<int64_t>(m), static_cast<int64_t>(K),
                      static_cast<int64_t>(m), AX_scratch.data, CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&dm_Y,   static_cast<int64_t>(m), static_cast<int64_t>(K),
                      static_cast<int64_t>(m), Y.data,         CUDA_R_64F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&dm_ATY, static_cast<int64_t>(n), static_cast<int64_t>(K),
                      static_cast<int64_t>(n), ATY_scratch.data, CUDA_R_64F, CUSPARSE_ORDER_COL);

  // SpMM workspace.
  std::size_t ws1 = 0, ws2 = 0;
  cusparseSpMM_bufferSize(cs, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &kOne, mat, dm_X, &kZero, dm_AX, CUDA_R_64F,
                          CUSPARSE_SPMM_ALG_DEFAULT, &ws1);
  cusparseSpMM_bufferSize(cs, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &kOne, mat, dm_Y, &kZero, dm_ATY, CUDA_R_64F,
                          CUSPARSE_SPMM_ALG_DEFAULT, &ws2);
  void* ws = nullptr;
  const std::size_t ws_size = std::max(ws1, ws2);
  if (ws_size > 0) cudaMalloc(&ws, ws_size);

  // Build SafeBoundProblem for dual bound extraction (Neumaier-Shcherbina #519).
  std::vector<double> cost_min(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    cost_min[static_cast<std::size_t>(j)] =
        model.col_cost[static_cast<std::size_t>(j)] *
        (model.sense == ObjSense::kMinimize ? 1.0 : -1.0);
  }
  const core::SafeBoundProblem sbp{
      &model.matrix,
      std::span<const double>{cost_min},
      std::span<const double>{model.row_lower},
      std::span<const double>{model.row_upper},
      std::span<const double>{model.col_lower},
      std::span<const double>{model.col_upper},
  };

  const int max_iters = options.get_int("gpu_batch_max_iter");
  const int n_threads = kBlock;

  std::vector<BatchNodeResult> results(static_cast<std::size_t>(K),
                                       BatchNodeResult{-std::numeric_limits<double>::infinity(), 0, false});
  std::vector<bool> active(static_cast<std::size_t>(K), true);
  int active_count = K;

  // Host copy of dual iterate for safe-bound evaluation.
  std::vector<double> Y_host(static_cast<std::size_t>(m) * static_cast<std::size_t>(K));

  for (int iter = 0; iter < max_iters && active_count > 0; ++iter) {
    // Step 1: AX = A * X  (SpMM: m x K = m x n × n x K)
    cusparseSpMM(cs, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kOne, mat, dm_X, &kZero, dm_AX, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, ws);
    // Step 2: AX -= rhs broadcast (Y_new = Y + sigma * (AX - rhs))
    {
      const dim3 grid((m + n_threads - 1) / n_threads, K);
      k_subtract_rhs<<<grid, n_threads>>>(AX_scratch.data, d_rhs, m, K);
    }
    // Step 3: dual update: Y += sigma * AX_scratch, then project to dual cone.
    {
      const dim3 grid((m + n_threads - 1) / n_threads, K);
      k_dual_update<<<grid, n_threads>>>(Y.data, AX_scratch.data, d_row_lo, d_row_hi, m, K, sigma);
    }
    // Step 4: ATY = A^T * Y  (SpMM: n x K = n x m × m x K)
    // Update dm_Y to point to current Y (values already updated in-place).
    cusparseDnMatSetValues(dm_Y, Y.data);
    cusparseSpMM(cs, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kOne, mat, dm_Y, &kZero, dm_ATY, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, ws);
    // Step 5: primal update X -= tau * (ATY - c), clip to bounds.
    {
      const dim3 grid((n + n_threads - 1) / n_threads, K);
      k_primal_update<<<grid, n_threads>>>(X.data, ATY_scratch.data, d_cost,
                                           d_lb.data, d_ub.data, n, K, tau);
    }

    // Every kCheckEvery iterations: check safe bounds and mark pruned nodes.
    if ((iter + 1) % kCheckEvery == 0 || iter + 1 == max_iters) {
      if (cudaMemcpy(Y_host.data(), Y.data,
                     static_cast<std::size_t>(m) * static_cast<std::size_t>(K) * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess) {
        for (int k = 0; k < K; ++k) {
          if (!active[static_cast<std::size_t>(k)]) continue;
          const std::span<const double> y_k(Y_host.data() + k * m,
                                            static_cast<std::size_t>(m));
          const core::SafeBound sb = core::safe_dual_bound(sbp, y_k);
          results[static_cast<std::size_t>(k)].dual_bound = sb.value;
          results[static_cast<std::size_t>(k)].iterations = iter + 1;
          if (std::isfinite(sb.value) && sb.value > incumbent + tol::kPrimalFeasibility) {
            results[static_cast<std::size_t>(k)].pruned = true;
            active[static_cast<std::size_t>(k)] = false;
            --active_count;
          }
        }
      }
    }
  }

  // Cleanup.
  if (ws) cudaFree(ws);
  cusparseDestroyDnMat(dm_X); cusparseDestroyDnMat(dm_AX);
  cusparseDestroyDnMat(dm_Y); cusparseDestroyDnMat(dm_ATY);
  cusparseDestroySpMat(mat);
  cusparseDestroy(cs);
  cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
  cudaFree(d_cost); cudaFree(d_row_lo); cudaFree(d_row_hi); cudaFree(d_rhs);

  return results;
}

}  // namespace sankhya::gpu
