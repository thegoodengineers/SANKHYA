// SPDX-License-Identifier: Apache-2.0
// SANKHYA - per-device state and kernels of the row-partitioned multi-GPU PDHG (#295).
// References and the reproducibility argument: pdhg_multi_gpu_device.hpp.

#include "pdhg_multi_gpu_device.hpp"

#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace sankhya::gpu::multi {
namespace {

const double kOne = 1.0;
const double kZero = 0.0;

#define MG_CUDA(expr)                        \
  do {                                       \
    if ((expr) != cudaSuccess) return false; \
  } while (0)

#define MG_CS(expr)                                      \
  do {                                                   \
    if ((expr) != CUSPARSE_STATUS_SUCCESS) return false; \
  } while (0)

bool launch_ok() { return cudaPeekAtLastError() == cudaSuccess; }

// Primal step [CP11 eq. 4, PDLP sec. 3]: x' = proj_[l,u](x - tau (c + A^T y)), the
// extrapolation 2x' - x, dx = x' - x, and this block's share of 0.5 omega ||dx||^2.
__global__ void k_primal(const double* __restrict__ x, const double* __restrict__ at_y,
                         const double* __restrict__ cost, const double* __restrict__ col_lo,
                         const double* __restrict__ col_hi, double* __restrict__ x_next,
                         double* __restrict__ extrapolated, double* __restrict__ dx,
                         double* __restrict__ block_sums, double tau, double omega, int n) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  double mv = 0.0;
  if (j < n) {
    double xnj = x[j] - tau * (cost[j] + at_y[j]);
    if (!isinf(col_lo[j]) && xnj < col_lo[j]) xnj = col_lo[j];
    if (!isinf(col_hi[j]) && xnj > col_hi[j]) xnj = col_hi[j];
    x_next[j] = xnj;
    const double dxj = xnj - x[j];
    extrapolated[j] = 2.0 * xnj - x[j];
    dx[j] = dxj;
    mv = 0.5 * omega * dxj * dxj;
  }
  const double bsum = BlockReduce(temp).Sum(mv);
  if (threadIdx.x == 0) block_sums[blockIdx.x] = bsum;
}

// Dual step by the Moreau identity [CP11 sec. 4]: v = y + sigma A x_ext,
// y' = v - sigma proj_[rl,ru](v / sigma), and this block's share of 0.5 ||dy||^2 / omega.
__global__ void k_dual(const double* __restrict__ y, const double* __restrict__ a_x,
                       const double* __restrict__ row_lo, const double* __restrict__ row_hi,
                       double* __restrict__ y_next, double* __restrict__ dy,
                       double* __restrict__ block_sums, double sigma, double omega, int m) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double mv = 0.0;
  if (i < m) {
    const double v = y[i] + sigma * a_x[i];
    double pv = v / sigma;
    if (!isinf(row_lo[i]) && pv < row_lo[i]) pv = row_lo[i];
    if (!isinf(row_hi[i]) && pv > row_hi[i]) pv = row_hi[i];
    const double yni = v - sigma * pv;
    y_next[i] = yni;
    const double dyi = yni - y[i];
    dy[i] = dyi;
    mv = 0.5 * dyi * dyi / omega;
  }
  const double bsum = BlockReduce(temp).Sum(mv);
  if (threadIdx.x == 0) block_sums[blockIdx.x] = bsum;
}

// This block's share of dy^T A dx, the interaction term of the PDLP step-size rule.
__global__ void k_interaction(const double* __restrict__ dy, const double* __restrict__ adx,
                              double* __restrict__ block_sums, int m) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double v = (i < m) ? dy[i] * adx[i] : 0.0;
  const double bsum = BlockReduce(temp).Sum(v);
  if (threadIdx.x == 0) block_sums[blockIdx.x] = bsum;
}

// One block per slot: thread t sums block sums t, t + 256, ... in increasing order, then a
// BlockReduce (a fixed tree for a fixed block size) combines the threads. The result depends
// only on the block sums, never on scheduling.
__global__ void k_finalize(const double* __restrict__ block_sums, int stride, int count0,
                           int count1, int count2, double* __restrict__ out) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int slot = blockIdx.x;
  const int count = slot == 0 ? count0 : (slot == 1 ? count1 : count2);
  const double* src = block_sums + static_cast<std::size_t>(slot) * stride;
  double s = 0.0;
  for (int b = threadIdx.x; b < count; b += blockDim.x) s += src[b];
  const double total = BlockReduce(temp).Sum(s);
  if (threadIdx.x == 0) out[slot] = total;
}

__global__ void k_accumulate(const double* __restrict__ v, double* __restrict__ sum, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) sum[j] += v[j];
}

double* zeros(std::size_t count) {
  if (count == 0) return nullptr;
  double* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(double)) != cudaSuccess) return nullptr;
  if (cudaMemset(p, 0, count * sizeof(double)) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

template <typename T>
bool upload(const T* host, T** dev, std::size_t count) {
  if (count == 0) return true;
  if (cudaMalloc(dev, count * sizeof(T)) != cudaSuccess) return false;
  return cudaMemcpy(*dev, host, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

int blocks_for(int count) { return (count + kBlockSize - 1) / kBlockSize; }

}  // namespace

DeviceState::~DeviceState() {
  if (device_id < 0) return;
  cudaSetDevice(device_id);
  if (stream) cudaStreamSynchronize(stream);
  if (vm) cusparseDestroyDnVec(vm);
  if (vn) cusparseDestroyDnVec(vn);
  if (mat_t) cusparseDestroySpMat(mat_t);
  if (mat) cusparseDestroySpMat(mat);
  if (cs) cusparseDestroy(cs);
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_xn), static_cast<void*>(d_ext),
                  static_cast<void*>(d_dx), static_cast<void*>(d_aty),
                  static_cast<void*>(d_xsum), static_cast<void*>(d_cost),
                  static_cast<void*>(d_clo), static_cast<void*>(d_chi),
                  static_cast<void*>(d_partial), static_cast<void*>(d_recv),
                  static_cast<void*>(d_y), static_cast<void*>(d_yn), static_cast<void*>(d_dy),
                  static_cast<void*>(d_ax), static_cast<void*>(d_adx),
                  static_cast<void*>(d_ysum), static_cast<void*>(d_rlo),
                  static_cast<void*>(d_rhi), static_cast<void*>(d_blocks),
                  static_cast<void*>(d_scalars), static_cast<void*>(d_rp),
                  static_cast<void*>(d_ci), static_cast<void*>(d_trp),
                  static_cast<void*>(d_tci), static_cast<void*>(d_v), static_cast<void*>(d_tv),
                  d_spmv})
    if (p) cudaFree(p);
  if (stream) cudaStreamDestroy(stream);
}

bool setup_device(DeviceState& d, const CsrView& full_csr, const Scaling& scaling,
                  const std::vector<double>& x0_scaled, int row_start, int row_end,
                  int n_cols) {
  MG_CUDA(cudaSetDevice(d.device_id));
  // A blocking stream: legacy default-stream work (none is issued on this path) would still
  // order against it.
  MG_CUDA(cudaStreamCreate(&d.stream));

  const int lm = row_end - row_start;
  d.n = n_cols;
  d.local_m = lm;
  d.row_start = row_start;
  d.blocks_n = blocks_for(n_cols);
  d.blocks_m = blocks_for(lm);
  d.block_stride = std::max(1, std::max(d.blocks_n, d.blocks_m));

  const auto n = static_cast<std::size_t>(n_cols);
  for (double** p : {&d.d_x, &d.d_xn, &d.d_ext, &d.d_dx, &d.d_aty, &d.d_xsum, &d.d_partial}) {
    *p = zeros(n);
    if (n > 0 && *p == nullptr) return false;
  }
  d.d_recv = zeros(n * static_cast<std::size_t>(d.num_slots));
  d.d_blocks = zeros(3 * static_cast<std::size_t>(d.block_stride));
  d.d_scalars = zeros(3);
  if ((n > 0 && d.d_recv == nullptr) || !d.d_blocks || !d.d_scalars) return false;
  if (!upload(scaling.cost.data(), &d.d_cost, n) ||
      !upload(scaling.col_lower.data(), &d.d_clo, n) ||
      !upload(scaling.col_upper.data(), &d.d_chi, n))
    return false;
  if (n > 0 && cudaMemcpy(d.d_x, x0_scaled.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice) != cudaSuccess)
    return false;

  const auto lms = static_cast<std::size_t>(lm);
  for (double** p : {&d.d_y, &d.d_yn, &d.d_dy, &d.d_ax, &d.d_adx, &d.d_ysum}) {
    *p = zeros(lms);
    if (lm > 0 && *p == nullptr) return false;
  }
  if (!upload(scaling.row_lower.data() + row_start, &d.d_rlo, lms) ||
      !upload(scaling.row_upper.data() + row_start, &d.d_rhi, lms))
    return false;

  // A_k: rows [row_start, row_end) of the scaled CSR, re-based to local row numbers.
  const auto& rp = full_csr.row_starts();
  const auto& ci = full_csr.column_indices();
  const auto& cv = full_csr.values();
  const int nz0 = rp[static_cast<std::size_t>(row_start)];
  const int nz1 = rp[static_cast<std::size_t>(row_end)];
  d.local_nnz = nz1 - nz0;
  const auto lnnz = static_cast<std::size_t>(d.local_nnz);
  std::vector<int> local_rp(lms + 1);
  for (int i = 0; i <= lm; ++i)
    local_rp[static_cast<std::size_t>(i)] = rp[static_cast<std::size_t>(row_start + i)] - nz0;

  // A_k^T as CSR (n x local_m) by a counting transpose: within each column the local rows
  // come out in increasing order, so the stored order, and the SpMV's sum, is fixed.
  std::vector<int> t_rp(n + 1, 0), t_ci(lnnz);
  std::vector<double> t_v(lnnz);
  for (std::size_t e = 0; e < lnnz; ++e) ++t_rp[static_cast<std::size_t>(ci[nz0 + e]) + 1];
  for (std::size_t j = 0; j < n; ++j) t_rp[j + 1] += t_rp[j];
  {
    std::vector<int> next(t_rp.begin(), t_rp.end() - 1);
    for (int i = 0; i < lm; ++i)
      for (int e = local_rp[static_cast<std::size_t>(i)];
           e < local_rp[static_cast<std::size_t>(i) + 1]; ++e) {
        const auto col = static_cast<std::size_t>(ci[static_cast<std::size_t>(nz0 + e)]);
        const auto at = static_cast<std::size_t>(next[col]++);
        t_ci[at] = i;
        t_v[at] = cv[static_cast<std::size_t>(nz0 + e)];
      }
  }

  if (lm == 0 || d.local_nnz == 0) return true;  // spmv_* write zeros; no cuSPARSE needed

  if (!upload(local_rp.data(), &d.d_rp, lms + 1) || !upload(ci.data() + nz0, &d.d_ci, lnnz) ||
      !upload(cv.data() + nz0, &d.d_v, lnnz) || !upload(t_rp.data(), &d.d_trp, n + 1) ||
      !upload(t_ci.data(), &d.d_tci, lnnz) || !upload(t_v.data(), &d.d_tv, lnnz))
    return false;

  MG_CS(cusparseCreate(&d.cs));
  MG_CS(cusparseSetStream(d.cs, d.stream));
  MG_CS(cusparseCreateCsr(&d.mat, lm, n_cols, d.local_nnz, d.d_rp, d.d_ci, d.d_v,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F));
  MG_CS(cusparseCreateCsr(&d.mat_t, n_cols, lm, d.local_nnz, d.d_trp, d.d_tci, d.d_tv,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F));
  MG_CS(cusparseCreateDnVec(&d.vn, n_cols, d.d_ext, CUDA_R_64F));
  MG_CS(cusparseCreateDnVec(&d.vm, lm, d.d_ax, CUDA_R_64F));

  std::size_t bytes_a = 0, bytes_t = 0;
  MG_CS(cusparseSpMV_bufferSize(d.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, d.mat, d.vn,
                                &kZero, d.vm, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bytes_a));
  MG_CS(cusparseDnVecSetValues(d.vm, d.d_y));
  MG_CS(cusparseDnVecSetValues(d.vn, d.d_partial));
  MG_CS(cusparseSpMV_bufferSize(d.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, d.mat_t, d.vm,
                                &kZero, d.vn, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bytes_t));
  // One workspace serves both products: they are serialized on the card's one stream.
  const std::size_t bytes = std::max(bytes_a, bytes_t);
  if (bytes > 0) MG_CUDA(cudaMalloc(&d.d_spmv, bytes));
  return true;
}

bool spmv_aty(DeviceState& d) {
  if (d.local_m == 0 || d.local_nnz == 0) {
    // Nothing here ever changes: the partial stays the zero it was allocated as.
    return true;
  }
  MG_CS(cusparseDnVecSetValues(d.vm, d.d_y));
  MG_CS(cusparseDnVecSetValues(d.vn, d.d_partial));
  MG_CS(cusparseSpMV(d.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, d.mat_t, d.vm, &kZero, d.vn,
                     CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, d.d_spmv));
  return true;
}

bool spmv_ax(DeviceState& d, double* in_n, double* out_m) {
  if (d.local_m == 0) return true;
  if (d.local_nnz == 0) {
    MG_CUDA(cudaMemsetAsync(out_m, 0, static_cast<std::size_t>(d.local_m) * sizeof(double),
                            d.stream));
    return true;
  }
  MG_CS(cusparseDnVecSetValues(d.vn, in_n));
  MG_CS(cusparseDnVecSetValues(d.vm, out_m));
  MG_CS(cusparseSpMV(d.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, d.mat, d.vn, &kZero, d.vm,
                     CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, d.d_spmv));
  return true;
}

bool launch_primal(DeviceState& d, double tau, double omega) {
  if (d.n == 0) return true;
  k_primal<<<d.blocks_n, kBlockSize, 0, d.stream>>>(d.d_x, d.d_aty, d.d_cost, d.d_clo, d.d_chi,
                                                     d.d_xn, d.d_ext, d.d_dx, d.d_blocks, tau,
                                                     omega, d.n);
  return launch_ok();
}

bool launch_dual(DeviceState& d, double sigma, double omega) {
  if (d.local_m == 0) return true;
  k_dual<<<d.blocks_m, kBlockSize, 0, d.stream>>>(d.d_y, d.d_ax, d.d_rlo, d.d_rhi, d.d_yn,
                                                   d.d_dy, d.d_blocks + d.block_stride, sigma,
                                                   omega, d.local_m);
  return launch_ok();
}

bool launch_interaction_and_scalars(DeviceState& d, double* host_scalars) {
  if (d.local_m > 0) {
    k_interaction<<<d.blocks_m, kBlockSize, 0, d.stream>>>(
        d.d_dy, d.d_adx, d.d_blocks + 2 * static_cast<std::size_t>(d.block_stride), d.local_m);
    if (!launch_ok()) return false;
  }
  k_finalize<<<3, kBlockSize, 0, d.stream>>>(d.d_blocks, d.block_stride, d.blocks_n, d.blocks_m,
                                             d.blocks_m, d.d_scalars);
  if (!launch_ok()) return false;
  MG_CUDA(cudaMemcpyAsync(host_scalars, d.d_scalars, 3 * sizeof(double), cudaMemcpyDeviceToHost,
                          d.stream));
  return true;
}

bool launch_accumulate(DeviceState& d) {
  if (d.n > 0) k_accumulate<<<d.blocks_n, kBlockSize, 0, d.stream>>>(d.d_x, d.d_xsum, d.n);
  if (d.local_m > 0)
    k_accumulate<<<d.blocks_m, kBlockSize, 0, d.stream>>>(d.d_y, d.d_ysum, d.local_m);
  return launch_ok();
}

bool download(DeviceState& d, const double* src, double* dst, std::size_t count) {
  if (count == 0) return true;
  MG_CUDA(cudaMemcpyAsync(dst, src, count * sizeof(double), cudaMemcpyDeviceToHost, d.stream));
  MG_CUDA(cudaStreamSynchronize(d.stream));
  return true;
}

}  // namespace sankhya::gpu::multi
