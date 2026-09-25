// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU PDHG on-device iteration loop via CUDA Graphs (#478). See pdhg_graph.hpp for
// the design and the references.
//
// One iteration, as captured (the same arithmetic as pdhg_gpu.cu's per-iteration path; only
// where the step-size decision is made and how an accepted step is committed differ):
//   1. aty = A^T y
//   2. x_next, extrapolated, dx, movement_x       with tau = eta / omega read on the device
//   3. ax = A extrapolated                        (two-mat-vec: ax_next = A x_next, then
//                                                  ax = 2 ax_next - ax_cached, adx = the
//                                                  difference, #479)
//   4. y_next, dy, movement_y                     with sigma = eta * omega
//   5. adx = A dx                                 (not with two-mat-vec)
//   6. interaction = sum dy * adx
//   7. the three scalars from the block partials in a fixed order (pdhg_reduce.cuh), then
//      the adaptive step rule [PDLP section 3.1] in thread 0: accept or reject, the next eta,
//      the accepted count
//   8. commit, when accepted: x = x_next, y = y_next, the running sums, the product cache

#include "pdhg_graph.hpp"

#include <algorithm>

#include "pdhg_reduce.cuh"

namespace sankhya::gpu {
namespace {

constexpr int kThreads = 256;

__global__ void k_primal_dev(const double* __restrict__ x, const double* __restrict__ at_y,
                             const double* __restrict__ cost, const double* __restrict__ col_lo,
                             const double* __restrict__ col_hi, double* __restrict__ x_next,
                             double* __restrict__ extrapolated, double* __restrict__ dx,
                             double* partials, const double* __restrict__ eta,
                             const double* __restrict__ omega, int n) {
  const double w = *omega;
  const double tau = *eta / w;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  double mv_thread = 0.0;
  if (j < n) {
    double xnj = x[j] - tau * (cost[j] + at_y[j]);
    if (!isinf(col_lo[j]) && xnj < col_lo[j]) xnj = col_lo[j];
    if (!isinf(col_hi[j]) && xnj > col_hi[j]) xnj = col_hi[j];
    x_next[j] = xnj;
    const double dxj = xnj - x[j];
    extrapolated[j] = 2.0 * xnj - x[j];
    dx[j] = dxj;
    mv_thread = 0.5 * w * dxj * dxj;
  }
  detail::write_block_partial<kThreads>(mv_thread, partials);
}

__global__ void k_dual_dev(const double* __restrict__ y, const double* __restrict__ a_x,
                           const double* __restrict__ row_lo, const double* __restrict__ row_hi,
                           double* __restrict__ y_next, double* __restrict__ dy, double* partials,
                           const double* __restrict__ eta, const double* __restrict__ omega,
                           int m) {
  const double w = *omega;
  const double sigma = *eta * w;
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
    mv_thread = 0.5 * dyi * dyi / w;
  }
  detail::write_block_partial<kThreads>(mv_thread, partials);
}

__global__ void k_derive_dev(const double* __restrict__ ax_next,
                             const double* __restrict__ ax_cached, double* __restrict__ ax_bar,
                             double* __restrict__ adx, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const double next = ax_next[i];
  const double cached = ax_cached[i];
  ax_bar[i] = 2.0 * next - cached;
  adx[i] = next - cached;
}

__global__ void k_interaction_dev(const double* __restrict__ dy, const double* __restrict__ adx,
                                  double* partials, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double val = (i < m) ? dy[i] * adx[i] : 0.0;
  detail::write_block_partial<kThreads>(val, partials);
}

// [PDLP] section 3.1, exactly as the host applies it in pdhg_gpu.cu: accept when eta is within
// the observed limit movement / interaction; the next eta is min(shrink * limit, grow * eta)
// with the exponents of the accepted count, clamped to [1e-12, eta_ceil]; a step with no
// interaction carries no information and leaves eta alone. One block of kThreads: the block
// sums the partials in a fixed order (#478), thread 0 applies the rule.
__global__ void k_step_dev(const double* __restrict__ partials, int bn, int bm, double* eta,
                           long long* accepted, int* accept_flag, double eta_ceil) {
  double scalars[3] = {0.0, 0.0, 0.0};
  detail::sum_step_partials<kThreads>(partials, bn, bm, scalars);
  if (threadIdx.x != 0) return;
  const double movement = scalars[0] + scalars[1];
  const double interaction = fabs(scalars[2]);
  const bool no_info = interaction <= 0.0;
  const double limit = no_info ? INFINITY : movement / interaction;
  const double e = *eta;
  const double expo = fmax(2.0, static_cast<double>(*accepted + 1));
  const double shrink = 1.0 - pow(expo, -0.3);
  const double grow = 1.0 + pow(expo, -0.6);
  const double proposed = fmin(shrink * limit, grow * e);
  const int accept = e <= limit ? 1 : 0;
  *accept_flag = accept;
  if (accept) *accepted += 1;
  if (!no_info) *eta = fmin(fmax(proposed, 1e-12), eta_ceil);
}

// An accepted step, committed by copy: the graph's addresses are fixed, so the host path's
// pointer swap is not available here.
__global__ void k_commit_dev(const int* __restrict__ accept_flag, double* __restrict__ x,
                             const double* __restrict__ x_next, double* __restrict__ x_sum,
                             double* __restrict__ y, const double* __restrict__ y_next,
                             double* __restrict__ y_sum, double* __restrict__ ax_cached,
                             const double* __restrict__ ax_next, int n, int m) {
  if (*accept_flag == 0) return;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) {
    x[k] = x_next[k];
    x_sum[k] += x_next[k];
  }
  if (k < m) {
    y[k] = y_next[k];
    y_sum[k] += y_next[k];
    if (ax_cached != nullptr) ax_cached[k] = ax_next[k];
  }
}

const double kOne = 1.0;
const double kZero = 0.0;

// Every launch goes to the loop's own stream: the legacy default stream cannot be used while
// a capture is in progress.
bool spmv(const DeviceLoopBuffers& b, cudaStream_t stream, cusparseOperation_t op, double* in,
          double* out) {
  const bool transpose = op == CUSPARSE_OPERATION_TRANSPOSE;
  const int out_size = transpose ? b.n : b.m;
  if (b.m == 0 || b.nnz == 0) {
    return out_size == 0 ||
           cudaMemsetAsync(out, 0, static_cast<size_t>(out_size) * sizeof(double), stream) ==
               cudaSuccess;
  }
  cusparseDnVecDescr_t vin = transpose ? b.vec_m : b.vec_n;
  cusparseDnVecDescr_t vout = transpose ? b.vec_n : b.vec_m;
  // With A^T held explicitly (deterministic mode, #478) the transpose product is a
  // non-transpose product on it, the only form CSR_ALG2 makes bit-wise repeatable.
  const bool explicit_t = transpose && b.matrix_t != nullptr;
  return cusparseDnVecSetValues(vin, in) == CUSPARSE_STATUS_SUCCESS &&
         cusparseDnVecSetValues(vout, out) == CUSPARSE_STATUS_SUCCESS &&
         cusparseSpMV(b.cusparse, explicit_t ? CUSPARSE_OPERATION_NON_TRANSPOSE : op, &kOne,
                      explicit_t ? b.matrix_t : b.matrix, vin, &kZero, vout, CUDA_R_64F,
                      b.spmv_alg, explicit_t ? b.spmv_buffer_t : b.spmv_buffer) ==
             CUSPARSE_STATUS_SUCCESS;
}

}  // namespace

bool DeviceLoop::record_iteration() {
  const int bn = (b_.n + kThreads - 1) / kThreads;
  const int bm = (b_.m + kThreads - 1) / kThreads;
  const int bk = (std::max(b_.n, b_.m) + kThreads - 1) / kThreads;
  if (!spmv(b_, stream_, CUSPARSE_OPERATION_TRANSPOSE, b_.y, b_.aty)) return false;
  if (b_.n > 0) {
    k_primal_dev<<<bn, kThreads, 0, stream_>>>(b_.x, b_.aty, b_.cost, b_.col_lo, b_.col_hi,
                                                b_.xn, b_.ext, b_.dx, b_.partials, b_.eta,
                                                b_.omega, b_.n);
  }
  if (b_.two_matvec) {
    if (!spmv(b_, stream_, CUSPARSE_OPERATION_NON_TRANSPOSE, b_.xn, b_.axn)) return false;
    if (b_.m > 0) {
      k_derive_dev<<<bm, kThreads, 0, stream_>>>(b_.axn, b_.axc, b_.ax, b_.adx, b_.m);
    }
  } else if (!spmv(b_, stream_, CUSPARSE_OPERATION_NON_TRANSPOSE, b_.ext, b_.ax)) {
    return false;
  }
  if (b_.m > 0) {
    k_dual_dev<<<bm, kThreads, 0, stream_>>>(b_.y, b_.ax, b_.row_lo, b_.row_hi, b_.yn, b_.dy,
                                              b_.partials + bn, b_.eta, b_.omega, b_.m);
  }
  if (!b_.two_matvec && !spmv(b_, stream_, CUSPARSE_OPERATION_NON_TRANSPOSE, b_.dx, b_.adx)) return false;
  if (b_.m > 0) {
    k_interaction_dev<<<bm, kThreads, 0, stream_>>>(b_.dy, b_.adx, b_.partials + bn + bm,
                                                     b_.m);
  }
  k_step_dev<<<1, kThreads, 0, stream_>>>(b_.partials, bn, bm, b_.eta, b_.accepted,
                                           b_.accept_flag, b_.eta_ceil);
  if (bk > 0) {
    k_commit_dev<<<bk, kThreads, 0, stream_>>>(b_.accept_flag, b_.x, b_.xn, b_.xsum, b_.y, b_.yn,
                                                b_.ysum, b_.two_matvec ? b_.axc : nullptr, b_.axn,
                                                b_.n, b_.m);
  }
  return cudaPeekAtLastError() == cudaSuccess;
}

bool DeviceLoop::init(const DeviceLoopBuffers& buffers, int block) {
  b_ = buffers;
  block_ = block;
  if (cudaStreamCreate(&stream_) != cudaSuccess) return false;
  if (b_.nnz > 0 && cusparseSetStream(b_.cusparse, stream_) != CUSPARSE_STATUS_SUCCESS) {
    return false;
  }
  // Each product once before the capture, into scratch every iteration overwrites (A^T y
  // into aty, A x into ax): with CUSPARSE_SPMV_CSR_ALG2 (deterministic mode) the first call
  // on a matrix sets up its state in the workspace, and that set-up belongs outside the graph
  // rather than replayed in it (#478).
  if (!spmv(b_, stream_, CUSPARSE_OPERATION_TRANSPOSE, b_.y, b_.aty) ||
      !spmv(b_, stream_, CUSPARSE_OPERATION_NON_TRANSPOSE, b_.x, b_.ax) ||
      cudaStreamSynchronize(stream_) != cudaSuccess) {
    return false;
  }
  // One iteration per recorded sequence, `block` of them in the graph: the host waits once
  // per block, not once per iteration.
  if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    return false;
  }
  bool recorded = true;
  for (int k = 0; k < block_ && recorded; ++k) recorded = record_iteration();
  const cudaError_t ended = cudaStreamEndCapture(stream_, &graph_);
  if (!recorded || ended != cudaSuccess) return false;
  return cudaGraphInstantiate(&exec_, graph_, 0) == cudaSuccess;
}

bool DeviceLoop::run_block() {
  return cudaGraphLaunch(exec_, stream_) == cudaSuccess &&
         cudaStreamSynchronize(stream_) == cudaSuccess;
}

DeviceLoop::~DeviceLoop() {
  if (exec_ != nullptr) cudaGraphExecDestroy(exec_);
  if (graph_ != nullptr) cudaGraphDestroy(graph_);
  if (stream_ != nullptr) {
    // The handle is the caller's; give it back its default stream before this one goes.
    if (b_.cusparse != nullptr) cusparseSetStream(b_.cusparse, nullptr);
    cudaStreamDestroy(stream_);
  }
}

}  // namespace sankhya::gpu
