// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the PDHG convergence evaluation on the device (#478 item 3).
//
// References (written from the papers; no solver source was consulted):
//   Applegate et al., "Practical Large-Scale Linear Programming using Primal-Dual Hybrid
//     Gradient", NeurIPS 2021, sections 3.3 and 4.3 (the KKT error the restarts and the
//     stopping rule are decided on).
//   Lu & Yang, "cuPDLP.jl", arXiv:2311.12180 (the evaluation kept on the device, only
//     scalars to the host).
//   Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM 2002, sections
//     3.1 and 4.1 (dot-product error, and why a fixed summation order fixes the rounding).
// The formulas are pdhg::evaluate's (src/pdhg/pdhg_evaluate.cpp), term for term.

#include "pdhg_device_eval.cuh"

#include <cusparse.h>
#include <math_constants.h>

#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "sankhya/sparse.hpp"
#include "sankhya/types.hpp"

namespace sankhya::gpu::eval {
namespace {

constexpr int kThreads = 256;
constexpr double kAbsentBound = kMpsInfinity;  // is_finite_bound's threshold, include/sankhya/types.hpp

// The positions of the two maxima among a point's kSumsPerPoint totals.
constexpr int kRowComplementarity = 3;
constexpr int kColumnComplementarity = kRowSums + 2;

__device__ __forceinline__ bool finite_bound(double v) {
  return !(isinf(v) || fabs(v) >= kAbsentBound);
}
// std::max and std::min exactly, so a tie or a NaN resolves as on the host.
__device__ __forceinline__ double max_of(double a, double b) { return (a < b) ? b : a; }
__device__ __forceinline__ double min_of(double a, double b) { return (b < a) ? b : a; }

struct SumOp {
  __device__ __forceinline__ double operator()(double a, double b) const { return a + b; }
};
struct MaxOp {
  __device__ __forceinline__ double operator()(double a, double b) const { return max_of(a, b); }
};

using BlockReduce = cub::BlockReduce<double, kThreads>;

// This block's total of v into dst[blockIdx.x]; the temporary storage is reused after.
template <typename Op>
__device__ __forceinline__ void block_total(double v, double* dst, Op op,
                                            typename BlockReduce::TempStorage& temp) {
  const double t = BlockReduce(temp).Reduce(v, op);
  if (threadIdx.x == 0) dst[blockIdx.x] = t;
  __syncthreads();
}

// A complementarity product as the host's running max sees it: max(0, v), with NaN (the
// 0 * inf of a zero multiplier on a one-sided row) ignored.
__device__ __forceinline__ double as_complementarity(double v) { return v > 0.0 ? v : 0.0; }

// Row side of pdhg::evaluate for rows [0, m) of this card: y_u = Dr y, A x_u = Dr^-1 (Ahat x).
// partials: kRowSums segments of `stride` slots, one slot per block.
__global__ void k_rows(const double* __restrict__ y, const double* __restrict__ ax,
                       const double* __restrict__ scale, const double* __restrict__ lo,
                       const double* __restrict__ hi, const double* __restrict__ y_restart,
                       double* __restrict__ partials, int stride, int m) {
  __shared__ typename BlockReduce::TempStorage temp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double primal_sq = 0.0, support = 0.0, dual_sq = 0.0, compl_max = 0.0, restart_sq = 0.0;
  if (i < m) {
    const double r = scale[i];
    const double yi = y[i] * r;
    const double a = ax[i] / r;
    const double l = lo[i], u = hi[i];
    const bool fl = finite_bound(l), fu = finite_bound(u);
    double violation = 0.0;
    if (fl) violation = max_of(violation, l - a);
    if (fu) violation = max_of(violation, a - u);
    primal_sq = violation * violation;
    if (yi > 0.0) {
      if (fu) support = yi * u; else dual_sq = yi * yi;
    } else if (yi < 0.0) {
      if (fl) support = yi * l; else dual_sq = yi * yi;
    }
    if (l != u) {  // an equality row is always tight
      const double inf = CUDART_INF;
      const double lower_slack = fl ? a - l : inf;
      const double upper_slack = fu ? u - a : inf;
      compl_max = as_complementarity(fabs(yi) * min_of(lower_slack, upper_slack));
    }
    if (y_restart != nullptr) {
      const double d = y[i] - y_restart[i];
      restart_sq = d * d;
    }
  }
  block_total(primal_sq, partials, SumOp{}, temp);
  block_total(support, partials + stride, SumOp{}, temp);
  block_total(dual_sq, partials + 2 * stride, SumOp{}, temp);
  block_total(compl_max, partials + 3 * stride, MaxOp{}, temp);
  block_total(restart_sq, partials + 4 * stride, SumOp{}, temp);
}

// Column side: x_u = Dc x, d = c + Dc^-1 (Ahat^T y). partials as for k_rows.
__global__ void k_columns(const double* __restrict__ x, const double* __restrict__ aty,
                          const double* __restrict__ scale, const double* __restrict__ cost,
                          const double* __restrict__ lo, const double* __restrict__ hi,
                          const double* __restrict__ x_restart, double* __restrict__ partials,
                          int stride, int n) {
  __shared__ typename BlockReduce::TempStorage temp;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  double dual_sq = 0.0, bound = 0.0, compl_max = 0.0, objective = 0.0, restart_sq = 0.0;
  if (j < n) {
    const double c = scale[j];
    const double xj = x[j] * c;
    const double d = cost[j] + aty[j] / c;
    const double l = lo[j], u = hi[j];
    const bool fl = finite_bound(l), fu = finite_bound(u);
    if (d > 0.0) {
      if (fl) bound = d * l; else dual_sq = d * d;
    } else if (d < 0.0) {
      if (fu) bound = d * u; else dual_sq = d * d;
    }
    if (l != u) {  // a fixed column is always tight
      const double inf = CUDART_INF;
      const double lower_slack = fl ? xj - l : inf;
      const double upper_slack = fu ? u - xj : inf;
      compl_max = as_complementarity(fabs(d) * min_of(lower_slack, upper_slack));
    }
    objective = cost[j] * xj;
    if (x_restart != nullptr) {
      const double dx = x[j] - x_restart[j];
      restart_sq = dx * dx;
    }
  }
  block_total(dual_sq, partials, SumOp{}, temp);
  block_total(bound, partials + stride, SumOp{}, temp);
  block_total(compl_max, partials + 2 * stride, MaxOp{}, temp);
  block_total(objective, partials + 3 * stride, SumOp{}, temp);
  block_total(restart_sq, partials + 4 * stride, SumOp{}, temp);
}

// One block per total: thread t combines slots t, t + kThreads, ... left to right, then a
// BlockReduce combines the threads. The order depends on the slot count alone.
__global__ void k_finish(const double* __restrict__ partials, int bm, int bn, int point_stride,
                         double* __restrict__ sums) {
  __shared__ typename BlockReduce::TempStorage temp;
  const int point = blockIdx.x / kSumsPerPoint;
  const int k = blockIdx.x % kSumsPerPoint;
  const double* base = partials + static_cast<std::size_t>(point) * point_stride;
  const double* src = k < kRowSums ? base + k * bm : base + kRowSums * bm + (k - kRowSums) * bn;
  const int count = k < kRowSums ? bm : bn;
  const bool is_max = k == kRowComplementarity || k == kColumnComplementarity;
  double t = 0.0;
  for (int s = static_cast<int>(threadIdx.x); s < count; s += kThreads) {
    t = is_max ? max_of(t, src[s]) : t + src[s];
  }
  const double total =
      is_max ? BlockReduce(temp).Reduce(t, MaxOp{}) : BlockReduce(temp).Reduce(t, SumOp{});
  if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

__global__ void k_divide(const double* __restrict__ sum, double count, double* __restrict__ out,
                         int len) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < len) out[j] = sum[j] / count;
}

int blocks_for(int count) { return (count + kThreads - 1) / kThreads; }
bool launched() { return cudaPeekAtLastError() == cudaSuccess; }

double* device_zeros(std::size_t count) {
  if (count == 0) return nullptr;
  double* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(double)) != cudaSuccess) return nullptr;
  if (cudaMemset(p, 0, count * sizeof(double)) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

double* device_copy(const double* host, std::size_t count) {
  double* p = device_zeros(count);
  if (p == nullptr) return nullptr;
  if (cudaMemcpy(p, host, count * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

}  // namespace

// ---- Host side --------------------------------------------------------------

RowSums row_sums(const double* sums, int point) {
  const double* s = sums + static_cast<std::size_t>(point) * kSumsPerPoint;
  return RowSums{s[0], s[1], s[2], s[3], s[4]};
}

ColumnSums column_sums(const double* sums, int point) {
  const double* s = sums + static_cast<std::size_t>(point) * kSumsPerPoint + kRowSums;
  return ColumnSums{s[0], s[1], s[2], s[3], s[4]};
}

void accumulate(RowSums& into, const RowSums& part) {
  into.primal_sq += part.primal_sq;
  into.support += part.support;
  into.dual_sq += part.dual_sq;
  into.complementarity = std::max(into.complementarity, part.complementarity);
  into.restart_sq += part.restart_sq;
}

pdhg::Residuals assemble(const pdhg::Problem& problem, const RowSums& rows,
                         const ColumnSums& columns) {
  pdhg::Residuals r;
  r.absolute_primal = std::sqrt(rows.primal_sq);
  r.primal = r.absolute_primal / (1.0 + problem.bound_norm);
  r.absolute_dual = std::sqrt(columns.dual_sq + rows.dual_sq);
  r.dual = r.absolute_dual / (1.0 + problem.cost_norm);
  r.complementarity = std::max(rows.complementarity, columns.complementarity);
  r.primal_objective = columns.objective;
  r.dual_objective = columns.bound - rows.support;
  const double absolute_gap = std::fabs(r.primal_objective - r.dual_objective);
  r.gap = absolute_gap / (1.0 + std::fabs(r.primal_objective) + std::fabs(r.dual_objective));
  r.gap_as_verified = absolute_gap / std::max(1.0, std::fabs(r.primal_objective));
  return r;
}

// ---- DeviceEvaluator ----------------------------------------------------------

DeviceEvaluator::~DeviceEvaluator() {
  if (device_ < 0) return;
  int previous = 0;
  const bool restore = cudaGetDevice(&previous) == cudaSuccess;
  cudaSetDevice(device_);
  for (double* p : {row_scale_, row_lo_, row_hi_, ax_, y_avg_, y_restart_, y_best_, col_scale_,
                    cost_, col_lo_, col_hi_, aty_, x_avg_, x_restart_, x_best_, partials_,
                    sums_})
    if (p) cudaFree(p);
  if (host_sums_) cudaFreeHost(host_sums_);
  if (restore) cudaSetDevice(previous);
}

bool DeviceEvaluator::init(const pdhg::Problem& problem, const Scaling& scaling, int row_begin,
                           int row_end, bool columns, const double* x_start_device) {
  if (cudaGetDevice(&device_) != cudaSuccess) return false;
  const Model& model = *problem.model;
  n_ = static_cast<int>(model.num_cols());
  m_ = row_end - row_begin;
  bn_ = columns ? blocks_for(n_) : 0;
  bm_ = blocks_for(m_);
  columns_ = columns;
  const auto n = static_cast<std::size_t>(n_);
  const auto m = static_cast<std::size_t>(m_);
  const auto r0 = static_cast<std::size_t>(row_begin);

  if (m > 0) {
    row_scale_ = device_copy(scaling.row.data() + r0, m);
    row_lo_ = device_copy(model.row_lower.data() + r0, m);
    row_hi_ = device_copy(model.row_upper.data() + r0, m);
    ax_ = device_zeros(m);
    y_avg_ = device_zeros(m);
    y_restart_ = device_zeros(m);  // the host path's y_restart starts at zero
    y_best_ = device_zeros(m);
    if (!row_scale_ || !row_lo_ || !row_hi_ || !ax_ || !y_avg_ || !y_restart_ || !y_best_)
      return false;
  }
  if (n > 0) {
    x_avg_ = device_zeros(n);
    if (!x_avg_) return false;
  }
  if (columns && n > 0) {
    col_scale_ = device_copy(scaling.column.data(), n);
    cost_ = device_copy(problem.cost.data(), n);
    col_lo_ = device_copy(model.col_lower.data(), n);
    col_hi_ = device_copy(model.col_upper.data(), n);
    aty_ = device_zeros(n);
    x_restart_ = device_zeros(n);
    x_best_ = device_zeros(n);  // the host path's best_x starts at zero
    if (!col_scale_ || !cost_ || !col_lo_ || !col_hi_ || !aty_ || !x_restart_ || !x_best_)
      return false;
    if (x_start_device != nullptr &&
        cudaMemcpy(x_restart_, x_start_device, n * sizeof(double), cudaMemcpyDeviceToDevice) !=
            cudaSuccess)
      return false;
  }
  const std::size_t point_stride =
      static_cast<std::size_t>(kRowSums * bm_ + kColumnSums * bn_);
  partials_ = device_zeros(std::max<std::size_t>(1, kMaxPoints * point_stride));
  sums_ = device_zeros(kMaxPoints * kSumsPerPoint);
  if (!partials_ || !sums_) return false;
  if (cudaMallocHost(reinterpret_cast<void**>(&host_sums_),
                     kMaxPoints * kSumsPerPoint * sizeof(double)) != cudaSuccess) {
    host_sums_ = nullptr;
    return false;
  }
  std::fill(host_sums_, host_sums_ + kMaxPoints * kSumsPerPoint, 0.0);
  return true;
}

bool DeviceEvaluator::average(cudaStream_t s, const double* x_sum, const double* y_sum,
                              double count) {
  if (n_ > 0) k_divide<<<blocks_for(n_), kThreads, 0, s>>>(x_sum, count, x_avg_, n_);
  if (m_ > 0) k_divide<<<blocks_for(m_), kThreads, 0, s>>>(y_sum, count, y_avg_, m_);
  return launched();
}

bool DeviceEvaluator::rows(cudaStream_t s, int point, const double* y, const double* ax,
                           bool with_restart) {
  if (m_ == 0) return true;
  const std::size_t point_stride =
      static_cast<std::size_t>(kRowSums * bm_ + kColumnSums * bn_);
  k_rows<<<bm_, kThreads, 0, s>>>(y, ax, row_scale_, row_lo_, row_hi_,
                                  with_restart ? y_restart_ : nullptr,
                                  partials_ + static_cast<std::size_t>(point) * point_stride,
                                  bm_, m_);
  return launched();
}

bool DeviceEvaluator::columns(cudaStream_t s, int point, const double* x, const double* aty,
                              bool with_restart) {
  if (!columns_ || n_ == 0) return columns_;
  const std::size_t point_stride =
      static_cast<std::size_t>(kRowSums * bm_ + kColumnSums * bn_);
  k_columns<<<bn_, kThreads, 0, s>>>(
      x, aty, col_scale_, cost_, col_lo_, col_hi_, with_restart ? x_restart_ : nullptr,
      partials_ + static_cast<std::size_t>(point) * point_stride + kRowSums * bm_, bn_, n_);
  return launched();
}

bool DeviceEvaluator::finish(cudaStream_t s, int points) {
  const int point_stride = kRowSums * bm_ + kColumnSums * bn_;
  std::fill(host_sums_ + points * kSumsPerPoint, host_sums_ + kMaxPoints * kSumsPerPoint, 0.0);
  k_finish<<<points * kSumsPerPoint, kThreads, 0, s>>>(partials_, bm_, bn_, point_stride, sums_);
  if (!launched()) return false;
  return cudaMemcpyAsync(host_sums_, sums_,
                         static_cast<std::size_t>(points) * kSumsPerPoint * sizeof(double),
                         cudaMemcpyDeviceToHost, s) == cudaSuccess;
}

bool DeviceEvaluator::copy_point(cudaStream_t s, double* x_dst, double* y_dst, const double* x,
                                 const double* y) const {
  if (columns_ && n_ > 0 &&
      cudaMemcpyAsync(x_dst, x, static_cast<std::size_t>(n_) * sizeof(double),
                      cudaMemcpyDeviceToDevice, s) != cudaSuccess)
    return false;
  return m_ == 0 || cudaMemcpyAsync(y_dst, y, static_cast<std::size_t>(m_) * sizeof(double),
                                    cudaMemcpyDeviceToDevice, s) == cudaSuccess;
}

bool DeviceEvaluator::set_restart(cudaStream_t s, const double* x, const double* y) {
  return copy_point(s, x_restart_, y_restart_, x, y);
}

bool DeviceEvaluator::set_best(cudaStream_t s, const double* x, const double* y) {
  return copy_point(s, x_best_, y_best_, x, y);
}

bool DeviceEvaluator::download_best(cudaStream_t s, double* x, double* y) const {
  if (columns_ && n_ > 0 &&
      cudaMemcpyAsync(x, x_best_, static_cast<std::size_t>(n_) * sizeof(double),
                      cudaMemcpyDeviceToHost, s) != cudaSuccess)
    return false;
  if (m_ > 0 && cudaMemcpyAsync(y, y_best_, static_cast<std::size_t>(m_) * sizeof(double),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess)
    return false;
  return cudaStreamSynchronize(s) == cudaSuccess;
}

// ---- Test seam ----------------------------------------------------------------

namespace {

// The deterministic single-GPU engine's products, set up on their own for the test:
// CSR_ALG2 non-transpose on A and on an explicit A^T, one workspace per descriptor.
struct TestProducts {
  int *rp{}, *ci{}, *trp{}, *tci{};
  double *v{}, *tv{};
  void *buf{}, *tbuf{};
  cusparseHandle_t cs{};
  cusparseSpMatDescr_t a{}, at{};
  cusparseDnVecDescr_t vn{}, vm{};
  ~TestProducts() {
    if (vm) cusparseDestroyDnVec(vm);
    if (vn) cusparseDestroyDnVec(vn);
    if (at) cusparseDestroySpMat(at);
    if (a) cusparseDestroySpMat(a);
    if (cs) cusparseDestroy(cs);
    for (void* p : {static_cast<void*>(rp), static_cast<void*>(ci), static_cast<void*>(trp),
                    static_cast<void*>(tci), static_cast<void*>(v), static_cast<void*>(tv), buf,
                    tbuf})
      if (p) cudaFree(p);
  }
};

template <typename T>
bool upload_to(const std::vector<T>& host, T** dev) {
  if (host.empty()) return true;
  if (cudaMalloc(dev, host.size() * sizeof(T)) != cudaSuccess) return false;
  return cudaMemcpy(*dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
         cudaSuccess;
}

bool product(TestProducts& p, cusparseSpMatDescr_t mat, void* buf, double* in, double* out) {
  const double one = 1.0, zero = 0.0;
  return cusparseDnVecSetValues(mat == p.a ? p.vn : p.vm, in) == CUSPARSE_STATUS_SUCCESS &&
         cusparseDnVecSetValues(mat == p.a ? p.vm : p.vn, out) == CUSPARSE_STATUS_SUCCESS &&
         cusparseSpMV(p.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat,
                      mat == p.a ? p.vn : p.vm, &zero, mat == p.a ? p.vm : p.vn, CUDA_R_64F,
                      CUSPARSE_SPMV_CSR_ALG2, buf) == CUSPARSE_STATUS_SUCCESS;
}

}  // namespace

bool evaluate_for_testing(const pdhg::Problem& problem, const Scaling& scaling,
                          const std::vector<double>& x_s, const std::vector<double>& y_s,
                          const std::vector<double>& x_restart,
                          const std::vector<double>& y_restart, double count,
                          TestEvaluation* out) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return false;
  const int n = static_cast<int>(problem.model->num_cols());
  const int m = static_cast<int>(problem.model->num_rows());
  if (n == 0 || m == 0) return false;  // the seam is for real models
  const CsrView csr(scaling.matrix);
  const int nnz = static_cast<int>(csr.values().size());
  if (nnz == 0) return false;

  TestProducts p;
  double *dx = nullptr, *dy = nullptr, *dxr = nullptr, *dyr = nullptr;
  auto release = [&]() {
    for (double* q : {dx, dy, dxr, dyr})
      if (q) cudaFree(q);
  };
  const bool uploaded =
      upload_to(csr.row_starts(), &p.rp) && upload_to(csr.column_indices(), &p.ci) &&
      upload_to(csr.values(), &p.v) && upload_to(scaling.matrix.column_starts(), &p.trp) &&
      upload_to(scaling.matrix.row_indices(), &p.tci) &&
      upload_to(scaling.matrix.values(), &p.tv) && upload_to(x_s, &dx) && upload_to(y_s, &dy) &&
      upload_to(x_restart, &dxr) && upload_to(y_restart, &dyr);
  if (!uploaded) {
    release();
    return false;
  }
  const double one = 1.0, zero = 0.0;
  std::size_t bytes = 0, tbytes = 0;
  const bool described =
      cusparseCreate(&p.cs) == CUSPARSE_STATUS_SUCCESS &&
      cusparseCreateCsr(&p.a, m, n, nnz, p.rp, p.ci, p.v, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS &&
      cusparseCreateCsr(&p.at, n, m, nnz, p.trp, p.tci, p.tv, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS &&
      cusparseCreateDnVec(&p.vn, n, dx, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS &&
      cusparseCreateDnVec(&p.vm, m, dy, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS &&
      cusparseSpMV_bufferSize(p.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, p.a, p.vn, &zero,
                              p.vm, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                              &bytes) == CUSPARSE_STATUS_SUCCESS &&
      cusparseSpMV_bufferSize(p.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, p.at, p.vm, &zero,
                              p.vn, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                              &tbytes) == CUSPARSE_STATUS_SUCCESS &&
      (bytes == 0 || cudaMalloc(&p.buf, bytes) == cudaSuccess) &&
      (tbytes == 0 || cudaMalloc(&p.tbuf, tbytes) == cudaSuccess);
  bool ok = described;
  {
    DeviceEvaluator ev;
    const cudaStream_t s = nullptr;
    ok = ok && ev.init(problem, scaling, 0, m, true, dxr) && ev.set_restart(s, dxr, dyr);
    const bool averaged = count > 0.0;
    const int point = averaged ? 1 : 0;
    if (ok && averaged) ok = ev.average(s, dx, dy, count);
    double* px = averaged ? ev.x_average() : dx;
    double* py = averaged ? ev.y_average() : dy;
    ok = ok && product(p, p.a, p.buf, px, ev.ax()) && ev.rows(s, point, py, ev.ax(), !averaged) &&
         product(p, p.at, p.tbuf, py, ev.aty()) &&
         ev.columns(s, point, px, ev.aty(), !averaged) && ev.finish(s, point + 1) &&
         cudaStreamSynchronize(s) == cudaSuccess;
    if (ok) {
      const RowSums rs = row_sums(ev.host_sums(), point);
      const ColumnSums cs = column_sums(ev.host_sums(), point);
      out->residuals = assemble(problem, rs, cs);
      out->restart_dx = std::sqrt(cs.restart_sq);
      out->restart_dy = std::sqrt(rs.restart_sq);
    }
  }
  release();
  return ok;
}

}  // namespace sankhya::gpu::eval
