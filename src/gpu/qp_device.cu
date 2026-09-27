// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the first-order QP operator on the device (#493). See qp_device.hpp.
//
// References (derived from the papers; no solver source consulted):
//   Condat, JOTA 158 (2013); Vu, Adv. Comput. Math. 38 (2013): the step this runs.
//   Lu & Yang, PDQP, arXiv:2311.07710: the same operator on a card, with Q x as one more
//     sparse product beside A x and A^T y.
//   NVIDIA cuSPARSE documentation: CUSPARSE_SPMV_CSR_ALG2, the algorithm documented as
//     bit-wise repeatable run to run, used for all three products on explicitly stored
//     matrices (A and A^T as CSR, Q expanded to its full symmetric CSR).
//
// Every reduction is fixed-order (pdhg_reduce.cuh), so a device run repeats itself. The
// arithmetic is the host operator's expression for expression (src/qp/qp_operator.cpp);
// the two differ in rounding only where the sparse products sum in a different order.

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../qp/qp_operator.hpp"
#include "pdhg_reduce.cuh"
#include "qp_device.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/types.hpp"

namespace sankhya::gpu {
namespace {

constexpr int kThreads = 256;
const double kOne = 1.0;
const double kZero = 0.0;

// ---- kernels -----------------------------------------------------------------------------

__device__ __forceinline__ double d_project(double v, double lo, double hi) {
  // is_finite_bound on the host: a bound is infinite when it is +-inf.
  if (!isinf(lo) && v < lo) return lo;
  if (!isinf(hi) && v > hi) return hi;
  return v;
}

// x' = proj_box(x - tau (cost + qx + aty)), xbar = 2 x' - x. cost and qx carry the sense.
__global__ void k_primal(int n, const double* __restrict__ x, const double* __restrict__ qx,
                         const double* __restrict__ aty, const double* __restrict__ cost,
                         const double* __restrict__ lo, const double* __restrict__ hi,
                         double tau, double* __restrict__ xn, double* __restrict__ ext) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const double gradient = cost[j] + qx[j] + aty[j];
  const double next = d_project(x[j] - tau * gradient, lo[j], hi[j]);
  xn[j] = next;
  ext[j] = 2.0 * next - x[j];
}

// y' = v - sigma proj_C(v / sigma), v = y + sigma (A xbar).
__global__ void k_dual(int m, const double* __restrict__ y, const double* __restrict__ ax,
                       const double* __restrict__ lo, const double* __restrict__ hi,
                       double sigma, double* __restrict__ yn) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const double v = y[i] + sigma * ax[i];
  yn[i] = v - sigma * d_project(v / sigma, lo[i], hi[i]);
}

// z <- w ((1 + rho) tz - rho z) + (1 - w) anchor, w = (k + 1) / (k + 2): halpern_blend.
__global__ void k_blend(int n, double* __restrict__ z, const double* __restrict__ tz,
                        const double* __restrict__ anchor, double k, double rho) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const double w = (k + 1.0) / (k + 2.0);
  const double anchor_share = 1.0 - w;
  const double reflected = (1.0 + rho) * tz[j] - rho * z[j];
  z[j] = w * reflected + anchor_share * anchor[j];
}

// Per-block partial sums of (a - b)^2 into partials[blockIdx.x].
__global__ void k_sqdiff_partials(int n, const double* __restrict__ a,
                                  const double* __restrict__ b, double* __restrict__ partials) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  double v = 0.0;
  if (j < n) {
    const double d = a[j] - b[j];
    v = d * d;
  }
  detail::write_block_partial<kThreads>(v, partials);
}

// out[0] = sum of the first bn partials, out[1] = sum of the next bm, each in a fixed order
// by one block (the same scheme as sum_step_partials).
__global__ void k_finalize_two(const double* __restrict__ partials, int bn, int bm,
                               double* __restrict__ out) {
  using BlockReduce = cub::BlockReduce<double, kThreads>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int counts[2] = {bn, bm};
  int offset = 0;
  for (int s = 0; s < 2; ++s) {
    double t = 0.0;
    for (int k = static_cast<int>(threadIdx.x); k < counts[s]; k += kThreads) {
      t += partials[offset + k];
    }
    const double total = BlockReduce(temp).Sum(t);
    if (threadIdx.x == 0) out[s] = total;
    __syncthreads();
    offset += counts[s];
  }
}

// ---- the operator ------------------------------------------------------------------------

[[nodiscard]] int blocks_for(int count) {
  return count > 0 ? (count + kThreads - 1) / kThreads : 1;
}

class DeviceOperator final : public qp::QpOperator {
 public:
  ~DeviceOperator() override {
    for (cusparseDnVecDescr_t v : {vn_in_, vn_out_, vm_in_, vm_out_}) {
      if (v) cusparseDestroyDnVec(v);
    }
    for (cusparseSpMatDescr_t s : {mat_a_, mat_at_, mat_q_}) {
      if (s) cusparseDestroySpMat(s);
    }
    if (cs_) cusparseDestroy(cs_);
    for (void* p : {static_cast<void*>(d_x_),
                    static_cast<void*>(d_y_),
                    static_cast<void*>(d_xn_),
                    static_cast<void*>(d_yn_),
                    static_cast<void*>(d_ext_),
                    static_cast<void*>(d_qx_),
                    static_cast<void*>(d_aty_),
                    static_cast<void*>(d_ax_),
                    static_cast<void*>(d_cost_),
                    static_cast<void*>(d_clo_),
                    static_cast<void*>(d_chi_),
                    static_cast<void*>(d_rlo_),
                    static_cast<void*>(d_rhi_),
                    static_cast<void*>(d_xanchor_),
                    static_cast<void*>(d_yanchor_),
                    static_cast<void*>(d_xrestart_),
                    static_cast<void*>(d_yrestart_),
                    static_cast<void*>(d_partials_),
                    static_cast<void*>(d_scalars_),
                    static_cast<void*>(d_arp_),
                    static_cast<void*>(d_aci_),
                    static_cast<void*>(d_av_),
                    static_cast<void*>(d_trp_),
                    static_cast<void*>(d_tci_),
                    static_cast<void*>(d_tv_),
                    static_cast<void*>(d_qrp_),
                    static_cast<void*>(d_qci_),
                    static_cast<void*>(d_qv_),
                    ws_a_,
                    ws_at_,
                    ws_q_}) {
      if (p) cudaFree(p);
    }
    if (h_scalars_) cudaFreeHost(h_scalars_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  [[nodiscard]] bool init(const Model& model, std::string* reason) {
    n_ = static_cast<int>(model.num_cols());
    m_ = static_cast<int>(model.num_rows());
    const double sense = model.sense_multiplier();
    const auto n = static_cast<std::size_t>(n_);
    const auto m = static_cast<std::size_t>(m_);
    bn_ = blocks_for(n_);
    bm_ = blocks_for(m_);

    if (cudaStreamCreate(&stream_) != cudaSuccess) return fail(reason, "cudaStreamCreate");
    for (double** p : {&d_x_, &d_xn_, &d_ext_, &d_qx_, &d_aty_, &d_cost_, &d_clo_, &d_chi_,
                       &d_xanchor_, &d_xrestart_}) {
      if (!zeros(p, n)) return fail(reason, "cudaMalloc (columns)");
    }
    for (double** p : {&d_y_, &d_yn_, &d_ax_, &d_rlo_, &d_rhi_, &d_yanchor_, &d_yrestart_}) {
      if (!zeros(p, m)) return fail(reason, "cudaMalloc (rows)");
    }
    if (!zeros(&d_partials_, static_cast<std::size_t>(bn_ + bm_)) || !zeros(&d_scalars_, 2))
      return fail(reason, "cudaMalloc (reductions)");
    if (cudaMallocHost(reinterpret_cast<void**>(&h_scalars_), 2 * sizeof(double)) !=
        cudaSuccess)
      return fail(reason, "cudaMallocHost");

    // Problem data. The cost carries the sense; so does Q, so the kernels add plain sums.
    std::vector<double> cost(n), x0(n);
    for (std::size_t j = 0; j < n; ++j) {
      cost[j] = sense * model.col_cost[j];
      const double lo = model.col_lower[j], hi = model.col_upper[j];
      double v = 0.0;
      if (is_finite_bound(lo) && v < lo) v = lo;
      if (is_finite_bound(hi) && v > hi) v = hi;
      x0[j] = v;
    }
    if (!upload(cost.data(), d_cost_, n) || !upload(x0.data(), d_x_, n) ||
        !upload(model.col_lower.data(), d_clo_, n) ||
        !upload(model.col_upper.data(), d_chi_, n) ||
        !upload(model.row_lower.data(), d_rlo_, m) ||
        !upload(model.row_upper.data(), d_rhi_, m))
      return fail(reason, "upload (vectors)");

    if (cusparseCreate(&cs_) != CUSPARSE_STATUS_SUCCESS) return fail(reason, "cusparseCreate");
    if (cusparseSetStream(cs_, stream_) != CUSPARSE_STATUS_SUCCESS)
      return fail(reason, "cusparseSetStream");
    if (cusparseCreateDnVec(&vn_in_, n_, d_ext_, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateDnVec(&vn_out_, n_, d_aty_, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
      return fail(reason, "cusparseCreateDnVec (columns)");
    if (m_ > 0 &&
        (cusparseCreateDnVec(&vm_in_, m_, d_y_, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS ||
         cusparseCreateDnVec(&vm_out_, m_, d_ax_, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS))
      return fail(reason, "cusparseCreateDnVec (rows)");

    // A as CSR, A^T as CSR (the CSC arrays of A, as stored), Q as its full symmetric CSR
    // from the lower triangle, scaled by the sense.
    if (m_ > 0 && model.matrix.num_nonzeros() > 0) {
      const CsrView csr(model.matrix);
      if (!upload_csr(csr.row_starts(), csr.column_indices(), csr.values(), 1.0, &d_arp_,
                      &d_aci_, &d_av_) ||
          !upload_csr(model.matrix.column_starts(), model.matrix.row_indices(),
                      model.matrix.values(), 1.0, &d_trp_, &d_tci_, &d_tv_))
        return fail(reason, "upload (A)");
      const auto nnz = static_cast<int64_t>(model.matrix.num_nonzeros());
      if (cusparseCreateCsr(&mat_a_, m_, n_, nnz, d_arp_, d_aci_, d_av_, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                            CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS ||
          cusparseCreateCsr(&mat_at_, n_, m_, nnz, d_trp_, d_tci_, d_tv_, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                            CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        return fail(reason, "cusparseCreateCsr (A)");
      if (!workspace(mat_a_, vn_in_, vm_out_, &ws_a_) ||
          !workspace(mat_at_, vm_in_, vn_out_, &ws_at_))
        return fail(reason, "cusparseSpMV_bufferSize (A)");
    }
    if (model.hessian.num_nonzeros() > 0) {
      std::vector<Index> rp, ci;
      std::vector<double> v;
      full_symmetric_csr(model.hessian, sense, &rp, &ci, &v);
      if (!upload_csr(rp, ci, v, 1.0, &d_qrp_, &d_qci_, &d_qv_))
        return fail(reason, "upload (Q)");
      if (cusparseCreateCsr(&mat_q_, n_, n_, static_cast<int64_t>(v.size()), d_qrp_, d_qci_,
                            d_qv_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
        return fail(reason, "cusparseCreateCsr (Q)");
      // Q's product reads x through the n-vector descriptor and writes qx: same shapes as
      // A^T y's output, so the same descriptor objects serve with their pointers swapped.
      if (!workspace(mat_q_, vn_in_, vn_out_, &ws_q_))
        return fail(reason, "cusparseSpMV_bufferSize (Q)");
    }
    return sync(reason);
  }

  bool step(double tau, double sigma) override {
    // qx = Q x (or 0), aty = A^T y (or 0).
    if (mat_q_) {
      if (!spmv(mat_q_, vn_in_, d_x_, vn_out_, d_qx_, ws_q_)) return false;
    }
    if (mat_at_) {
      if (!spmv(mat_at_, vm_in_, d_y_, vn_out_, d_aty_, ws_at_)) return false;
    }
    k_primal<<<bn_, kThreads, 0, stream_>>>(n_, d_x_, d_qx_, d_aty_, d_cost_, d_clo_, d_chi_,
                                            tau, d_xn_, d_ext_);
    if (mat_a_) {
      if (!spmv(mat_a_, vn_in_, d_ext_, vm_out_, d_ax_, ws_a_)) return false;
      k_dual<<<bm_, kThreads, 0, stream_>>>(m_, d_y_, d_ax_, d_rlo_, d_rhi_, sigma, d_yn_);
    }
    return cudaGetLastError() == cudaSuccess;
  }

  double fixed_point_residual(double tau, double sigma) override {
    double dx2 = 0.0, dy2 = 0.0;
    if (!two_squared_distances(d_xn_, d_x_, d_yn_, d_y_, &dx2, &dy2)) return NAN;
    return std::sqrt(dx2 / tau + (m_ > 0 ? dy2 / sigma : 0.0));
  }

  bool advance(double k, double rho) override {
    if (rho < 0.0) {
      std::swap(d_x_, d_xn_);
      std::swap(d_y_, d_yn_);
      return true;
    }
    k_blend<<<bn_, kThreads, 0, stream_>>>(n_, d_x_, d_xn_, d_xanchor_, k, rho);
    if (m_ > 0) k_blend<<<bm_, kThreads, 0, stream_>>>(m_, d_y_, d_yn_, d_yanchor_, k, rho);
    return cudaGetLastError() == cudaSuccess;
  }

  bool take_tz() override { return copy(d_xn_, d_x_, n_) && copy(d_yn_, d_y_, m_); }
  bool set_anchor() override {
    return copy(d_x_, d_xanchor_, n_) && copy(d_y_, d_yanchor_, m_);
  }
  bool set_restart() override {
    return copy(d_x_, d_xrestart_, n_) && copy(d_y_, d_yrestart_, m_);
  }

  bool restart_distance(double* dx, double* dy) override {
    double dx2 = 0.0, dy2 = 0.0;
    if (!two_squared_distances(d_x_, d_xrestart_, d_y_, d_yrestart_, &dx2, &dy2)) return false;
    *dx = std::sqrt(dx2);
    *dy = std::sqrt(dy2);
    return true;
  }

  bool download(bool tz, std::vector<double>* x, std::vector<double>* y) override {
    x->resize(static_cast<std::size_t>(n_));
    y->resize(static_cast<std::size_t>(m_));
    if (cudaStreamSynchronize(stream_) != cudaSuccess) return false;
    if (n_ > 0 &&
        cudaMemcpy(x->data(), tz ? d_xn_ : d_x_, static_cast<std::size_t>(n_) * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
    if (m_ > 0 &&
        cudaMemcpy(y->data(), tz ? d_yn_ : d_y_, static_cast<std::size_t>(m_) * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
    return true;
  }

  const char* where() const override { return "device"; }

 private:
  static bool fail(std::string* reason, const char* what) {
    if (reason) *reason = what;
    return false;
  }
  bool sync(std::string* reason) const {
    if (cudaStreamSynchronize(stream_) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      return fail(reason, "a CUDA error during setup");
    return true;
  }
  static bool zeros(double** p, std::size_t count) {
    if (count == 0) {
      *p = nullptr;
      return true;
    }
    if (cudaMalloc(reinterpret_cast<void**>(p), count * sizeof(double)) != cudaSuccess)
      return false;
    return cudaMemset(*p, 0, count * sizeof(double)) == cudaSuccess;
  }
  template <typename T>
  static bool upload(const T* src, T* dst, std::size_t count) {
    if (count == 0) return true;
    return cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  template <typename T>
  static bool upload_new(const std::vector<T>& src, T** dst) {
    if (src.empty()) return true;
    if (cudaMalloc(reinterpret_cast<void**>(dst), src.size() * sizeof(T)) != cudaSuccess)
      return false;
    return upload(src.data(), *dst, src.size());
  }
  static bool upload_csr(const std::vector<Index>& rp, const std::vector<Index>& ci,
                         const std::vector<double>& v, double scale, int** d_rp, int** d_ci,
                         double** d_v) {
    static_assert(sizeof(Index) == sizeof(int), "cuSPARSE 32-bit indices");
    std::vector<int> rp32(rp.begin(), rp.end()), ci32(ci.begin(), ci.end());
    std::vector<double> scaled(v);
    if (scale != 1.0) {
      for (double& value : scaled) value *= scale;
    }
    return upload_new(rp32, d_rp) && upload_new(ci32, d_ci) && upload_new(scaled, d_v);
  }
  /// Q as a full symmetric CSR from the stored lower triangle, times `scale`. Within a row
  /// the columns are ascending, so the product's summation order is fixed by the pattern.
  static void full_symmetric_csr(const SparseMatrix& lower, double scale,
                                 std::vector<Index>* rp, std::vector<Index>* ci,
                                 std::vector<double>* v) {
    const Index n = lower.num_cols();
    const auto un = static_cast<std::size_t>(n);
    std::vector<Index> count(un + 1, 0);
    for (Index j = 0; j < n; ++j) {
      const ColumnView column = lower.column(j);
      for (Index k = 0; k < column.size; ++k) {
        ++count[static_cast<std::size_t>(column.rows[k]) + 1];
        if (column.rows[k] != j) ++count[static_cast<std::size_t>(j) + 1];
      }
    }
    rp->assign(un + 1, 0);
    for (std::size_t i = 0; i < un; ++i) (*rp)[i + 1] = (*rp)[i] + count[i + 1];
    ci->assign(static_cast<std::size_t>((*rp)[un]), 0);
    v->assign(ci->size(), 0.0);
    std::vector<Index> next(rp->begin(), rp->end() - 1);
    // Columns j ascending: entry (i, j) lands in row i, and (j, i) in row j, so each row's
    // columns come out ascending because j only grows.
    for (Index j = 0; j < n; ++j) {
      const ColumnView column = lower.column(j);
      for (Index k = 0; k < column.size; ++k) {
        const Index i = column.rows[k];
        const double value = scale * column.values[k];
        auto& slot_i = next[static_cast<std::size_t>(i)];
        (*ci)[static_cast<std::size_t>(slot_i)] = j;
        (*v)[static_cast<std::size_t>(slot_i)] = value;
        ++slot_i;
        if (i != j) {
          auto& slot_j = next[static_cast<std::size_t>(j)];
          (*ci)[static_cast<std::size_t>(slot_j)] = i;
          (*v)[static_cast<std::size_t>(slot_j)] = value;
          ++slot_j;
        }
      }
    }
  }
  bool workspace(cusparseSpMatDescr_t mat, cusparseDnVecDescr_t in, cusparseDnVecDescr_t out,
                 void** ws) const {
    std::size_t bytes = 0;
    if (cusparseSpMV_bufferSize(cs_, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, mat, in, &kZero,
                                out, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                &bytes) != CUSPARSE_STATUS_SUCCESS)
      return false;
    if (bytes == 0) return true;
    return cudaMalloc(ws, bytes) == cudaSuccess;
  }
  bool spmv(cusparseSpMatDescr_t mat, cusparseDnVecDescr_t in, double* in_ptr,
            cusparseDnVecDescr_t out, double* out_ptr, void* ws) const {
    if (cusparseDnVecSetValues(in, in_ptr) != CUSPARSE_STATUS_SUCCESS ||
        cusparseDnVecSetValues(out, out_ptr) != CUSPARSE_STATUS_SUCCESS)
      return false;
    return cusparseSpMV(cs_, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, mat, in, &kZero, out,
                        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, ws) == CUSPARSE_STATUS_SUCCESS;
  }
  bool copy(const double* src, double* dst, int count) const {
    if (count <= 0) return true;
    return cudaMemcpyAsync(dst, src, static_cast<std::size_t>(count) * sizeof(double),
                           cudaMemcpyDeviceToDevice, stream_) == cudaSuccess;
  }
  /// ||a - b||^2 over n and ||c - d||^2 over m, fixed order, to the host.
  bool two_squared_distances(const double* a, const double* b, const double* c, const double* d,
                             double* first, double* second) {
    k_sqdiff_partials<<<bn_, kThreads, 0, stream_>>>(n_, a, b, d_partials_);
    if (m_ > 0) {
      k_sqdiff_partials<<<bm_, kThreads, 0, stream_>>>(m_, c, d, d_partials_ + bn_);
    } else if (cudaMemsetAsync(d_partials_ + bn_, 0, sizeof(double), stream_) != cudaSuccess) {
      return false;
    }
    k_finalize_two<<<1, kThreads, 0, stream_>>>(d_partials_, bn_, bm_, d_scalars_);
    if (cudaMemcpyAsync(h_scalars_, d_scalars_, 2 * sizeof(double), cudaMemcpyDeviceToHost,
                        stream_) != cudaSuccess ||
        cudaStreamSynchronize(stream_) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      return false;
    *first = h_scalars_[0];
    *second = h_scalars_[1];
    return true;
  }

  int n_ = 0, m_ = 0, bn_ = 1, bm_ = 1;
  cudaStream_t stream_{};
  cusparseHandle_t cs_{};
  cusparseSpMatDescr_t mat_a_{}, mat_at_{}, mat_q_{};
  cusparseDnVecDescr_t vn_in_{}, vn_out_{}, vm_in_{}, vm_out_{};
  void *ws_a_{}, *ws_at_{}, *ws_q_{};
  double *d_x_{}, *d_y_{}, *d_xn_{}, *d_yn_{}, *d_ext_{}, *d_qx_{}, *d_aty_{}, *d_ax_{};
  double *d_cost_{}, *d_clo_{}, *d_chi_{}, *d_rlo_{}, *d_rhi_{};
  double *d_xanchor_{}, *d_yanchor_{}, *d_xrestart_{}, *d_yrestart_{};
  double *d_partials_{}, *d_scalars_{}, *h_scalars_{};
  int *d_arp_{}, *d_aci_{}, *d_trp_{}, *d_tci_{}, *d_qrp_{}, *d_qci_{};
  double *d_av_{}, *d_tv_{}, *d_qv_{};
};

}  // namespace

std::unique_ptr<qp::QpOperator> make_qp_device_operator(const Model& model,
                                                        std::string* reason) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    if (reason) *reason = "no CUDA device";
    (void)cudaGetLastError();
    return nullptr;
  }
  auto op = std::make_unique<DeviceOperator>();
  if (!op->init(model, reason)) {
    (void)cudaGetLastError();
    return nullptr;
  }
  return op;
}

}  // namespace sankhya::gpu
