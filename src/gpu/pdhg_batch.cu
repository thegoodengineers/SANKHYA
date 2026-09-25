// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the batched PDHG's CUDA backend (#520). See pdhg_batch.hpp, and for the method
// src/pdhg/batch_pdhg.hpp.
//
// Per iteration two kernels, one thread per (column, LP) and one per (row, LP):
//   k_primal  aty = (A^T y)_j summed over column j in ascending row, then the projected
//             gradient step, the extrapolated point xbar = 2 x+ - x and the running sum;
//   k_dual    (A xbar)_i summed over row i in ascending column, then the dual prox step and
//             the running sum.
// Threads are numbered LP-fastest (index j * K + k), so the K threads of one column or row
// read the same matrix entries and neighbouring vector entries: the sparse matrix times a
// dense n x K block, with the matrix read once per warp rather than once per LP. This is
// the SpMM of the issue, written as a kernel rather than a cuSPARSE call because cuSPARSE
// does not promise a summation order, and the CPU reference must be matched bit for bit.
//
// Every product, sum, difference and quotient is an explicit round-to-nearest intrinsic
// (__dmul_rn, __dadd_rn, __dsub_rn, __ddiv_rn), so nvcc cannot contract a*b + c into a fused
// multiply-add; src/pdhg/batch_pdhg_cpu.cpp performs the same operations in the same order
// with contraction off. The restart statistics are partial sums over fixed chunks of
// kBatchPdhgChunk rows or columns, one thread per (iterate, chunk, LP), which the host adds
// in chunk order for either backend.

#include "pdhg_batch.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "device.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::gpu {
namespace {

constexpr int kBlock = 256;

__device__ inline double clamp_d(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void k_primal(int n, int K, const int* __restrict__ start, const int* __restrict__ index,
                         const double* __restrict__ value, const double* __restrict__ cost,
                         const double* __restrict__ lower, const double* __restrict__ upper,
                         const double* __restrict__ tau, const double* __restrict__ y,
                         double* __restrict__ x, double* __restrict__ xbar,
                         double* __restrict__ x_sum) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<long long>(n) * K) return;
  const int j = static_cast<int>(t / K);
  const int k = static_cast<int>(t % K);
  double aty = 0.0;
  for (int p = start[j]; p < start[j + 1]; ++p) {
    aty = __dadd_rn(aty, __dmul_rn(value[p], y[static_cast<long long>(index[p]) * K + k]));
  }
  const double g = __dsub_rn(cost[j], aty);
  const double v = __dsub_rn(x[t], __dmul_rn(tau[k], g));
  const double next = clamp_d(v, lower[t], upper[t]);
  xbar[t] = __dsub_rn(__dmul_rn(2.0, next), x[t]);
  x[t] = next;
  x_sum[t] = __dadd_rn(x_sum[t], next);
}

__global__ void k_dual(int m, int K, const int* __restrict__ start, const int* __restrict__ index,
                       const double* __restrict__ value, const double* __restrict__ row_lower,
                       const double* __restrict__ row_upper, const double* __restrict__ sigma,
                       const double* __restrict__ xbar, double* __restrict__ y,
                       double* __restrict__ y_sum) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<long long>(m) * K) return;
  const int i = static_cast<int>(t / K);
  const int k = static_cast<int>(t % K);
  double ax = 0.0;
  for (int p = start[i]; p < start[i + 1]; ++p) {
    ax = __dadd_rn(ax, __dmul_rn(value[p], xbar[static_cast<long long>(index[p]) * K + k]));
  }
  const double rl = row_lower[i];
  const double ru = row_upper[i];
  const double s = sigma[k];
  const double v = __dsub_rn(y[t], __dmul_rn(s, ax));
  const double lo = isfinite(ru) ? -__dmul_rn(s, ru) : -INFINITY;
  const double hi = isfinite(rl) ? -__dmul_rn(s, rl) : INFINITY;
  const double next = __dsub_rn(v, clamp_d(v, lo, hi));
  y[t] = next;
  y_sum[t] = __dadd_rn(y_sum[t], next);
}

/// Iterate s of LP k at `index`: the current value (s = 0) or the average since the restart.
__device__ inline double point(int s, const double* value, const double* sum, const double* count,
                               long long at, int k) {
  if (s == 0 || !(count[k] > 0.0)) return value[at];
  return __ddiv_rn(sum[at], count[k]);
}

__global__ void k_row_partials(int m, int K, int chunks, int chunk, const int* __restrict__ start,
                               const int* __restrict__ index, const double* __restrict__ value,
                               const double* __restrict__ row_lower,
                               const double* __restrict__ row_upper, const double* __restrict__ x,
                               const double* __restrict__ x_sum, const double* __restrict__ y,
                               const double* __restrict__ y_sum,
                               const double* __restrict__ y_restart,
                               const double* __restrict__ count, double* __restrict__ out) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= 2LL * chunks * K) return;
  const int k = static_cast<int>(t % K);
  const int c = static_cast<int>((t / K) % chunks);
  const int s = static_cast<int>(t / (static_cast<long long>(K) * chunks));
  double q0 = 0.0;
  double q1 = 0.0;
  double q2 = 0.0;
  const int first = c * chunk;
  const int last = first + chunk < m ? first + chunk : m;
  for (int i = first; i < last; ++i) {
    double ax = 0.0;
    for (int p = start[i]; p < start[i + 1]; ++p) {
      const double xv = point(s, x, x_sum, count, static_cast<long long>(index[p]) * K + k, k);
      ax = __dadd_rn(ax, __dmul_rn(value[p], xv));
    }
    const double rl = row_lower[i];
    const double ru = row_upper[i];
    const double r = ax < rl ? __dsub_rn(rl, ax) : (ax > ru ? __dsub_rn(ax, ru) : 0.0);
    q0 = __dadd_rn(q0, __dmul_rn(r, r));
    const long long at = static_cast<long long>(i) * K + k;
    const double yv = point(s, y, y_sum, count, at, k);
    const double part = yv > 0.0 ? (isfinite(rl) ? __dmul_rn(yv, rl) : 0.0)
                                 : (yv < 0.0 ? (isfinite(ru) ? __dmul_rn(yv, ru) : 0.0) : 0.0);
    q1 = __dadd_rn(q1, part);
    const double move = __dsub_rn(yv, y_restart[at]);
    q2 = __dadd_rn(q2, __dmul_rn(move, move));
  }
  const long long base = static_cast<long long>(s) * 3;
  out[((base + 0) * chunks + c) * K + k] = q0;
  out[((base + 1) * chunks + c) * K + k] = q1;
  out[((base + 2) * chunks + c) * K + k] = q2;
}

__global__ void k_col_partials(int n, int K, int chunks, int chunk, const int* __restrict__ start,
                               const int* __restrict__ index, const double* __restrict__ value,
                               const double* __restrict__ cost, const double* __restrict__ lower,
                               const double* __restrict__ upper, const double* __restrict__ x,
                               const double* __restrict__ x_sum,
                               const double* __restrict__ x_restart,
                               const double* __restrict__ y, const double* __restrict__ y_sum,
                               const double* __restrict__ count, double* __restrict__ out) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= 2LL * chunks * K) return;
  const int k = static_cast<int>(t % K);
  const int c = static_cast<int>((t / K) % chunks);
  const int s = static_cast<int>(t / (static_cast<long long>(K) * chunks));
  double q0 = 0.0;
  double q1 = 0.0;
  double q2 = 0.0;
  double q3 = 0.0;
  const int first = c * chunk;
  const int last = first + chunk < n ? first + chunk : n;
  for (int j = first; j < last; ++j) {
    double aty = 0.0;
    for (int p = start[j]; p < start[j + 1]; ++p) {
      const double yv = point(s, y, y_sum, count, static_cast<long long>(index[p]) * K + k, k);
      aty = __dadd_rn(aty, __dmul_rn(value[p], yv));
    }
    const long long at = static_cast<long long>(j) * K + k;
    const double g = __dsub_rn(cost[j], aty);
    const double lb = lower[at];
    const double ub = upper[at];
    const bool lf = isfinite(lb);
    const bool uf = isfinite(ub);
    const double residual =
        lf && uf ? 0.0 : (lf ? (g < 0.0 ? g : 0.0) : (uf ? (g > 0.0 ? g : 0.0) : g));
    q0 = __dadd_rn(q0, __dmul_rn(residual, residual));
    const double xv = point(s, x, x_sum, count, at, k);
    q1 = __dadd_rn(q1, __dmul_rn(cost[j], xv));
    const double part = g > 0.0 ? (lf ? __dmul_rn(g, lb) : 0.0)
                                : (g < 0.0 ? (uf ? __dmul_rn(g, ub) : 0.0) : 0.0);
    q2 = __dadd_rn(q2, part);
    const double move = __dsub_rn(xv, x_restart[at]);
    q3 = __dadd_rn(q3, __dmul_rn(move, move));
  }
  const long long base = static_cast<long long>(s) * 4;
  out[((base + 0) * chunks + c) * K + k] = q0;
  out[((base + 1) * chunks + c) * K + k] = q1;
  out[((base + 2) * chunks + c) * K + k] = q2;
  out[((base + 3) * chunks + c) * K + k] = q3;
}

/// A restart, one thread per (entry, LP): kToAverage takes the average (sum / count) as the
/// new point, and every restarted LP takes its point as the restart point and empties its sum.
__global__ void k_restart(long long size, int K, const signed char* __restrict__ action,
                          const double* __restrict__ count, double* __restrict__ value,
                          double* __restrict__ sum, double* __restrict__ anchor) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= size * K) return;
  const int k = static_cast<int>(t % K);
  if (action[k] == 0) return;
  if (action[k] == 2 && count[k] > 0.0) value[t] = __ddiv_rn(sum[t], count[k]);
  anchor[t] = value[t];
  sum[t] = 0.0;
}

unsigned grid(long long threads) {
  return static_cast<unsigned>((threads + kBlock - 1) / kBlock);
}

/// A device array freed with its owner.
template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  bool allocate(std::size_t count) {
    size_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&ptr_), (count == 0 ? 1 : count) * sizeof(T)) ==
           cudaSuccess;
  }
  bool upload(const std::vector<T>& host) {
    if (!allocate(host.size())) return false;
    return host.empty() || cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                                      cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool download(std::vector<T>* host) const {
    host->resize(size_);
    return size_ == 0 || cudaMemcpy(host->data(), ptr_, size_ * sizeof(T),
                                    cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  bool zero() { return cudaMemset(ptr_, 0, (size_ == 0 ? 1 : size_) * sizeof(T)) == cudaSuccess; }
  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

class DeviceBatchBackend final : public pdhg::BatchBackend {
 public:
  explicit DeviceBatchBackend(const pdhg::BatchData& data)
      : n_(data.cols), m_(data.rows), k_(data.count), count_(static_cast<std::size_t>(k_), 0.0) {}

  bool setup(const pdhg::BatchData& data, std::string* why) {
    const bool ok = row_start_.upload(data.row_start) && row_index_.upload(data.row_index) &&
                    row_value_.upload(data.row_value) && col_start_.upload(data.col_start) &&
                    col_index_.upload(data.col_index) && col_value_.upload(data.col_value) &&
                    cost_.upload(data.cost) && row_lower_.upload(data.row_lower) &&
                    row_upper_.upload(data.row_upper) && col_lower_.upload(data.col_lower) &&
                    col_upper_.upload(data.col_upper) && x_.upload(data.x0) &&
                    x_restart_.upload(data.x0) && y_.upload(data.y0) &&
                    y_restart_.upload(data.y0) && x_sum_.allocate(data.x0.size()) &&
                    x_sum_.zero() && y_sum_.allocate(data.y0.size()) && y_sum_.zero() &&
                    xbar_.allocate(data.x0.size()) && tau_.allocate(count_.size()) &&
                    sigma_.allocate(count_.size()) && count_device_.allocate(count_.size()) &&
                    action_.allocate(count_.size()) &&
                    row_out_.allocate(static_cast<std::size_t>(2 * pdhg::kBatchRowQuantities) *
                                      static_cast<std::size_t>(pdhg::batch_chunks(m_)) *
                                      count_.size()) &&
                    col_out_.allocate(static_cast<std::size_t>(2 * pdhg::kBatchColQuantities) *
                                      static_cast<std::size_t>(pdhg::batch_chunks(n_)) *
                                      count_.size());
    if (!ok) {
      *why = std::string("allocation or upload failed: ") + cudaGetErrorString(cudaGetLastError());
      healthy_ = false;
    }
    return ok;
  }

  void iterate(Count iterations, const std::vector<double>& tau,
               const std::vector<double>& sigma) override {
    if (!healthy_) return;
    healthy_ = cudaMemcpy(tau_.get(), tau.data(), tau.size() * sizeof(double),
                          cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(sigma_.get(), sigma.data(), sigma.size() * sizeof(double),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    const long long cols = static_cast<long long>(n_) * k_;
    const long long rows = static_cast<long long>(m_) * k_;
    for (Count t = 0; t < iterations && healthy_; ++t) {
      k_primal<<<grid(cols), kBlock>>>(n_, k_, col_start_.get(), col_index_.get(),
                                       col_value_.get(), cost_.get(), col_lower_.get(),
                                       col_upper_.get(), tau_.get(), y_.get(), x_.get(),
                                       xbar_.get(), x_sum_.get());
      if (rows > 0) {
        k_dual<<<grid(rows), kBlock>>>(m_, k_, row_start_.get(), row_index_.get(),
                                       row_value_.get(), row_lower_.get(), row_upper_.get(),
                                       sigma_.get(), xbar_.get(), y_.get(), y_sum_.get());
      }
    }
    for (double& c : count_) c = c + static_cast<double>(iterations);
    healthy_ = healthy_ && cudaGetLastError() == cudaSuccess;
  }

  void measure(pdhg::BatchPartials* partials) override {
    if (!healthy_ || !upload_count()) return;
    const int row_chunks = pdhg::batch_chunks(m_);
    const int col_chunks = pdhg::batch_chunks(n_);
    const int chunk = tol::kBatchPdhgChunk;
    if (row_chunks > 0) {
      k_row_partials<<<grid(2LL * row_chunks * k_), kBlock>>>(
          m_, k_, row_chunks, chunk, row_start_.get(), row_index_.get(), row_value_.get(),
          row_lower_.get(), row_upper_.get(), x_.get(), x_sum_.get(), y_.get(), y_sum_.get(),
          y_restart_.get(), count_device_.get(), row_out_.get());
    }
    if (col_chunks > 0) {
      k_col_partials<<<grid(2LL * col_chunks * k_), kBlock>>>(
          n_, k_, col_chunks, chunk, col_start_.get(), col_index_.get(), col_value_.get(),
          cost_.get(), col_lower_.get(), col_upper_.get(), x_.get(), x_sum_.get(),
          x_restart_.get(), y_.get(), y_sum_.get(), count_device_.get(), col_out_.get());
    }
    healthy_ = cudaGetLastError() == cudaSuccess && row_out_.download(&partials->rows) &&
               col_out_.download(&partials->cols);
  }

  void restart(const std::vector<pdhg::BatchRestart>& action) override {
    if (!healthy_ || !upload_count()) return;
    std::vector<signed char> codes(action.size());
    bool any = false;
    for (std::size_t k = 0; k < action.size(); ++k) {
      codes[k] = static_cast<signed char>(action[k]);
      any = any || action[k] != pdhg::BatchRestart::kNone;
    }
    if (!any) return;
    healthy_ = cudaMemcpy(action_.get(), codes.data(), codes.size(), cudaMemcpyHostToDevice) ==
               cudaSuccess;
    if (!healthy_) return;
    k_restart<<<grid(static_cast<long long>(n_) * k_), kBlock>>>(
        n_, k_, action_.get(), count_device_.get(), x_.get(), x_sum_.get(), x_restart_.get());
    if (m_ > 0) {
      k_restart<<<grid(static_cast<long long>(m_) * k_), kBlock>>>(
          m_, k_, action_.get(), count_device_.get(), y_.get(), y_sum_.get(), y_restart_.get());
    }
    healthy_ = cudaGetLastError() == cudaSuccess;
    for (std::size_t k = 0; k < action.size(); ++k) {
      if (action[k] != pdhg::BatchRestart::kNone) count_[k] = 0.0;
    }
  }

  void read_duals(std::vector<double>* current, std::vector<double>* average) override {
    std::vector<double> sum;
    healthy_ = healthy_ && y_.download(current) && y_sum_.download(&sum) &&
               cudaDeviceSynchronize() == cudaSuccess;
    if (!healthy_) return;
    // The host divides as k_restart and the CPU reference do: IEEE division either way.
    average->resize(current->size());
    const auto k_count = static_cast<std::size_t>(k_);
    for (std::size_t at = 0; at < current->size(); ++at) {
      const double c = count_[at % k_count];
      (*average)[at] = c > 0.0 ? sum[at] / c : (*current)[at];
    }
  }

  [[nodiscard]] bool healthy() const override { return healthy_; }

 private:
  bool upload_count() {
    healthy_ = healthy_ && cudaMemcpy(count_device_.get(), count_.data(),
                                      count_.size() * sizeof(double),
                                      cudaMemcpyHostToDevice) == cudaSuccess;
    return healthy_;
  }

  int n_;
  int m_;
  int k_;
  std::vector<double> count_;  ///< iterations summed since each LP's restart, host copy
  bool healthy_ = true;
  DeviceArray<int> row_start_, row_index_, col_start_, col_index_;
  DeviceArray<double> row_value_, col_value_, cost_, row_lower_, row_upper_, col_lower_,
      col_upper_;
  DeviceArray<double> x_, x_sum_, x_restart_, xbar_, y_, y_sum_, y_restart_;
  DeviceArray<double> tau_, sigma_, count_device_, row_out_, col_out_;
  DeviceArray<signed char> action_;
};

}  // namespace

std::unique_ptr<pdhg::BatchBackend> make_device_batch_backend(const pdhg::BatchData& data,
                                                              std::string* why) {
  std::string description;
  if (!device_available(&description)) {
    *why = description.empty() ? "no CUDA device" : description;
    return nullptr;
  }
  auto backend = std::make_unique<DeviceBatchBackend>(data);
  if (!backend->setup(data, why)) return nullptr;
  return backend;
}

}  // namespace sankhya::gpu
