// SPDX-License-Identifier: Apache-2.0
// GPU activity-based bound propagation (#510). See domain_prop.hpp and, for the algorithm and
// its citation, src/mip/domain_propagation.hpp.
//
// Two kernels per round, both reading the bounds the round started with:
//   k_rows   one thread per row: the row's min and max activity summed over its entries in
//            CSR order, the infeasibility test, then per entry the implied bound, folded into
//            the column's candidate by an atomic min or max;
//   k_cols   one thread per column: take the candidate if it tightens by more than
//            kPropagationMinChange, round an integer column inward, test the box.
// Every product, sum, difference and quotient is an explicit round-to-nearest intrinsic
// (__dmul_rn, __dadd_rn, __dsub_rn, __ddiv_rn), so nvcc cannot contract a*b + c into a fused
// multiply-add: the CPU reference rounds each step, and the two must agree bit for bit. min
// and max do not depend on the order the rows arrive in, so the atomics add no variation.

#include "domain_prop.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#include "../mip/domain_propagation.hpp"
#include "device.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::gpu {
namespace {

constexpr int kBlock = 256;

__device__ void atomic_min_double(double* address, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(__double_as_longlong(value)));
    if (old == assumed) break;
  }
}

__device__ void atomic_max_double(double* address, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (value > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, static_cast<unsigned long long>(__double_as_longlong(value)));
    if (old == assumed) break;
  }
}

__global__ void k_reset(double* cand_lo, double* cand_hi, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  cand_lo[j] = -INFINITY;
  cand_hi[j] = INFINITY;
}

__global__ void k_rows(const int* __restrict__ start, const int* __restrict__ column,
                       const double* __restrict__ value, const double* __restrict__ row_lo,
                       const double* __restrict__ row_hi, const double* __restrict__ lo,
                       const double* __restrict__ hi, double* cand_lo, double* cand_hi,
                       int* infeasible, int m, double feasibility) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  double min_act = 0.0;
  double max_act = 0.0;
  int min_inf = 0;
  int max_inf = 0;
  for (int k = start[i]; k < start[i + 1]; ++k) {
    const double a = value[k];
    const int j = column[k];
    const double low = a > 0.0 ? __dmul_rn(a, lo[j]) : __dmul_rn(a, hi[j]);
    const double high = a > 0.0 ? __dmul_rn(a, hi[j]) : __dmul_rn(a, lo[j]);
    if (isinf(low)) {
      ++min_inf;
    } else {
      min_act = __dadd_rn(min_act, low);
    }
    if (isinf(high)) {
      ++max_inf;
    } else {
      max_act = __dadd_rn(max_act, high);
    }
  }
  const double ru = row_hi[i];
  const double rl = row_lo[i];
  if ((min_inf == 0 && isfinite(ru) && min_act > __dadd_rn(ru, feasibility)) ||
      (max_inf == 0 && isfinite(rl) && max_act < __dsub_rn(rl, feasibility))) {
    atomicExch(infeasible, 1);
    return;
  }
  for (int k = start[i]; k < start[i + 1]; ++k) {
    const double a = value[k];
    if (a == 0.0) continue;
    const int j = column[k];
    const double own_low = a > 0.0 ? __dmul_rn(a, lo[j]) : __dmul_rn(a, hi[j]);
    const double own_high = a > 0.0 ? __dmul_rn(a, hi[j]) : __dmul_rn(a, lo[j]);
    if (min_inf == 0 && isfinite(ru) && isfinite(own_low)) {
      const double implied = __ddiv_rn(__dsub_rn(ru, __dsub_rn(min_act, own_low)), a);
      if (a > 0.0) {
        atomic_min_double(&cand_hi[j], implied);
      } else {
        atomic_max_double(&cand_lo[j], implied);
      }
    }
    if (max_inf == 0 && isfinite(rl) && isfinite(own_high)) {
      const double implied = __ddiv_rn(__dsub_rn(rl, __dsub_rn(max_act, own_high)), a);
      if (a > 0.0) {
        atomic_max_double(&cand_lo[j], implied);
      } else {
        atomic_min_double(&cand_hi[j], implied);
      }
    }
  }
}

__global__ void k_cols(double* lo, double* hi, const double* __restrict__ cand_lo,
                       const double* __restrict__ cand_hi, const char* __restrict__ integer,
                       int* infeasible, unsigned long long* changed, int n, double min_change,
                       double integrality, double feasibility) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  double new_lo = lo[j];
  double new_hi = hi[j];
  if (cand_hi[j] < __dsub_rn(new_hi, min_change)) new_hi = cand_hi[j];
  if (cand_lo[j] > __dadd_rn(new_lo, min_change)) new_lo = cand_lo[j];
  if (integer[j] != 0) {
    if (isfinite(new_hi)) new_hi = floor(__dadd_rn(new_hi, integrality));
    if (isfinite(new_lo)) new_lo = ceil(__dsub_rn(new_lo, integrality));
  }
  if (new_lo > __dadd_rn(new_hi, feasibility)) {
    atomicExch(infeasible, 1);
    return;
  }
  const unsigned long long moved = static_cast<unsigned long long>(new_lo != lo[j]) +
                                   static_cast<unsigned long long>(new_hi != hi[j]);
  if (moved != 0) atomicAdd(changed, moved);
  lo[j] = new_lo;
  hi[j] = new_hi;
}

template <class T>
bool allocate(T** device, std::size_t count) {
  return cudaMalloc(reinterpret_cast<void**>(device), (count == 0 ? 1 : count) * sizeof(T)) ==
         cudaSuccess;
}

template <class T>
bool upload(T* device, const T* host, std::size_t count) {
  return count == 0 ||
         cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

struct DeviceBuffers {
  int *start{}, *column{}, *infeasible{};
  double *value{}, *row_lo{}, *row_hi{}, *lo{}, *hi{}, *cand_lo{}, *cand_hi{};
  char* integer{};
  unsigned long long* changed{};
  DeviceBuffers() = default;
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  ~DeviceBuffers() { release(); }
  void release() {
    // cudaFree(nullptr) is a no-op, so a partly allocated or released set frees cleanly.
    for (void* p : {static_cast<void*>(start), static_cast<void*>(column),
                    static_cast<void*>(infeasible), static_cast<void*>(value),
                    static_cast<void*>(row_lo), static_cast<void*>(row_hi),
                    static_cast<void*>(lo), static_cast<void*>(hi),
                    static_cast<void*>(cand_lo), static_cast<void*>(cand_hi),
                    static_cast<void*>(integer), static_cast<void*>(changed)}) {
      cudaFree(p);
    }
    start = column = infeasible = nullptr;
    value = row_lo = row_hi = lo = hi = cand_lo = cand_hi = nullptr;
    integer = nullptr;
    changed = nullptr;
  }
};

/// Seconds on the host's steady clock since `*mark`, moving the mark to now.
double lap(std::chrono::steady_clock::time_point* mark) {
  const auto now = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(now - *mark).count();
  *mark = now;
  return seconds;
}

/// The rounds on the device, from uploaded buffers. False when a CUDA call failed. Every
/// round ends in a synchronous read-back, so the host clock around this covers the kernels.
bool run_rounds(const DeviceBuffers& d, int m, int n, int rounds_limit, double integrality,
                PropResult* result) {
  const int row_blocks = (m + kBlock - 1) / kBlock;
  const int col_blocks = (n + kBlock - 1) / kBlock;
  for (int round = 0; round < rounds_limit; ++round) {
    ++result->rounds;
    if (n > 0) k_reset<<<col_blocks, kBlock>>>(d.cand_lo, d.cand_hi, n);
    if (m > 0) {
      k_rows<<<row_blocks, kBlock>>>(d.start, d.column, d.value, d.row_lo, d.row_hi, d.lo, d.hi,
                                     d.cand_lo, d.cand_hi, d.infeasible, m,
                                     tol::kPrimalFeasibility);
    }
    int infeasible = 0;
    if (cudaMemcpy(&infeasible, d.infeasible, sizeof(int), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return false;
    }
    if (infeasible != 0) {
      result->infeasible = true;
      return true;
    }
    if (cudaMemset(d.changed, 0, sizeof(unsigned long long)) != cudaSuccess) return false;
    if (n > 0) {
      k_cols<<<col_blocks, kBlock>>>(d.lo, d.hi, d.cand_lo, d.cand_hi, d.integer, d.infeasible,
                                     d.changed, n, tol::kPropagationMinChange, integrality,
                                     tol::kPrimalFeasibility);
    }
    unsigned long long changed = 0;
    if (cudaMemcpy(&changed, d.changed, sizeof(changed), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&infeasible, d.infeasible, sizeof(int), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      return false;
    }
    if (infeasible != 0) {
      result->infeasible = true;
      return true;
    }
    result->tightened += static_cast<long long>(changed);
    if (changed == 0) break;
  }
  return true;
}

}  // namespace

PropResult propagate_bounds(const Model& model, const std::vector<double>& col_lb,
                            const std::vector<double>& col_ub, int rounds_limit,
                            double integrality) {
  PropResult result;
  result.col_lb = col_lb;
  result.col_ub = col_ub;
  auto mark = std::chrono::steady_clock::now();
  // The CUDA runtime makes the process's context lazily, on the first call that needs one.
  // cudaFree(nullptr) is such a call and does nothing else, so that cost is timed here and
  // not inside the first allocation. It is paid once per process: a later call finds the
  // context already made and this phase drops to the probe alone.
  const bool device = device_available(nullptr) && cudaFree(nullptr) == cudaSuccess;
  result.phases.context = lap(&mark);
  if (!device) return result;
  const mip::RowMajor rows = mip::row_major(model);
  const int m = static_cast<int>(rows.rows);
  const int n = static_cast<int>(rows.cols);
  std::vector<int> start(rows.start.begin(), rows.start.end());
  std::vector<int> column(rows.column.begin(), rows.column.end());
  std::vector<char> integer(static_cast<std::size_t>(n), 0);
  for (int j = 0; j < n; ++j) {
    integer[static_cast<std::size_t>(j)] =
        model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger ? 1 : 0;
  }
  result.phases.host = lap(&mark);
  DeviceBuffers d;
  const auto cols = static_cast<std::size_t>(n);
  const bool allocated =
      allocate(&d.start, start.size()) && allocate(&d.column, column.size()) &&
      allocate(&d.value, rows.value.size()) && allocate(&d.row_lo, model.row_lower.size()) &&
      allocate(&d.row_hi, model.row_upper.size()) && allocate(&d.lo, col_lb.size()) &&
      allocate(&d.hi, col_ub.size()) && allocate(&d.integer, integer.size()) &&
      allocate(&d.infeasible, 1) && allocate(&d.changed, 1) && allocate(&d.cand_lo, cols) &&
      allocate(&d.cand_hi, cols);
  result.phases.allocate = lap(&mark);
  if (!allocated) return result;
  const int zero = 0;
  const unsigned long long zero_count = 0;
  const bool uploaded =
      upload(d.start, start.data(), start.size()) &&
      upload(d.column, column.data(), column.size()) &&
      upload(d.value, rows.value.data(), rows.value.size()) &&
      upload(d.row_lo, model.row_lower.data(), model.row_lower.size()) &&
      upload(d.row_hi, model.row_upper.data(), model.row_upper.size()) &&
      upload(d.lo, col_lb.data(), col_lb.size()) && upload(d.hi, col_ub.data(), col_ub.size()) &&
      upload(d.integer, integer.data(), integer.size()) && upload(d.infeasible, &zero, 1) &&
      upload(d.changed, &zero_count, 1);
  result.phases.upload = lap(&mark);
  if (!uploaded) return result;
  const bool finished = run_rounds(d, m, n, rounds_limit, integrality, &result);
  result.phases.rounds = lap(&mark);
  if (!finished) return result;
  const bool downloaded =
      result.infeasible || n == 0 ||
      (cudaMemcpy(result.col_lb.data(), d.lo, cols * sizeof(double), cudaMemcpyDeviceToHost) ==
           cudaSuccess &&
       cudaMemcpy(result.col_ub.data(), d.hi, cols * sizeof(double), cudaMemcpyDeviceToHost) ==
           cudaSuccess);
  result.phases.download = lap(&mark);
  d.release();
  result.phases.release = lap(&mark);
  if (!downloaded) {
    result.col_lb = col_lb;
    result.col_ub = col_ub;
    return result;
  }
  result.ran = true;
  return result;
}

}  // namespace sankhya::gpu
