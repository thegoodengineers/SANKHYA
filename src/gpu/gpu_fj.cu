// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU Feasibility Jump (#508). See gpu_fj.hpp, and gpu_fj_device.cuh for the
// method, its citations and the determinism argument. This file is the search kernel and the
// host side: the upload, the launches, and the verification of every point on the host.

#include "gpu_fj.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "device.hpp"
#include "gpu_fj_device.cuh"
#include "sankhya/tolerances.hpp"

namespace sankhya::gpu {
namespace {

using fj::BlockState;
using fj::kThreads;
using fj::kWarps;
using fj::Problem;
using fj::Slots;

/// Recompute block b's activities from scratch, row by row in CSR order, and rebuild its
/// violated list in row order. Every thread of the block calls it.
__device__ void recompute(const Problem& p, const double* x, double* act, int* violated,
                          int* violated_at, BlockState* st, int* warp_counts) {
  for (int i = static_cast<int>(threadIdx.x); i < p.m; i += kThreads) {
    double sum = 0.0;
    for (int k = p.row_start[i]; k < p.row_start[i + 1]; ++k) sum += p.row_val[k] * x[p.row_col[k]];
    act[i] = sum;
  }
  __syncthreads();
  int count = 0;
  for (int base = 0; base < p.m; base += kThreads) {
    const int i = base + static_cast<int>(threadIdx.x);
    const bool flag = i < p.m && fj::violation(act[i], p.row_lo[i], p.row_hi[i]) > p.feasibility;
    int position = -1;
    const int total = fj::block_compact(flag, i, violated + count, &position, warp_counts);
    if (i < p.m) violated_at[i] = flag ? count + position : -1;
    count += total;
  }
  if (threadIdx.x == 0) {
    st->violated_count = count;
    st->work += p.nnz + p.m;
  }
  __syncthreads();
}

/// Move column `col` to `value`: the activities of its rows at once, then the violated list
/// for the rows whose state changed, in column order, by thread 0.
__device__ void apply_move(const Problem& p, double* x, double* act, int* violated,
                           int* violated_at, BlockState* st, int col, double value, int* changed,
                           int* warp_counts) {
  const double step = value - x[col];
  const int begin = p.col_start[col];
  const int end = p.col_start[col + 1];
  for (int base = begin; base < end; base += kThreads) {
    const int k = base + static_cast<int>(threadIdx.x);
    bool flip = false;
    int row = -1;
    if (k < end) {
      row = p.col_row[k];
      act[row] += p.col_val[k] * step;  // a column's rows are distinct: no two threads race
      const bool now = fj::violation(act[row], p.row_lo[row], p.row_hi[row]) > p.feasibility;
      flip = now != (violated_at[row] >= 0);
    }
    int position = -1;
    const int total = fj::block_compact(flip, row, changed, &position, warp_counts);
    if (threadIdx.x == 0) {
      for (int t = 0; t < total; ++t) {
        const int i = changed[t];
        const int at = violated_at[i];
        if (at < 0) {
          violated_at[i] = st->violated_count;
          violated[st->violated_count++] = i;
        } else {
          const int last = violated[--st->violated_count];
          violated[at] = last;
          violated_at[last] = at;
          violated_at[i] = -1;
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    x[col] = value;
    st->work += end - begin + 1;
    ++st->moves;
    st->checked = 0;
  }
}

/// Up to `steps` moves or weight updates of every block's search. The state lives in global
/// memory between launches, so the launch boundary never changes a search's path.
__global__ void __launch_bounds__(kThreads)
    k_search(Problem p, Slots slots, int steps) {
  const int b = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const auto n = static_cast<std::size_t>(p.n);
  const auto m = static_cast<std::size_t>(p.m);
  double* x = slots.x + b * n;
  double* best_x = slots.best_x + b * n;
  double* act = slots.activity + b * m;
  double* weight = slots.weight + b * m;
  int* violated = slots.violated + b * m;
  int* violated_at = slots.violated_at + b * m;

  __shared__ BlockState st;
  __shared__ int warp_counts[kWarps];
  __shared__ int changed[kThreads];
  __shared__ fj::Jump warp_jump[kWarps];
  __shared__ long long warp_work[kWarps];
  __shared__ double reduce[kWarps];
  __shared__ fj::Jump chosen;
  if (tid == 0) st = slots.state[b];
  __syncthreads();
  const unsigned long long block_seed = fj::mix(p.seed + static_cast<unsigned long long>(b));

  for (int s = 0; s < steps; ++s) {
    if (st.done != 0 || st.work >= p.work_limit) break;
    if (st.violated_count == 0 && st.checked == 0) {
      // Feasible by the running activities: re-measure every row before believing it.
      recompute(p, x, act, violated, violated_at, &st, warp_counts);
      if (tid == 0) st.checked = 1;
      __syncthreads();
      if (st.violated_count > 0) continue;
      double part = 0.0;
      for (int j = tid; j < p.n; j += kThreads) part += p.cost[j] * x[j];
      part = fj::warp_sum(part);
      if (lane == 0) reduce[warp] = part;
      __syncthreads();
      double objective = 0.0;
      for (int w = 0; w < kWarps; ++w) objective += reduce[w];
      const bool better =
          st.found == 0 ||
          objective < st.best_objective - p.min_score * fmax(1.0, fabs(st.best_objective));
      if (better) {
        for (int j = tid; j < p.n; j += kThreads) best_x[j] = x[j];
      }
      __syncthreads();
      if (tid == 0) {
        if (better) {
          st.best_objective = objective;
          ++st.found;
        }
        if (p.n_costed == 0) st.done = 1;  // nothing left to improve
      }
      __syncthreads();
      if (st.done != 0) break;
    }

    // Sample and score: warp w takes samples w * spw .. w * spw + spw - 1, in that order.
    const unsigned long long step_seed = fj::mix(block_seed ^ static_cast<unsigned long long>(st.step));
    fj::Jump best{-1, 0.0, 0.0};
    long long work = 0;
    for (int c = 0; c < p.samples_per_warp; ++c) {
      const auto q = static_cast<unsigned long long>(warp * p.samples_per_warp + c);
      int j = -1;
      if (lane == 0) {
        const unsigned long long h1 = fj::mix(step_seed + 2 * q);
        const unsigned long long h2 = fj::mix(step_seed + 2 * q + 1);
        if (st.violated_count > 0) {
          const int row = violated[fj::pick(h1, st.violated_count)];
          const int length = p.row_start[row + 1] - p.row_start[row];
          if (length > 0) j = p.row_col[p.row_start[row] + fj::pick(h2, length)];
        } else {
          j = p.costed[fj::pick(h1, p.n_costed)];
        }
      }
      j = __shfl_sync(fj::kFullMask, j, 0);
      ++work;
      if (j < 0) continue;
      const fj::Jump jump = fj::warp_best_jump(p, x, act, weight, st.objective_weight, j, lane, &work);
      if (jump.column >= 0 && (best.column < 0 || jump.score > best.score)) best = jump;
    }
    if (lane == 0) {
      warp_jump[warp] = best;
      warp_work[warp] = work;
    }
    __syncthreads();
    if (tid == 0) {
      fj::Jump pick{-1, 0.0, 0.0};
      for (int w = 0; w < kWarps; ++w) {
        const fj::Jump& jump = warp_jump[w];
        if (jump.column >= 0 && (pick.column < 0 || jump.score > pick.score)) pick = jump;
        st.work += warp_work[w];
      }
      chosen = pick;
    }
    __syncthreads();
    if (chosen.column >= 0 && chosen.score > p.min_score) {
      apply_move(p, x, act, violated, violated_at, &st, chosen.column, chosen.value, changed,
                 warp_counts);
    } else {
      // The paper's update at a local minimum: every violated row's weight gains 1; with
      // no row violated, the objective's does.
      for (int t = tid; t < st.violated_count; t += kThreads) weight[violated[t]] += 1.0;
      if (tid == 0) {
        if (st.violated_count == 0) st.objective_weight += 1.0;
        st.work += st.violated_count + 1;
        ++st.weight_updates;
      }
    }
    if (tid == 0) ++st.step;
    __syncthreads();
  }
  if (tid == 0) slots.state[b] = st;
}

__global__ void k_fill(double* data, std::size_t count, double value) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) data[i] = value;
}

/// Every device buffer of one call, freed together.
class Buffers {
 public:
  Buffers() = default;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  ~Buffers() {
    for (void* p : owned_) cudaFree(p);
  }
  template <class T>
  bool allocate(T** device, std::size_t count) {
    void* p = nullptr;
    if (cudaMalloc(&p, (count == 0 ? 1 : count) * sizeof(T)) != cudaSuccess) return false;
    owned_.push_back(p);
    *device = static_cast<T*>(p);
    return true;
  }
  template <class T>
  bool upload(T** device, const std::vector<T>& host) {
    if (!allocate(device, host.size())) return false;
    return host.empty() || cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
                                      cudaMemcpyHostToDevice) == cudaSuccess;
  }

 private:
  std::vector<void*> owned_;
};

double bound(double v) {
  if (is_finite_bound(v)) return v;
  return v > 0.0 ? std::numeric_limits<double>::infinity()
                 : -std::numeric_limits<double>::infinity();
}

double objective_of(const std::vector<double>& cost, const std::vector<double>& x) {
  double value = 0.0;
  for (std::size_t j = 0; j < x.size(); ++j) value += cost[j] * x[j];
  return value;
}

}  // namespace

FjDeviceResult feasibility_jump(const Model& model, const std::vector<double>& start,
                                const mip::FeasibilityJumpSettings& settings, int restarts) {
  const auto clock = std::chrono::steady_clock::now();
  FjDeviceResult result;
  const auto finish = [&result, clock]() -> FjDeviceResult {
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count();
    return std::move(result);
  };
  if (!device_available(&result.reason)) return finish();
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const Index nnz = model.num_nonzeros();
  if (n >= INT_MAX || m >= INT_MAX || nnz >= INT_MAX) {
    result.reason = "the model needs 64-bit indices";
    return finish();
  }

  // The search's own copy of the model, as the CPU reference's set_up() builds it.
  const auto un = static_cast<std::size_t>(n);
  const auto um = static_cast<std::size_t>(m);
  const double sense = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  std::vector<double> lo(un), hi(un), cost(un), x(un);
  std::vector<unsigned char> integer(un);
  std::vector<int> costed;
  for (std::size_t j = 0; j < un; ++j) {
    double l = model.col_lower[j];
    double h = model.col_upper[j];
    integer[j] = model.col_type[j] == VarType::kInteger ? 1 : 0;
    if (integer[j] != 0) {
      if (is_finite_bound(l)) l = std::ceil(l - settings.integrality_tolerance);
      if (is_finite_bound(h)) h = std::floor(h + settings.integrality_tolerance);
    }
    if (std::isnan(l) || std::isnan(h) || l > h) {  // no point exists: ran, found nothing
      result.ran = true;
      return finish();
    }
    lo[j] = bound(l);
    hi[j] = bound(h);
    cost[j] = sense * model.col_cost[j];
    double v = j < start.size() && std::isfinite(start[j]) ? start[j] : 0.0;
    if (integer[j] != 0) v = std::round(v);
    x[j] = std::clamp(v, lo[j], hi[j]);
    if (settings.use_objective && cost[j] != 0.0 && l < h) costed.push_back(static_cast<int>(j));
  }
  std::vector<int> col_start(un + 1, 0), col_row, row_start(um + 1, 0), row_col;
  std::vector<double> col_val, row_val, row_lo(um), row_hi(um);
  col_row.reserve(static_cast<std::size_t>(nnz));
  col_val.reserve(static_cast<std::size_t>(nnz));
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      col_row.push_back(static_cast<int>(column.rows[k]));
      col_val.push_back(column.values[k]);
      ++row_start[static_cast<std::size_t>(column.rows[k]) + 1];
    }
    col_start[static_cast<std::size_t>(j) + 1] = static_cast<int>(col_row.size());
  }
  for (std::size_t i = 0; i < um; ++i) {
    row_start[i + 1] += row_start[i];
    row_lo[i] = bound(model.row_lower[i]);
    row_hi[i] = bound(model.row_upper[i]);
  }
  row_col.resize(col_row.size());
  row_val.resize(col_row.size());
  {
    std::vector<int> fill(row_start.begin(), row_start.end() - 1);
    for (std::size_t j = 0; j < un; ++j) {
      for (int k = col_start[j]; k < col_start[j + 1]; ++k) {
        const auto at = static_cast<std::size_t>(fill[static_cast<std::size_t>(col_row[k])]++);
        row_col[at] = static_cast<int>(j);
        row_val[at] = col_val[static_cast<std::size_t>(k)];
      }
    }
  }

  // Restarts: two per multiprocessor by default, as many as half the free memory holds.
  int sms = 1;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) sms = 1;
  std::size_t free_bytes = 0;
  if (!device_free_memory(&free_bytes, nullptr)) free_bytes = 0;
  const std::size_t per_block =
      2 * un * sizeof(double) + 2 * um * sizeof(double) + 2 * um * sizeof(int) + sizeof(BlockState);
  const auto fit = static_cast<long long>(
      static_cast<double>(free_bytes) * tol::kGpuFeasibilityJumpMemoryShare /
      static_cast<double>(per_block));
  int blocks = restarts > 0 ? restarts : sms * tol::kGpuFeasibilityJumpRestartsPerSm;
  blocks = static_cast<int>(std::min<long long>(blocks, fit));
  if (blocks < 1) {
    result.reason = "the card's free memory holds no search";
    return finish();
  }
  result.restarts = blocks;

  Buffers buffers;
  Problem p;
  p.n = static_cast<int>(n);
  p.m = static_cast<int>(m);
  p.nnz = static_cast<long long>(nnz);
  p.n_costed = static_cast<int>(costed.size());
  p.samples_per_warp = tol::kGpuFeasibilityJumpSamplesPerWarp;
  p.feasibility = settings.feasibility_tolerance;
  p.min_score = tol::kFeasibilityJumpMinScore;
  p.work_limit = static_cast<long long>(settings.work_limit);
  p.seed = settings.seed;
  const auto ub = static_cast<std::size_t>(blocks);
  int *d_col_start{}, *d_col_row{}, *d_row_start{}, *d_row_col{}, *d_costed{};
  double *d_col_val{}, *d_row_val{}, *d_row_lo{}, *d_row_hi{}, *d_lo{}, *d_hi{}, *d_cost{};
  unsigned char* d_integer{};
  Slots slots;
  bool ok = buffers.upload(&d_col_start, col_start) && buffers.upload(&d_col_row, col_row) &&
            buffers.upload(&d_col_val, col_val) && buffers.upload(&d_row_start, row_start) &&
            buffers.upload(&d_row_col, row_col) && buffers.upload(&d_row_val, row_val) &&
            buffers.upload(&d_row_lo, row_lo) && buffers.upload(&d_row_hi, row_hi) &&
            buffers.upload(&d_lo, lo) && buffers.upload(&d_hi, hi) &&
            buffers.upload(&d_cost, cost) && buffers.upload(&d_integer, integer) &&
            buffers.upload(&d_costed, costed) && buffers.allocate(&slots.x, ub * un) &&
            buffers.allocate(&slots.best_x, ub * un) &&
            buffers.allocate(&slots.activity, ub * um) &&
            buffers.allocate(&slots.weight, ub * um) &&
            buffers.allocate(&slots.violated, ub * um) &&
            buffers.allocate(&slots.violated_at, ub * um) &&
            buffers.allocate(&slots.state, ub);
  // Every search starts from `start` with every weight 1 and no row known to be violated, so
  // its first step measures every row from scratch, as the CPU reference's set_up() does.
  for (std::size_t b = 0; ok && b < ub && un > 0; ++b) {
    ok = cudaMemcpy(slots.x + b * un, x.data(), un * sizeof(double), cudaMemcpyHostToDevice) ==
         cudaSuccess;
  }
  const BlockState initial{0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<BlockState> states(ub, initial);
  ok = ok && cudaMemcpy(slots.state, states.data(), ub * sizeof(BlockState),
                        cudaMemcpyHostToDevice) == cudaSuccess;
  if (ok && ub * um > 0) {
    const std::size_t count = ub * um;
    k_fill<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(slots.weight,
                                                                                  count, 1.0);
    ok = cudaGetLastError() == cudaSuccess;
  }
  if (!ok) {
    result.reason = std::string("allocation or upload failed: ") +
                    cudaGetErrorString(cudaGetLastError());
    return finish();
  }
  p.col_start = d_col_start;
  p.col_row = d_col_row;
  p.col_val = d_col_val;
  p.row_start = d_row_start;
  p.row_col = d_row_col;
  p.row_val = d_row_val;
  p.row_lo = d_row_lo;
  p.row_hi = d_row_hi;
  p.lo = d_lo;
  p.hi = d_hi;
  p.cost = d_cost;
  p.integer = d_integer;
  p.costed = d_costed;
  result.ran = true;

  // Launch, poll, hand over. Per launch the best new point over the blocks (least objective,
  // then least block) that improves on the last one handed over is copied back and verified
  // against the original model; one the check refuses is counted and the next is tried.
  std::vector<BlockState> seen(ub, initial);
  std::vector<BlockState> now(ub);
  std::vector<double> point(un);
  double best = std::numeric_limits<double>::infinity();
  for (;;) {
    k_search<<<blocks, kThreads>>>(p, slots, tol::kGpuFeasibilityJumpLaunchSteps);
    ++result.launches;
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(now.data(), slots.state, ub * sizeof(BlockState), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      result.reason = std::string("search stopped by a CUDA error: ") +
                      cudaGetErrorString(cudaGetLastError());
      break;
    }
    std::vector<std::pair<double, int>> fresh;
    for (int b = 0; b < blocks; ++b) {
      const auto u = static_cast<std::size_t>(b);
      if (now[u].found != seen[u].found) fresh.emplace_back(now[u].best_objective, b);
    }
    std::sort(fresh.begin(), fresh.end());
    for (const auto& [device_objective, b] : fresh) {
      (void)device_objective;
      if (un > 0 && cudaMemcpy(point.data(), slots.best_x + static_cast<std::size_t>(b) * un,
                               un * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        break;
      }
      const double value = objective_of(cost, point);
      const bool improves = result.search.points.empty() ||
                            value < best - tol::kFeasibilityJumpMinScore *
                                               std::max(1.0, std::fabs(best));
      if (!improves) break;  // sorted: no later one improves either
      if (!mip::feasibility_jump_point_is_feasible(model, point,
                                                   settings.integrality_tolerance)) {
        ++result.rejected;
        continue;
      }
      result.search.points.push_back(point);
      if (settings.on_point) settings.on_point(point);
      best = value;
      break;
    }
    seen = now;
    bool finished = true;
    for (const BlockState& s : now) {
      finished = finished && (s.done != 0 || s.work >= p.work_limit);
    }
    if (finished || (settings.should_stop && settings.should_stop())) break;
  }
  for (const BlockState& s : seen) {
    result.search.work += static_cast<Count>(s.work);
    result.search.moves += static_cast<Count>(s.moves);
    result.search.weight_updates += static_cast<Count>(s.weight_updates);
  }
  return finish();
}

}  // namespace sankhya::gpu
