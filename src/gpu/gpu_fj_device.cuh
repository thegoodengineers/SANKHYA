// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU Feasibility Jump (#508), the device side. Included by gpu_fj.cu alone.
//
// Luteberget and Sartor, "Feasibility Jump: an LP-free Lagrangian MIP heuristic",
// Mathematical Programming Computation 15 (2023): the jump value, the weighted violation and
// the weight update. The parallel layout follows the GPU Feasibility Jump literature
// (Corduk, Sielski, Boucher & Aatish, "GPU-accelerated primal heuristics for mixed integer
// programming", arXiv:2510.20499): the constraint state on the device, one warp per
// candidate column scoring its jump over the column's nonzeros, independent searches in
// parallel blocks. Written from the papers and from the CPU reference in
// src/mip/feasibility_jump.cpp; no solver source was read.
//
// ONE BLOCK IS ONE SEARCH. Block b owns its own point, activities, row weights and violated
// list (the slots at offset b), and runs the CPU reference's loop:
//   sample   each warp draws tol::kGpuFeasibilityJumpSamplesPerWarp candidate columns, a random entry of a random
//            violated row (a random costed column once no row is violated), as the CPU does;
//   score    the warp computes the column's jump value and its score, lanes striding over
//            the column's nonzeros and reducing by shuffles (warp_best_jump below);
//   select   thread 0 takes the best score over the warps, in sample order;
//   move     every thread updates the activities of the column's rows at once, the rows
//            whose violated state changed are compacted in column order, and thread 0
//            updates the violated list with them;
//   or bump  at a local minimum every violated row's weight gains 1 (the objective's, when
//            none is violated): the paper's update, over the violated list in parallel.
// When no row is violated the activities are recomputed from scratch, row by row, before the
// point is believed, and a point better than the block's last one is kept in its best slot
// for the host to collect, verify against the original model and hand over.
//
// DETERMINISM. No atomics: a search's every reduction is a shuffle butterfly or a sum over the
// warps in a fixed order, and every random draw is a hash of (seed, block, step, sample). A
// run is bitwise reproducible for a seed, a restart count and a work budget on one build and
// card; a clock-based stop only changes where it ends.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace sankhya::gpu::fj {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr unsigned kFullMask = 0xffffffffU;

/// The model, read only, shared by every block. Bounds are IEEE infinities where absent.
struct Problem {
  int n = 0;
  int m = 0;
  long long nnz = 0;
  const int* col_start = nullptr;  ///< CSC
  const int* col_row = nullptr;
  const double* col_val = nullptr;
  const int* row_start = nullptr;  ///< CSR, for the sample and the exact recomputation
  const int* row_col = nullptr;
  const double* row_val = nullptr;
  const double* row_lo = nullptr;
  const double* row_hi = nullptr;
  const double* lo = nullptr;  ///< column bounds, integer ones rounded inwards
  const double* hi = nullptr;
  const double* cost = nullptr;  ///< in minimise space
  const unsigned char* integer = nullptr;
  const int* costed = nullptr;  ///< movable columns with a nonzero cost
  int n_costed = 0;
  int samples_per_warp = 1;
  double feasibility = 0.0;
  double min_score = 0.0;
  long long work_limit = 0;
  unsigned long long seed = 0;
};

/// One search's scalars; kept in shared memory while a launch runs.
struct BlockState {
  double objective_weight;
  double best_objective;
  long long work;
  long long moves;
  long long weight_updates;
  long long step;
  int violated_count;
  int checked;  ///< the point has been re-measured from scratch since the last move
  int done;     ///< feasible with nothing left to improve
  int found;    ///< points kept in the best slot so far
};

/// Per-block storage, block b at offset b * n or b * m.
struct Slots {
  double* x = nullptr;
  double* best_x = nullptr;
  double* activity = nullptr;
  double* weight = nullptr;
  int* violated = nullptr;
  int* violated_at = nullptr;
  BlockState* state = nullptr;
};

__device__ inline double violation(double activity, double lower, double upper) {
  double v = 0.0;
  if (isfinite(lower) && activity < lower) v += lower - activity;
  if (isfinite(upper) && activity > upper) v += activity - upper;
  return v;
}

/// splitmix64 finaliser: the counter-based generator behind every draw.
__device__ inline unsigned long long mix(unsigned long long z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/// A uniform index in [0, count) from the high half of a hash.
__device__ inline int pick(unsigned long long h, int count) {
  return static_cast<int>(((h >> 32) * static_cast<unsigned long long>(count)) >> 32);
}

/// Butterfly sum: every lane ends with the same value (a + b == b + a exactly).
__device__ inline double warp_sum(double v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFullMask, v, o);
  return v;
}

__device__ inline double warp_min(double v) {
  for (int o = 16; o > 0; o >>= 1) v = fmin(v, __shfl_xor_sync(kFullMask, v, o));
  return v;
}

/// Entry k's breakpoints for column value xj: where its row's activity crosses the bound it
/// meets first moving up (`first`) and the other one (`second`); NaN where the bound is
/// absent. Explicit round-to-nearest operations so every pass recomputes the same values.
__device__ inline bool entry_breaks(const Problem& p, const double* act, const double* weight,
                                    double xj, int k, double* first, double* second,
                                    double* pull) {
  const double a = p.col_val[k];
  if (a == 0.0) return false;
  const int i = p.col_row[k];
  const double rest = __dsub_rn(act[i], __dmul_rn(a, xj));
  *pull = __dmul_rn(weight[i], fabs(a));
  const double f = a > 0.0 ? p.row_lo[i] : p.row_hi[i];
  const double s = a > 0.0 ? p.row_hi[i] : p.row_lo[i];
  *first = isfinite(f) ? __ddiv_rn(__dsub_rn(f, rest), a) : NAN;
  *second = isfinite(s) ? __ddiv_rn(__dsub_rn(s, rest), a) : NAN;
  return true;
}

/// Change of the weighted violation plus the weighted objective when column j moves from xj
/// to value: the CPU reference's delta(), lanes striding over the column.
__device__ inline double warp_delta(const Problem& p, const double* act, const double* weight,
                                    double objective_weight, int j, double xj, double value,
                                    int lane) {
  const double step = value - xj;
  double change = 0.0;
  for (int k = p.col_start[j] + lane; k < p.col_start[j + 1]; k += 32) {
    const int i = p.col_row[k];
    const double before = act[i];
    const double after = before + p.col_val[k] * step;
    change += weight[i] * (violation(after, p.row_lo[i], p.row_hi[i]) -
                           violation(before, p.row_lo[i], p.row_hi[i]));
  }
  return objective_weight * p.cost[j] * step + warp_sum(change);
}

struct Jump {
  int column;
  double value;
  double score;  ///< decrease of the weighted violation plus the weighted objective
};

/// Smallest breakpoint strictly above `above_of` (all of them when it is -inf), +inf if none.
__device__ inline double warp_next_break(const Problem& p, const double* act,
                                         const double* weight, double xj, int j, double above_of,
                                         int lane) {
  double least = INFINITY;
  for (int k = p.col_start[j] + lane; k < p.col_start[j + 1]; k += 32) {
    double first, second, pull;
    if (!entry_breaks(p, act, weight, xj, k, &first, &second, &pull)) continue;
    if (first > above_of) least = fmin(least, first);
    if (second > above_of) least = fmin(least, second);
  }
  return warp_min(least);
}

/// The jump of column j, as best_jump() in feasibility_jump.cpp: the minimiser of the convex
/// piecewise-linear weighted violation in x_j nearest the current value, projected on the box
/// and rounded both ways on an integer column, and the better of those values by score.
///
/// The CPU sorts the breakpoints. A warp finds the first breakpoint where the slope turns
/// non-negative by a randomised selection instead: the pivot is the breakpoint with the least
/// hash among those still in the open interval, the slope just right of it is one reduction,
/// and the interval halves around it. With fixed random priorities that is a search in a
/// random treap, expected O(log L) rounds of O(L / 32) per lane for L breakpoints, where a
/// sort in a warp would need shared memory sized to the longest column.
///
/// A column of two integer values (a binary) skips all of that: the only jump is the other
/// value, and it is taken only when it strictly improves, exactly when the CPU would take it.
__device__ Jump warp_best_jump(const Problem& p, const double* x, const double* act,
                               const double* weight, double objective_weight, int j, int lane,
                               long long* work) {
  Jump jump{-1, 0.0, 0.0};
  const double lo = p.lo[j];
  const double hi = p.hi[j];
  if (!(lo < hi)) return jump;
  const double xj = x[j];
  const int begin = p.col_start[j];
  const int end = p.col_start[j + 1];
  const long long length = end - begin;
  double candidates[2] = {0.0, 0.0};
  int count = 1;
  if (p.integer[j] != 0 && hi - lo == 1.0) {
    candidates[0] = xj == lo ? hi : lo;
  } else {
    double part = 0.0;
    for (int k = begin + lane; k < end; k += 32) {
      double first, second, pull;
      if (entry_breaks(p, act, weight, xj, k, &first, &second, &pull) && !isnan(first)) {
        part -= pull;
      }
    }
    const double slope = objective_weight * p.cost[j] + warp_sum(part);
    *work += length + 1;
    // [left, right]: the minimisers of the function without the column's box.
    double left = -INFINITY;
    double right = -INFINITY;
    if (slope == 0.0) {
      right = warp_next_break(p, act, weight, xj, j, -INFINITY, lane);
      *work += length;
    } else if (slope < 0.0) {
      double below = -INFINITY;
      double above = INFINITY;
      double best = INFINITY;
      double best_slope = 0.0;
      for (;;) {
        unsigned long long key = ~0ULL;
        double pivot = 0.0;
        for (int k = begin + lane; k < end; k += 32) {
          double side[2], pull;
          if (!entry_breaks(p, act, weight, xj, k, &side[0], &side[1], &pull)) continue;
          for (int s = 0; s < 2; ++s) {
            if (!(side[s] > below && side[s] < above)) continue;
            const unsigned long long h =
                mix((static_cast<unsigned long long>(j) << 32) ^
                    static_cast<unsigned long long>(2 * (k - begin) + s));
            if (h < key || (h == key && side[s] < pivot)) {
              key = h;
              pivot = side[s];
            }
          }
        }
        for (int o = 16; o > 0; o >>= 1) {
          const unsigned long long other_key = __shfl_xor_sync(kFullMask, key, o);
          const double other_pivot = __shfl_xor_sync(kFullMask, pivot, o);
          if (other_key < key || (other_key == key && other_pivot < pivot)) {
            key = other_key;
            pivot = other_pivot;
          }
        }
        *work += length;
        if (key == ~0ULL) break;
        double rise = 0.0;
        for (int k = begin + lane; k < end; k += 32) {
          double first, second, pull;
          if (!entry_breaks(p, act, weight, xj, k, &first, &second, &pull)) continue;
          if (first <= pivot) rise += pull;
          if (second <= pivot) rise += pull;
        }
        const double at = slope + warp_sum(rise);
        *work += length;
        if (at >= 0.0) {
          above = pivot;
          best = pivot;
          best_slope = at;
        } else {
          below = pivot;
        }
      }
      if (best == INFINITY) {
        left = right = INFINITY;  // still falling after the last breakpoint
      } else {
        left = best;
        if (best_slope > 0.0) {
          right = left;
        } else {
          right = warp_next_break(p, act, weight, xj, j, left, lane);
          *work += length;
        }
      }
    }
    double target;
    if (left > hi) {
      target = hi;
    } else if (right < lo) {
      target = lo;
    } else {
      target = fmin(fmax(xj, fmax(left, lo)), fmin(right, hi));
    }
    if (!isfinite(target)) return jump;
    candidates[0] = target;
    if (p.integer[j] != 0) {
      candidates[0] = fmin(fmax(floor(target), lo), hi);
      candidates[1] = fmin(fmax(ceil(target), lo), hi);
      count = candidates[1] != candidates[0] ? 2 : 1;
    }
  }
  for (int c = 0; c < count; ++c) {
    const double value = candidates[c];
    if (value == xj) continue;
    const double score = -warp_delta(p, act, weight, objective_weight, j, xj, value, lane);
    *work += length + 1;
    const bool nearer = jump.column >= 0 && score == jump.score &&
                        fabs(value - xj) < fabs(jump.value - xj);
    if (jump.column < 0 || score > jump.score || nearer) jump = Jump{j, value, score};
  }
  return jump;
}

/// Ordered block-wide compaction: the flagged threads' items are written to out[0..total) in
/// thread order and each flagged thread learns its position. Every thread must call it.
__device__ inline int block_compact(bool flag, int item, int* out, int* position,
                                    int* warp_counts) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const unsigned ballot = __ballot_sync(kFullMask, flag);
  if (lane == 0) warp_counts[warp] = __popc(ballot);
  __syncthreads();
  int offset = 0;
  int total = 0;
  for (int w = 0; w < kWarps; ++w) {
    if (w < warp) offset += warp_counts[w];
    total += warp_counts[w];
  }
  offset += __popc(ballot & ((1U << lane) - 1U));
  if (flag) {
    out[offset] = item;
    *position = offset;
  }
  __syncthreads();  // warp_counts is reused by the next call
  return total;
}

}  // namespace sankhya::gpu::fj
