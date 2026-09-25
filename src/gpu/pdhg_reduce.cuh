// SPDX-License-Identifier: Apache-2.0
// SANKHYA - fixed-order device reductions for GPU PDHG (#478, #383).
//
// The movement and interaction scalars of the adaptive step rule [PDLP section 3.1] used to
// be summed by a CUB BlockReduce per block followed by an atomicAdd of each block's total
// into one double. Floating-point addition is not associative, and the order in which the
// blocks' atomics land is decided by the scheduler, so two runs on the same input summed
// the same numbers in different orders and could disagree in the last bit. The step rule
// compares eta against movement / interaction, a restart decision is a comparison of
// residual ratios, and one flipped comparison is a different trajectory from there on
// (#448: GPU iteration counts varied run to run on one card).
//
// Here every block writes its total to its own slot of a partials buffer, and one block of
// kThreads threads then sums the slots in an order fixed by the thread index alone: thread
// t adds slots t, t + kThreads, t + 2 kThreads, ... left to right, and cub::BlockReduce
// combines the kThreads partial sums. BlockReduce is a fixed sequence of warp shuffles and
// shared-memory steps with no atomics, so the data flow, and the rounding, depend only on
// the block size and the number of slots. The same input gives the same bits every run.
//
// References:
//   Applegate et al., "Practical Large-Scale Linear Programming using Primal-Dual Hybrid
//     Gradient", NeurIPS 2021, section 3.1 (the scalars this sums).
//   Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM 2002, section 4.1
//     (summation order and its rounding error; pairwise and blocked summation).
//   NVIDIA CUB documentation, cub::BlockReduce (the collective's data flow).
#pragma once

#include <cub/block/block_reduce.cuh>

namespace sankhya::gpu::detail {

/// Sums `v` over the calling block and writes the total to partials[blockIdx.x]. Every thread
/// of the block must call it; blockDim.x must equal kThreads.
template <int kThreads>
__device__ __forceinline__ void write_block_partial(double v, double* partials) {
  using BlockReduce = cub::BlockReduce<double, kThreads>;
  __shared__ typename BlockReduce::TempStorage temp;
  const double total = BlockReduce(temp).Sum(v);
  if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

/// The three step-rule scalars from their partials, by a single block of kThreads threads in
/// a fixed order. Layout of `partials`: [0, bn) movement x, [bn, bn + bm) movement y,
/// [bn + bm, bn + 2 bm) interaction. The totals are valid in thread 0 only.
template <int kThreads>
__device__ __forceinline__ void sum_step_partials(const double* __restrict__ partials, int bn,
                                                  int bm, double out[3]) {
  using BlockReduce = cub::BlockReduce<double, kThreads>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int counts[3] = {bn, bm, bm};
  int offset = 0;
  for (int s = 0; s < 3; ++s) {
    double t = 0.0;
    for (int k = static_cast<int>(threadIdx.x); k < counts[s]; k += kThreads) {
      t += partials[offset + k];
    }
    const double total = BlockReduce(temp).Sum(t);
    if (threadIdx.x == 0) out[s] = total;
    __syncthreads();  // the temporary storage is reused by the next sum
    offset += counts[s];
  }
}

}  // namespace sankhya::gpu::detail
