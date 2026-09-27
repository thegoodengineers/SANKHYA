// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for GPU PDHG memory estimation (#281).
//
// These tests run without a CUDA device: the estimator is pure arithmetic over model
// dimensions. Tests verify that the formula is monotone, accounts for all tracked
// allocations, and agrees with the policy constants documented in gpu_memory.hpp.

#include <gtest/gtest.h>

#include "gpu/gpu_memory.hpp"

namespace sankhya {
namespace {

using gpu::estimate_pdhg_gpu_memory;
using gpu::vram_reserve;

// ---- estimate_pdhg_gpu_memory -----------------------------------------------

TEST(GpuMemory, ZeroModelHasLibraryOverheadOnly) {
  // An empty model still needs the 16 MiB library-overhead constant.
  const std::size_t est = estimate_pdhg_gpu_memory(0, 0, 0);
  EXPECT_GE(est, 16ULL * 1024 * 1024);
}

TEST(GpuMemory, EstimateIncreasesWithNnz) {
  const std::size_t a = estimate_pdhg_gpu_memory(100, 200, 0);
  const std::size_t b = estimate_pdhg_gpu_memory(100, 200, 1000);
  const std::size_t c = estimate_pdhg_gpu_memory(100, 200, 100000);
  EXPECT_LT(a, b);
  EXPECT_LT(b, c);
}

TEST(GpuMemory, EstimateIncreasesWithRows) {
  const std::size_t a = estimate_pdhg_gpu_memory(1000, 500, 5000);
  const std::size_t b = estimate_pdhg_gpu_memory(10000, 500, 5000);
  EXPECT_LT(a, b);
}

TEST(GpuMemory, EstimateIncreasesWithCols) {
  const std::size_t a = estimate_pdhg_gpu_memory(500, 1000, 5000);
  const std::size_t b = estimate_pdhg_gpu_memory(500, 5000, 5000);
  EXPECT_LT(a, b);
}

TEST(GpuMemory, NegativeDimensionsAreClampedToZero) {
  // Should not underflow or crash.
  const std::size_t est = estimate_pdhg_gpu_memory(-1, -1, -1);
  EXPECT_GE(est, 16ULL * 1024 * 1024);
}

TEST(GpuMemory, SmallModelFormula) {
  // m=10, n=20, nnz=50 (GpuState layout after #382):
  //    9*20*8 = 1440   n-vecs (d_x,d_xn,d_ext,d_dx,d_aty,d_xsum,d_cost,d_clo,d_chi)
  //    8*10*8 =  640   m-vecs (d_y,d_yn,d_dy,d_ax,d_adx,d_ysum,d_rlo,d_rhi)
  //       3*8 =   24   scalars (d_scalars[3]: mv_x, mv_y, interaction)
  //  (10+1)*4 =   44   rowptr
  //    50*4   =  200   colidx
  //    50*8   =  400   vals
  //  16*1024^2        library overhead
  constexpr std::size_t kOverhead = 16ULL * 1024 * 1024;
  constexpr std::size_t expected = 1440 + 640 + 24 + 44 + 200 + 400 + kOverhead;
  EXPECT_EQ(estimate_pdhg_gpu_memory(10, 20, 50), expected);
}

TEST(GpuMemory, LargeModelIsInMibRange) {
  // 100k rows, 200k cols, 1M nonzeros: sparse storage ≈ 50 MiB (not GiB — the matrix is
  // sparse). n-vecs: 9*200k*8=14.4 MiB; m-vecs: 8*100k*8=6.4 MiB; CSR: ~12 MiB; 16 MiB
  // overhead. The estimate must exceed 40 MiB, well above the 16 MiB overhead constant alone.
  const std::size_t est = estimate_pdhg_gpu_memory(100000, 200000, 1000000);
  constexpr std::size_t k40Mib = 40ULL * 1024 * 1024;
  EXPECT_GT(est, k40Mib);
}

TEST(GpuMemory, DeterministicModeAddsASecondCsrForTheTranspose) {
  // #478: A^T held in CSR is n x m, so (n+1) row offsets, then the same nnz indices and
  // values as A. n=20, nnz=50: 21*4 + 50*4 + 50*8 = 84 + 200 + 400.
  EXPECT_EQ(gpu::estimate_pdhg_gpu_transpose_memory(20, 50), 684U);
  EXPECT_EQ(gpu::estimate_pdhg_gpu_transpose_memory(0, 0), sizeof(int));
  EXPECT_EQ(gpu::estimate_pdhg_gpu_transpose_memory(-1, -1), sizeof(int));
}

// ---- estimate_multi_gpu_partition_memory (#295) --------------------------------

TEST(GpuMemory, PartitionEstimateCountsOneCardsBlock) {
  // n=20 columns, a block of m=10 rows holding nnz=50, K=2 cards:
  //   (10 + K) n-vectors + 8 m-vectors        = (12*20 + 8*10) * 8 = 2560
  //   A_k CSR  (m+1)*4 + nnz*4 + nnz*8         = 44 + 200 + 400    = 644
  //   A_k^T CSR (n+1)*4 + nnz*4 + nnz*8        = 84 + 200 + 400    = 684
  //   evaluation 7 m + 8 n doubles             = (70 + 160) * 8    = 1840
  //   library overhead                         = 16 MiB
  const std::size_t overhead = 16ULL * 1024 * 1024;
  EXPECT_EQ(gpu::estimate_multi_gpu_partition_memory(20, 10, 50, 2),
            overhead + 2560U + 644U + 684U + 1840U);
  // A larger set replicates one more n-vector per card (the receive buffer grows with K).
  EXPECT_EQ(gpu::estimate_multi_gpu_partition_memory(20, 10, 50, 3) -
                gpu::estimate_multi_gpu_partition_memory(20, 10, 50, 2),
            20U * sizeof(double));
  // Clamped like the single-card estimate; K below 1 counts as 1.
  EXPECT_EQ(gpu::estimate_multi_gpu_partition_memory(-1, -1, -1, 0),
            overhead + 2 * sizeof(int));
}

TEST(GpuMemory, PartitionEstimateGrowsWithTheBlockAndGatesAgainstTheReserve) {
  // Two balanced blocks of a 1,000,000 x 1,000,000 model with 8 nonzeros a column: about
  // 340 MB a card. The block's share grows with its rows and nonzeros, the replicated
  // n-vectors do not shrink with the block, and the gate is the single-card policy's
  // (estimate + reserve against free): a 2 GiB card with 512 MiB free refuses that block, a
  // card with 1 GiB free holds it.
  const std::size_t half =
      gpu::estimate_multi_gpu_partition_memory(1000000, 500000, 4000000, 2);
  const std::size_t quarter =
      gpu::estimate_multi_gpu_partition_memory(1000000, 250000, 2000000, 2);
  EXPECT_LT(quarter, half);
  EXPECT_GT(quarter, half / 2);  // the replicated columns are paid on every card
  const std::size_t total_bytes = 2048ULL * 1024 * 1024;
  EXPECT_GT(half + gpu::vram_reserve(total_bytes), 512ULL * 1024 * 1024);
  EXPECT_LT(half + gpu::vram_reserve(total_bytes), 1024ULL * 1024 * 1024);
}

// ---- vram_reserve -----------------------------------------------------------

TEST(GpuMemory, ReserveIsAtLeast256Mib) {
  // Even with zero total VRAM the floor applies.
  EXPECT_GE(vram_reserve(0), 256ULL * 1024 * 1024);
}

TEST(GpuMemory, ReserveIsTenPercentWhenLarger) {
  // 10 GiB total → 10% = 1 GiB > 256 MiB floor.
  constexpr std::size_t total = 10ULL * 1024 * 1024 * 1024;
  EXPECT_EQ(vram_reserve(total), total / 10);
}

TEST(GpuMemory, ReserveFloorAppliesForSmallVram) {
  // 1 GiB total → 10% = 102 MiB < 256 MiB floor.
  constexpr std::size_t total = 1ULL * 1024 * 1024 * 1024;
  EXPECT_EQ(vram_reserve(total), 256ULL * 1024 * 1024);
}

TEST(GpuMemory, ReserveAt6GibVram) {
  // 6144 MiB (RTX 4050 Laptop GPU) → 10% = 614 MiB > 256 MiB floor.
  constexpr std::size_t total = 6144ULL * 1024 * 1024;
  EXPECT_EQ(vram_reserve(total), total / 10);
}

}  // namespace
}  // namespace sankhya
