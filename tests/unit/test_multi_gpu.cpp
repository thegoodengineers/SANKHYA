// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for multi-GPU partitioning and device utilities (#295).
//
// The partition (even and nonzero-balanced) and option-parsing tests are CPU-only and run
// without a CUDA device. The device enumeration and solver tests are guarded by
// SANKHYA_ENABLE_CUDA.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gpu/multi_device.hpp"
#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/pdhg_gpu.hpp"
#include "gpu/pdhg_multi_gpu.hpp"
#endif

namespace sankhya {
namespace {

using gpu::parse_device_ids;
using gpu::partition_rows;
using gpu::partition_rows_by_nonzeros;
using gpu::partition_weight;
using gpu::RowPartition;

// ---- parse_device_ids ----------------------------------------------------

TEST(MultiGpu, ParseAutoReturnsDevice0) {
  const auto ids = parse_device_ids("auto");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseEmptyReturnsDevice0) {
  const auto ids = parse_device_ids("");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseSingleDevice) {
  const auto ids = parse_device_ids("0");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseTwoDevices) {
  const auto ids = parse_device_ids("0,1");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
}

TEST(MultiGpu, ParseThreeDevices) {
  const auto ids = parse_device_ids("0,1,2");
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
  EXPECT_EQ(ids[2], 2);
}

TEST(MultiGpu, ParseNonZeroStart) {
  const auto ids = parse_device_ids("2,3");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 2);
  EXPECT_EQ(ids[1], 3);
}

TEST(MultiGpu, ParseDuplicatesRemoved) {
  const auto ids = parse_device_ids("0,0,1");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
}

TEST(MultiGpu, ParseNegativeSkipped) {
  // Negative IDs are silently dropped; result falls back to {0}.
  const auto ids = parse_device_ids("-1,0");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST(MultiGpu, ParseGarbageFallsBack) {
  const auto ids = parse_device_ids("abc");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

// ---- partition_rows ------------------------------------------------------

TEST(MultiGpu, PartitionZeroRows) {
  const auto parts = partition_rows(0, {0, 1});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0].row_start, 0);
  EXPECT_EQ(parts[0].row_end, 0);
  EXPECT_EQ(parts[0].local_m(), 0);
  EXPECT_EQ(parts[1].row_start, 0);
  EXPECT_EQ(parts[1].row_end, 0);
}

TEST(MultiGpu, PartitionEvenRows) {
  // 10 rows across 2 devices -> 5 each.
  const auto parts = partition_rows(10, {0, 1});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0].device_id, 0);
  EXPECT_EQ(parts[0].row_start, 0);
  EXPECT_EQ(parts[0].row_end, 5);
  EXPECT_EQ(parts[1].device_id, 1);
  EXPECT_EQ(parts[1].row_start, 5);
  EXPECT_EQ(parts[1].row_end, 10);
}

TEST(MultiGpu, PartitionOddRows) {
  // 11 rows across 2 devices -> 6, 5.
  const auto parts = partition_rows(11, {0, 1});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0].local_m(), 6);
  EXPECT_EQ(parts[1].local_m(), 5);
  EXPECT_EQ(parts[0].row_end, parts[1].row_start);
  EXPECT_EQ(parts[1].row_end, 11);
}

TEST(MultiGpu, PartitionThreeDevices) {
  // 10 rows across 3 devices -> 4, 3, 3.
  const auto parts = partition_rows(10, {0, 1, 2});
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0].local_m(), 4);
  EXPECT_EQ(parts[1].local_m(), 3);
  EXPECT_EQ(parts[2].local_m(), 3);
  // Contiguous.
  EXPECT_EQ(parts[0].row_end, parts[1].row_start);
  EXPECT_EQ(parts[1].row_end, parts[2].row_start);
  EXPECT_EQ(parts[2].row_end, 10);
}

TEST(MultiGpu, PartitionCoversAllRows) {
  for (int m = 1; m <= 20; ++m) {
    for (int k = 1; k <= 4; ++k) {
      std::vector<int> devs;
      devs.reserve(static_cast<std::size_t>(k));
      for (int i = 0; i < k; ++i) devs.push_back(i);
      const auto parts = partition_rows(m, devs);
      ASSERT_EQ(static_cast<int>(parts.size()), k);
      int total = 0;
      for (const auto& p : parts) total += p.local_m();
      EXPECT_EQ(total, m) << "m=" << m << " k=" << k;
    }
  }
}

TEST(MultiGpu, PartitionSingleDevice) {
  const auto parts = partition_rows(7, {3});
  ASSERT_EQ(parts.size(), 1u);
  EXPECT_EQ(parts[0].device_id, 3);
  EXPECT_EQ(parts[0].row_start, 0);
  EXPECT_EQ(parts[0].row_end, 7);
}

TEST(MultiGpu, PartitionEmptyDeviceList) {
  const auto parts = partition_rows(5, {});
  EXPECT_TRUE(parts.empty());
}

// ---- partition_rows_by_nonzeros (#295) ------------------------------------

// Checks the invariants every nonzero-balanced partition must hold: one block per device in
// order, contiguous, covering [0, m), and each block within one row's weight of W / K.
void expect_balanced_partition(const std::vector<Index>& row_starts,
                               const std::vector<int>& devices) {
  const auto parts = partition_rows_by_nonzeros(row_starts, devices);
  ASSERT_EQ(parts.size(), devices.size());
  const int m = static_cast<int>(row_starts.size()) - 1;
  std::int64_t total = 0;
  std::int64_t heaviest_row = 0;
  for (int i = 0; i < m; ++i) {
    const std::int64_t w =
        static_cast<std::int64_t>(row_starts[static_cast<std::size_t>(i) + 1]) -
        row_starts[static_cast<std::size_t>(i)] + 1;
    total += w;
    heaviest_row = std::max(heaviest_row, w);
  }
  int next = 0;
  std::int64_t covered = 0;
  const double ideal = static_cast<double>(total) / static_cast<double>(devices.size());
  for (std::size_t k = 0; k < parts.size(); ++k) {
    EXPECT_EQ(parts[k].device_id, devices[k]);
    EXPECT_EQ(parts[k].row_start, next) << "block " << k << " is not contiguous";
    EXPECT_GE(parts[k].row_end, parts[k].row_start);
    next = parts[k].row_end;
    const std::int64_t w = partition_weight(row_starts, parts[k]);
    covered += w;
    EXPECT_LE(std::fabs(static_cast<double>(w) - ideal), static_cast<double>(heaviest_row))
        << "block " << k << " weighs " << w << " against an ideal of " << ideal
        << " with rows of weight up to " << heaviest_row;
  }
  EXPECT_EQ(next, m);
  EXPECT_EQ(covered, total);
}

std::vector<Index> row_starts_from_counts(const std::vector<Index>& counts) {
  std::vector<Index> rs(counts.size() + 1, 0);
  for (std::size_t i = 0; i < counts.size(); ++i) rs[i + 1] = rs[i] + counts[i];
  return rs;
}

TEST(MultiGpu, NonzeroPartitionSplitsASkewedMatrixByWorkNotByRows) {
  // Ten dense linking rows of 100 nonzeros on top of ninety rows of one. Split by rows the
  // first card gets rows 0-49, 1,090 units of work, and the second 100: an 11x imbalance.
  std::vector<Index> counts(10, 100);
  counts.resize(100, 1);
  const auto rs = row_starts_from_counts(counts);
  const auto even = partition_rows(100, {0, 1});
  EXPECT_EQ(partition_weight(rs, even[0]), 1090);
  EXPECT_EQ(partition_weight(rs, even[1]), 100);

  const auto parts = partition_rows_by_nonzeros(rs, {0, 1});
  ASSERT_EQ(parts.size(), 2u);
  // W = 1,090 + 100 = 1,190, target 595 at the boundary: five dense rows weigh 505, six 606,
  // and 606 is the nearer. The second card takes the other four dense rows and all ninety
  // light ones: 404 + 180 = 584.
  EXPECT_EQ(parts[0].row_start, 0);
  EXPECT_EQ(parts[0].row_end, 6);
  EXPECT_EQ(parts[1].row_end, 100);
  EXPECT_EQ(partition_weight(rs, parts[0]), 606);
  EXPECT_EQ(partition_weight(rs, parts[1]), 584);
  expect_balanced_partition(rs, {0, 1});
}

TEST(MultiGpu, NonzeroPartitionOfUniformRowsIsAnEvenSplit) {
  for (int m = 1; m <= 40; ++m)
    for (int k = 1; k <= 5; ++k) {
      std::vector<int> devices;
      for (int i = 0; i < k; ++i) devices.push_back(i);
      const auto rs =
          row_starts_from_counts(std::vector<Index>(static_cast<std::size_t>(m), 7));
      const auto parts = partition_rows_by_nonzeros(rs, devices);
      int smallest = m, largest = 0;
      for (const auto& p : parts) {
        smallest = std::min(smallest, p.local_m());
        largest = std::max(largest, p.local_m());
      }
      EXPECT_LE(largest - smallest, 1) << "m=" << m << " k=" << k;
      expect_balanced_partition(rs, devices);
    }
}

TEST(MultiGpu, NonzeroPartitionHoldsItsInvariantsOnRandomRowLengths) {
  std::mt19937 rng(20260925);
  for (int trial = 0; trial < 200; ++trial) {
    const int m = std::uniform_int_distribution<int>(0, 300)(rng);
    const int k = std::uniform_int_distribution<int>(1, 8)(rng);
    std::vector<Index> counts(static_cast<std::size_t>(m));
    for (auto& c : counts)
      // Mostly short rows with the occasional very long one, as linking rows are.
      c = std::uniform_int_distribution<int>(0, 9)(rng) == 0
              ? std::uniform_int_distribution<Index>(50, 500)(rng)
              : std::uniform_int_distribution<Index>(0, 6)(rng);
    std::vector<int> devices;
    for (int i = 0; i < k; ++i) devices.push_back(i);
    const auto rs = row_starts_from_counts(counts);
    SCOPED_TRACE("trial " + std::to_string(trial));
    expect_balanced_partition(rs, devices);
    // Deterministic: the same input gives the same boundaries.
    const auto a = partition_rows_by_nonzeros(rs, devices);
    const auto b = partition_rows_by_nonzeros(rs, devices);
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].row_end, b[i].row_end);
  }
}

TEST(MultiGpu, NonzeroPartitionEdgeCases) {
  EXPECT_TRUE(partition_rows_by_nonzeros({0, 3, 5}, {}).empty());
  // No rows: every device gets the empty block.
  for (const auto& p : partition_rows_by_nonzeros({0}, {0, 1})) EXPECT_EQ(p.local_m(), 0);
  for (const auto& p : partition_rows_by_nonzeros({}, {0, 1})) EXPECT_EQ(p.local_m(), 0);
  // More devices than rows: blocks may be empty, but all rows are covered once.
  expect_balanced_partition({0, 4, 8}, {0, 1, 2, 3});
  // One row outweighing the rest: it stands alone, and its block is as close as it can be.
  expect_balanced_partition(row_starts_from_counts({1, 1, 1000, 1, 1}), {0, 1});
  // Device ids are carried through in order, whatever their values.
  const auto parts = partition_rows_by_nonzeros({0, 1, 2, 3, 4}, {3, 1});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0].device_id, 3);
  EXPECT_EQ(parts[1].device_id, 1);
}

// ---- RowPartition helpers ------------------------------------------------

TEST(MultiGpu, LocalMIsCorrect) {
  RowPartition p{0, 10, 20};
  EXPECT_EQ(p.local_m(), 10);
}

// ---- CUDA-only tests (device_count, can_peer_access, solver) -------------
#ifdef SANKHYA_ENABLE_CUDA

TEST(MultiGpu, DeviceCountNonNegative) {
  EXPECT_GE(gpu::device_count(), 0);
}

TEST(MultiGpu, PeerAccessSelfIsTrue) {
  const int cnt = gpu::device_count();
  for (int i = 0; i < cnt; ++i) EXPECT_TRUE(gpu::can_peer_access(i, i));
}

// Evidence test: solve_pdhg_multi_gpu actually runs (reviewer item 3).
//
// Uses device_ids = {0, 0}: two virtual partitions on the same physical GPU.
// This exercises the full multi-GPU code path (row partitioning, per-device cuSPARSE
// SpMVs, host-mediated allreduce) without needing a second physical GPU.
// The result is compared against the single-GPU solver to prove correctness.
TEST(MultiGpu, SolveTwoVirtualDevicesMatchesSingleGpu) {
  if (gpu::device_count() == 0) {
    GTEST_SKIP() << "no CUDA device present";
  }

  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib/afiro.mps")
          .string();
  Model model;
  const io::ReadResult r = io::read_model(path, &model);
  ASSERT_TRUE(r.ok) << path << ": " << r.error;

  Options opts;
  opts.set_bool("log_to_console", false);
  opts.set_double("pdhg_tolerance", 1e-6);
  opts.set_bool("pdhg_polish", false);
  opts.set_int("iteration_limit", 500000);

  Logger silent(nullptr);

  // Single-GPU reference.
  const Solution single = gpu::solve_pdhg_gpu(model, opts, silent);
  if (single.algorithm.find("cuda") == std::string::npos &&
      single.algorithm.find("gpu") == std::string::npos) {
    GTEST_SKIP() << "CUDA backend not in this build (algorithm=\"" << single.algorithm << "\")";
  }
  ASSERT_EQ(single.status, SolveStatus::kOptimal) << single.message;

  // Multi-GPU path with {0, 0}: rows are split into two halves, both run on device 0.
  const Solution multi = gpu::solve_pdhg_multi_gpu(model, opts, {0, 0}, silent);
  EXPECT_EQ(multi.algorithm, "pdhg-cuda-multi") << "multi-GPU code path did not execute";
  ASSERT_EQ(multi.status, SolveStatus::kOptimal) << multi.message;

  const double scale = std::max(1.0, std::fabs(single.objective));
  EXPECT_NEAR(multi.objective, single.objective, 1e-5 * scale)
      << "multi-GPU objective diverged from single-GPU";
}

#endif  // SANKHYA_ENABLE_CUDA

}  // namespace
}  // namespace sankhya
