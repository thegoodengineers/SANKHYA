// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU device enumeration and work partitioning (#295).
//
// Pure C++ API — no CUDA runtime types in the header. The CPU-safe subset
// (partition_rows, parse_device_ids) is compiled into sankhya_objects unconditionally;
// the CUDA subset (device_count, can_peer_access) lives in multi_device.cu and is only
// present in SANKHYA_ENABLE_CUDA builds.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::gpu {

// A half-open row range owned by one CUDA device.
struct RowPartition {
  int device_id;
  int row_start;  // inclusive
  int row_end;    // exclusive
  [[nodiscard]] int local_m() const { return row_end - row_start; }
};

// Partition m rows across device_ids as evenly as possible.
// Remainder rows go one-per-device to the first (remainder) devices.
[[nodiscard]] std::vector<RowPartition> partition_rows(int m,
                                                       const std::vector<int>& device_ids);

// The work a row block costs one iteration of PDHG: its nonzeros (the two SpMVs with A_k and
// the one with A_k^T each touch every one of them) plus one per row (the dual update, the
// interaction product and the running dual sum each touch every row once).
[[nodiscard]] std::int64_t partition_weight(const std::vector<Index>& row_starts,
                                            const RowPartition& part);

// Partition rows [0, m) into contiguous blocks, one per entry of device_ids, balanced by
// partition_weight rather than by row count (#295): with m = row_starts.size() - 1 and
// W = nnz + m the total weight, boundary k (k = 1..K-1) is the row r whose prefix weight is
// nearest k * W / K, ties to the lower row, never before boundary k - 1. Every block is then
// within one row's weight of W / K. row_starts is a CSR row-pointer array (m + 1 entries,
// nondecreasing); an empty or malformed one partitions zero rows. An even split by rows, the
// earlier scheme (partition_rows), puts a dense linking block on one card whole.
[[nodiscard]] std::vector<RowPartition> partition_rows_by_nonzeros(
    const std::vector<Index>& row_starts, const std::vector<int>& device_ids);

// Parse comma-separated device IDs (e.g. "0,1,2"). Returns {0} for empty or "auto".
// Negative values and non-integer tokens are silently skipped. Duplicates are removed.
[[nodiscard]] std::vector<int> parse_device_ids(const std::string& option);

#ifdef SANKHYA_ENABLE_CUDA
// Returns the number of CUDA-capable devices present. Returns 0 on driver error.
[[nodiscard]] int device_count();

// Returns true when device `from` can read device `to`'s memory via P2P / NVLink.
// `from == to` always returns true.
[[nodiscard]] bool can_peer_access(int from, int to);
#endif

}  // namespace sankhya::gpu
