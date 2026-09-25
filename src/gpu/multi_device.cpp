// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU partition and option utilities (#295).
//
// CPU-only; no CUDA calls. Compiled unconditionally so that tests can exercise
// partition_rows / parse_device_ids without a CUDA build.

#include "multi_device.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace sankhya::gpu {

std::vector<RowPartition> partition_rows(int m, const std::vector<int>& device_ids) {
  const int k = static_cast<int>(device_ids.size());
  std::vector<RowPartition> parts;
  if (k <= 0) return parts;
  parts.reserve(static_cast<std::size_t>(k));
  const int base = (m >= 0 ? m : 0) / k;
  const int rem = (m >= 0 ? m : 0) % k;
  int row = 0;
  for (int i = 0; i < k; ++i) {
    const int sz = base + (i < rem ? 1 : 0);
    parts.push_back({device_ids[static_cast<std::size_t>(i)], row, row + sz});
    row += sz;
  }
  return parts;
}

std::int64_t partition_weight(const std::vector<Index>& row_starts, const RowPartition& part) {
  if (part.row_end <= part.row_start) return 0;
  const auto lo = static_cast<std::size_t>(part.row_start);
  const auto hi = static_cast<std::size_t>(part.row_end);
  if (hi >= row_starts.size()) return 0;
  return static_cast<std::int64_t>(row_starts[hi]) - row_starts[lo] +
         (part.row_end - part.row_start);
}

std::vector<RowPartition> partition_rows_by_nonzeros(const std::vector<Index>& row_starts,
                                                     const std::vector<int>& device_ids) {
  const int k = static_cast<int>(device_ids.size());
  std::vector<RowPartition> parts;
  if (k <= 0) return parts;
  parts.reserve(static_cast<std::size_t>(k));
  const int m = row_starts.empty() ? 0 : static_cast<int>(row_starts.size()) - 1;
  // Prefix weight of the first r rows: nonzeros in them plus r. Strictly increasing in r.
  auto prefix = [&](int r) -> std::int64_t {
    return static_cast<std::int64_t>(row_starts[static_cast<std::size_t>(r)]) - row_starts[0] +
           r;
  };
  const std::int64_t total = m > 0 ? prefix(m) : 0;
  int start = 0;
  for (int i = 0; i < k; ++i) {
    int end = m;
    if (i + 1 < k) {
      // Target k * W / K in exact integer arithmetic: compare 2 * K * prefix against
      // 2 * (i + 1) * W, so the nearest row is chosen without rounding the target.
      const std::int64_t target2k = 2 * static_cast<std::int64_t>(i + 1) * total;
      const auto kk = static_cast<std::int64_t>(k);
      // Smallest r >= start with K * prefix(r) >= (i + 1) * W.
      int lo = start, hi = m;
      while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (2 * kk * prefix(mid) >= target2k)
          hi = mid;
        else
          lo = mid + 1;
      }
      end = lo;
      // The row before may be nearer the target; ties go to the lower row.
      if (end > start) {
        const std::int64_t above = 2 * kk * prefix(end) - target2k;
        const std::int64_t below = target2k - 2 * kk * prefix(end - 1);
        if (below <= above) end = end - 1;
      }
    }
    parts.push_back({device_ids[static_cast<std::size_t>(i)], start, end});
    start = end;
  }
  return parts;
}

std::vector<int> parse_device_ids(const std::string& opt) {
  if (opt.empty() || opt == "auto") return {0};
  std::vector<int> ids;
  std::size_t pos = 0;
  while (pos <= opt.size()) {
    const std::size_t comma = opt.find(',', pos);
    const std::size_t end = (comma == std::string::npos) ? opt.size() : comma;
    if (end > pos) {
      const std::string tok = opt.substr(pos, end - pos);
      try {
        const int id = std::stoi(tok);
        if (id >= 0) ids.push_back(id);
      } catch (...) {
      }
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (ids.empty()) return {0};
  // Remove duplicates, keeping first occurrence.
  std::vector<int> unique_ids;
  for (int id : ids) {
    if (std::find(unique_ids.begin(), unique_ids.end(), id) == unique_ids.end())
      unique_ids.push_back(id);
  }
  return unique_ids;
}

}  // namespace sankhya::gpu
