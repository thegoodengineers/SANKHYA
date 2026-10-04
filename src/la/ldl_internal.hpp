// SPDX-License-Identifier: Apache-2.0
// SANKHYA - helpers shared by the two orderings in ldl.cpp and ldl_amd.cpp.
#pragma once

#include <cstddef>

#include "la/ldl.hpp"

namespace sankhya {
namespace ldl_detail {

/// A deadline asked in proportion to the work done rather than once per row or column
/// (#468). Every O(nnz) pass over a matrix - the assembly of the normal equations, the
/// set-up of the ordering's graph, the permuted pattern, the elimination tree, the refresh of
/// the values before a factorization - costs what its rows or columns hold, and one dense
/// row or column makes a fixed count of them an unbounded amount of work. The caller adds
/// what each step did; the predicate is asked once the total passes kWorkPerCheck, about
/// sixty-five thousand operations, which is tens of microseconds and so far below any time
/// limit a caller can set, and far above the cost of reading a clock. The first call always
/// asks, so a deadline that has already passed is seen before any work is done. Without a
/// predicate it never stops and changes nothing.
class DeadlineByWork {
 public:
  explicit DeadlineByWork(const SparseLdl::ShouldStop& should_stop)
      : should_stop_(should_stop) {}

  [[nodiscard]] bool expired(std::size_t done) {
    work_ += done;
    if (work_ < kWorkPerCheck || !should_stop_) return false;
    work_ = 0;
    return should_stop_();
  }

 private:
  static constexpr std::size_t kWorkPerCheck = std::size_t{1} << 16;
  const SparseLdl::ShouldStop& should_stop_;
  std::size_t work_ = kWorkPerCheck;
};

}  // namespace ldl_detail
}  // namespace sankhya
