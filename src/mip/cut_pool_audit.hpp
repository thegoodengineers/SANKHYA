// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a test seam for the cut pool's row removal (#497).
//
// With mip_cut_pooling an aged cut row is deleted from the node LP and appended again when
// a node's LP point violates it. The search reports each removal and each re-addition here,
// with the cut's activity at the LP point that decided it and what the node LP looks like
// afterwards, so a test can confirm that a re-added cut was violated, that it came back as
// the same row, and that a removed one is gone. Empty (the default) costs one test of a
// std::function per event.
#pragma once

#include <cstddef>
#include <functional>

#include "sankhya/types.hpp"

namespace sankhya::mip {

struct CutPoolEvent {
  enum class Kind {
    kRemoved,  ///< aged out and deleted from the node LP
    kReadded   ///< violated at a node's LP point and appended again
  };
  Kind kind = Kind::kRemoved;
  /// Which pooled cut: its index in the order cuts were appended.
  std::size_t pool_index = 0;
  /// The cut's activity a x at the node LP point that decided the event, and its right-hand
  /// side: activity - rhs is the violation that brought a cut back.
  double activity = 0.0;
  double rhs = 0.0;
  /// Rows of the node LP after the event.
  Index lp_rows = 0;
  /// The row the cut is in the node LP after the event, or -1 when it is not one.
  Index row = -1;
  /// For kReadded: that row holds exactly the cut's coefficients and right-hand side.
  bool row_is_the_cut = false;
};

using CutPoolHook = std::function<void(const CutPoolEvent&)>;

/// The hook every removal and re-addition is reported to. Not thread-safe; set it before a
/// sequential solve and clear it after.
CutPoolHook& cut_pool_audit_for_testing();

}  // namespace sankhya::mip
