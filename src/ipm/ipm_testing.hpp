// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a fault-injection seam for the LP interior point's tests (review of #473).
//
// The recovery from a non-finite Newton direction (#209) is reached on a real model only
// when its factorization happens to overflow in the solve, which a small committed model
// does not do on every path. This seam lets a test poison the next predictor directions, so
// the recovery runs on any model the test chooses. It is zero in every real solve and only
// tests write it; nothing in src/ sets it.
#pragma once

#include <atomic>

namespace sankhya::ipm::testing {

/// While positive, the next predictor direction the interior point computes has a NaN
/// written into it (and the count drops by one), so the step is rejected as non-finite
/// exactly as an overflowing solve would be.
inline std::atomic<int> poison_next_directions{0};

/// Takes one pending poisoning, if there is one.
inline bool take_poisoned_direction() {
  int left = poison_next_directions.load(std::memory_order_relaxed);
  while (left > 0 && !poison_next_directions.compare_exchange_weak(left, left - 1,
                                                                   std::memory_order_relaxed)) {
  }
  return left > 0;
}

}  // namespace sankhya::ipm::testing
