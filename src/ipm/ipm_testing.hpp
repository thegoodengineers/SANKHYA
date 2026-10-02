// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a fault-injection seam for the LP interior point's tests (review of #473).
//
// The recovery from a non-finite Newton direction (#209) is reached on a real model only
// when its factorization happens to overflow in the solve, which a small committed model
// does not do on every path. This seam lets a test poison the next predictor directions, so
// the recovery runs on any model the test chooses. It is zero in every real solve and only
// tests write it; nothing in src/ sets it. The same holds for the observer of the
// dense-column solves (#467), through which a test sees every system that path solves.
#pragma once

#include <atomic>
#include <functional>
#include <vector>

namespace sankhya::ipm {
class DenseColumnCorrection;
struct PcgReport;
}  // namespace sankhya::ipm

namespace sankhya::ipm::testing {

/// Called after every Newton-direction solve on the dense-column path (#467) with the
/// correction (which holds A, Theta, the row shift and delta of the system it solved), the
/// right-hand side, the answer and the conjugate-gradient report, so that a test can solve
/// the same system the default way and compare. Empty in every real solve; only tests set
/// it, and they clear it again before returning.
using DenseSolveObserver =
    std::function<void(const DenseColumnCorrection&, const std::vector<double>& rhs,
                       const std::vector<double>& solution, const PcgReport&)>;
inline DenseSolveObserver dense_solve_observer;

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

/// While positive, the next column-side solve (#469, ipm_normal_side) is reported as not
/// converged whatever its backward error (and the count drops by one), so a test reaches
/// the handling of an unconverged conjugate-gradient solve on any model.
inline std::atomic<int> reject_next_column_side_solves{0};

inline bool take_rejected_column_side_solve() {
  int left = reject_next_column_side_solves.load(std::memory_order_relaxed);
  while (left > 0 && !reject_next_column_side_solves.compare_exchange_weak(
                         left, left - 1, std::memory_order_relaxed)) {
  }
  return left > 0;
}

}  // namespace sankhya::ipm::testing
