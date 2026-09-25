// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a test seam that observes every accepted PDHG iterate (#479).
//
// The two-mat-vec path (pdhg_two_matvec) is held to the three-product path iterate by
// iterate, not only at the answer, and the reported solution is the best evaluated point,
// not the iterate. This hook hands a test the SCALED iterate each engine has just committed.
// It is null in every production solve.
#pragma once

#include <cstddef>

#include "sankhya/types.hpp"

namespace sankhya::pdhg {

/// Called after every accepted step with the iterate just committed: `iteration` is the
/// accepted count (1 for the first step), x has n entries and y has m, both in the
/// engine's scaled space. With pdhg_two_matvec, `ax_cached` is the A x_k the next step will
/// derive from and `ax_fresh` is A x_k computed afresh for the test (m entries each); both
/// are null on the three-product path.
using IterateTrace = void (*)(void* context, Count iteration, const double* x, std::size_t n,
                              const double* y, std::size_t m, const double* ax_cached,
                              const double* ax_fresh);

struct IterateTraceHook {
  IterateTrace callback = nullptr;
  void* context = nullptr;
};

/// The process-wide hook. A test sets it around a single-engine solve and clears it after;
/// the CPU engine (pdhg.cpp) and the CUDA per-iteration loop (pdhg_gpu.cu) call it. The
/// CUDA device loop (#478) does not: it has no host step to observe.
IterateTraceHook& iterate_trace_for_testing();

}  // namespace sankhya::pdhg
