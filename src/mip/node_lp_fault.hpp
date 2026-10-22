// SPDX-License-Identifier: Apache-2.0
// SANKHYA - test-only fault injection for node LPs (#733).
#pragma once

#include <atomic>

namespace sankhya::mip {

/// While positive, each non-root node LP that solves is reported numerical_error instead and
/// the count drops by one. Zero (the default) changes nothing. Tests only: it is how the
/// "node LP failed numerically" path of the tree is reached without a real phase-1 stall.
inline std::atomic<long long> fail_next_node_lps_for_testing{0};

}  // namespace sankhya::mip
