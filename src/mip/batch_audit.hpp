// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a test seam for the batched PDHG's prunes (#520).
//
// #520's first acceptance item: every pruning by a batched bound re-checked by the rational
// oracle. The search reports each such prune here, with the box the bound was computed for,
// so a test can solve that box exactly and confirm that nothing better than the cutoff was
// thrown away. Empty (the default) costs one test of a std::function per prune.
#pragma once

#include <functional>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::mip {

struct BatchPruneRecord {
  /// A strong-branching child closed by its batched bound, rather than an open node.
  bool strong_branching = false;
  /// The box the bound holds for, in the search's (presolved) columns.
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  /// The batched safe bound, minimise space, without the objective offset.
  double bound = 0.0;
  /// What it was compared with: the incumbent's objective (minimise space, no offset), or
  /// the pool cutoff when filling a complete pool.
  double cutoff = 0.0;
  /// can_prune()'s margin: the bound, rounded up to the objective step when one is known,
  /// was at least cutoff - margin.
  double margin = 0.0;
  double objective_step = 0.0;
};

using BatchPruneHook = std::function<void(const BatchPruneRecord&)>;

/// The hook every batched prune is reported to. Not thread-safe; set it before a sequential
/// solve and clear it after.
BatchPruneHook& batch_prune_audit_for_testing();

}  // namespace sankhya::mip
