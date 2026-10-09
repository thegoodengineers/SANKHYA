// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: the parity heuristic (#841), scheduled. The GF(2) elimination
// is parity.cpp; this file turns each fixing it proposes into a sub-MIP, and offers what the
// sub-MIP finds to offer_incumbent(), the only way anything becomes the incumbent.

#include "branch_and_bound_internal.hpp"
#include "feasibility_jump.hpp"
#include "parallel_search.hpp"
#include "parity.hpp"

#include <cstdint>
#include <vector>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

// Once, before the root LP, like Feasibility Jump: on a model whose parity rows pin its
// binaries down, the fixing IS the answer's binary part, and an incumbent now prunes from
// the first node. The free columns of the elimination start at the box point closest to
// zero (feasibility_jump_zero_start), the cheapest guess on a model that minimises presses.
void BranchAndBound::run_parity_in(std::size_t slot) {
  if (!schedule_.parity || integer_columns_.empty()) return;
  if (seed_ != nullptr && !seed_->is_root) return;
  HeuristicStats& s = heuristic_stats_[slot];
  const Timer clock;
  ++s.calls;
  const ParityFixings f =
      parity_fixings(original_, feasibility_jump_zero_start(original_), tol::kParityCandidates);
  std::size_t tried = 0;
  for (const std::vector<std::uint8_t>& bits : f.candidates) {
    if (control_ != nullptr && control_->interruption_requested()) break;
    if (schedule_.seconds_budgets && limits_.time_exhausted(timer_.elapsed_seconds())) break;
    ++tried;
    Model restricted = original_;
    for (std::size_t k = 0; k < f.columns.size(); ++k) {
      const auto u = static_cast<std::size_t>(f.columns[k]);
      restricted.col_lower[u] = restricted.col_upper[u] = static_cast<double>(bits[k]);
    }
    Logger quiet(nullptr);
    const Solution found = solve_branch_and_bound(
        restricted,
        sub_mip_options(
            options_, tol::kParitySubMipNodes,
            sub_mip_seconds(limits_, timer_.elapsed_seconds(), schedule_.seconds_budgets)),
        quiet, control_);
    s.work += found.nodes;
    if (claims_a_point(found.status) && !found.col_value.empty()) {
      (void)offer_from(slot, found.col_value);
    }
  }
  s.seconds += clock.elapsed_seconds();
  logger_.info(
      "Parity (#841): {} parity row(s) over {} binary column(s), null space of dimension {}{}; "
      "{} fixing(s) tried in {:.3f}s",
      f.parity_rows, f.columns.size(), f.null_space_dimension,
      f.inconsistent ? ", inconsistent mod 2" : "", tried, clock.elapsed_seconds());
}

}  // namespace sankhya::mip
