// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: the parity heuristic (#841), scheduled. The GF(2) elimination
// is parity.cpp; this file turns each fixing it proposes into a sub-MIP, and offers what the
// sub-MIP finds to offer_incumbent(), the only way anything becomes the incumbent.

#include "branch_and_bound_internal.hpp"
#include "feasibility_jump.hpp"
#include "parallel_search.hpp"
#include "parity.hpp"

#include <cstdint>
#include <string>
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
  // ONLY A NULL SPACE THE CANDIDATES COVER WHOLE. With dimension k the rows allow 2^k
  // assignments of their binaries; when 2^k exceeds kParityCandidates the fixings tried are
  // an arbitrary corner of that space, chosen by column order, and each costs a sub-MIP on
  // the whole model. On neos8 (MIPLIB 2017) 12 parity rows over 780 binaries leave k = 768:
  // the 16 sub-MIPs took 38-42 s of a 57 s solve before the root LP, and the incumbent they
  // gave on two seeds the root rounding finds anyway (bench run 44010017090). On the
  // lights-out models the heuristic is for (#841) the rows pin the binaries down and k is
  // small. The elimination itself is kept, and logged, whatever k is.
  const bool whole = f.null_space_dimension < 31 &&
                     (Index{1} << f.null_space_dimension) <= Index{tol::kParityCandidates};
  // AND ONLY A FIXING THAT IS A NEIGHBOURHOOD: at least kSubMipMinFixedFraction of the
  // binaries, the rule RINS and RENS keep (Danna, Rothberg and Le Pape 2005; Berthold 2014).
  // Below it each sub-MIP is a second search of nearly the whole model. On hypothyroid-k1
  // (MIPLIB 2017) 2 parity rows over 3 of 2602 binaries leave k = 1, and the two sub-MIPs
  // took 11.3 s of 60 before the root LP, on all three seeds, and found nothing (bench run
  // 44348500088). On the lights-out models the parity rows cover every press.
  Index binaries = 0;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    if (original_.col_lower[u] >= 0.0 && original_.col_upper[u] <= 1.0) ++binaries;
  }
  const bool neighbourhood = static_cast<double>(f.columns.size()) >=
                             tol::kSubMipMinFixedFraction * static_cast<double>(binaries);
  std::size_t tried = 0;
  for (const std::vector<std::uint8_t>& bits : f.candidates) {
    if (!whole || !neighbourhood) break;
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
      "{} fixing(s) tried in {:.3f}s{}",
      f.parity_rows, f.columns.size(), f.null_space_dimension,
      f.inconsistent ? ", inconsistent mod 2" : "", tried, clock.elapsed_seconds(),
      !whole ? " (the null space is larger than the fixings can cover, so none is tried)"
      : !neighbourhood && !f.candidates.empty()
          ? fmt::format(" (they fix {} of {} binaries, under half, so none is tried)",
                        f.columns.size(), binaries)
          : std::string());
}

}  // namespace sankhya::mip
