// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: two more primal heuristics for #841, each off by default.
//
//   propagation dive  Achterberg, "Constraint Integer Programming", thesis, TU Berlin 2007,
//                     ch. 9 (diving with domain propagation), and Berthold, "Heuristic
//                     algorithms in global MINLP solvers", thesis, TU Berlin 2014, ch. 3
//                     (fix-and-propagate with backtracking): fix one integer column at a
//                     time to its rounded relaxation value, propagate the rows, and on a
//                     proved-empty box undo back to the latest decision that still has an
//                     untried value and take that one - a depth-first search with no LP
//                     until every integer column is fixed, bounded by a counted budget.
//   local branching   Fischetti and Lodi, "Local branching", Math. Programming 98 (2003):
//                     around a reference point x' with binary support S, the sub-MIP with
//                         sum_{j in B, x'_j = 0} x_j + sum_{j in S} (1 - x_j) <= k
//                     added, searched with a node limit; recentred on each improvement. The
//                     reference is the incumbent, or, with none, the rounded root
//                     relaxation (the infeasible-reference variant of Fischetti and Lodi,
//                     "Repairing MIP infeasibility through local branching", Computers &
//                     OR 35, 2008).
//
// Both propose; offer_incumbent() decides, against the original model.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

void BranchAndBound::propagation_dive(std::size_t slot, const std::vector<double>& guide) {
  HeuristicStats& s = heuristic_stats_[slot];
  const Timer clock;
  ++s.calls;
  const std::size_t mark = saved_.size();
  const auto value_of = [&](std::size_t u) {
    return u < guide.size() ? guide[u] : working_.col_lower[u];
  };
  // Least fractional first (Gamrath, Berthold, Heinz and Winkler 2016's order for
  // fix-and-propagate): the relaxation is surest of those, so they are the safest decisions.
  std::vector<Index> order = integer_columns_;
  std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {
    const double fa = std::fabs(value_of(static_cast<std::size_t>(a)) -
                                std::round(value_of(static_cast<std::size_t>(a))));
    const double fb = std::fabs(value_of(static_cast<std::size_t>(b)) -
                                std::round(value_of(static_cast<std::size_t>(b))));
    return fa < fb;
  });
  bool continuous = false;
  for (Index j = 0; j < original_.num_cols() && !continuous; ++j)
    continuous = original_.col_type[static_cast<std::size_t>(j)] != VarType::kInteger;

  struct Decision {
    std::size_t column;
    double other;      ///< the untried value; NaN when there is none
    std::size_t mark;  ///< saved_ before the fix
    std::size_t next;  ///< the position in `order` the fix was taken at
  };
  std::vector<Decision> stack;
  const Count nonzeros = std::max<Count>(1, original_.matrix.num_nonzeros());
  const Count budget = std::max<Count>(1, tol::kPropagationDiveWork / nonzeros);
  Count propagations = 0;
  std::size_t next = 0;
  const auto backtrack = [&]() {
    while (!stack.empty() && std::isnan(stack.back().other)) stack.pop_back();
    if (stack.empty()) return false;
    Decision& d = stack.back();
    unwind_to(d.mark);
    next = d.next;
    const double v = d.other;
    d.other = std::numeric_limits<double>::quiet_NaN();
    tighten_lower(d.column, v);
    tighten_upper(d.column, v);
    return true;
  };

  while (propagations < budget) {
    if (control_ != nullptr && control_->interruption_requested()) break;
    if (schedule_.seconds_budgets && limits_.time_exhausted(timer_.elapsed_seconds())) break;
    ++propagations;
    if (!propagate()) {
      if (!backtrack()) break;  // every decision tried both ways: the dive is exhausted
      continue;
    }
    while (next < order.size()) {
      const auto u = static_cast<std::size_t>(order[next]);
      if (working_.col_lower[u] != working_.col_upper[u]) break;
      ++next;
    }
    if (next == order.size()) {
      // Every integer column fixed: the continuous ones come from one LP, or there are none.
      std::vector<double> x = working_.col_lower;
      if (continuous) {
        const Solution lp = solve_node();
        if (lp.status != SolveStatus::kOptimal) {
          if (!backtrack()) break;
          continue;
        }
        x = lp.col_value;
      }
      (void)offer_from(slot, x);
      break;
    }
    const auto u = static_cast<std::size_t>(order[next]);
    const double lower = working_.col_lower[u];
    const double upper = working_.col_upper[u];
    const double target = std::clamp(value_of(u), lower, upper);
    const double v = std::clamp(std::round(target), lower, upper);
    double other = target >= v ? v + 1.0 : v - 1.0;
    if (other < lower || other > upper) other = target >= v ? v - 1.0 : v + 1.0;
    if (other < lower || other > upper) other = std::numeric_limits<double>::quiet_NaN();
    stack.push_back({u, other, saved_.size(), next});
    tighten_lower(u, v);
    tighten_upper(u, v);
  }
  s.work += propagations;
  unwind_to(mark);
  s.seconds += clock.elapsed_seconds();
}

void BranchAndBound::local_branching(std::size_t slot, const std::vector<double>& relaxation) {
  HeuristicStats& s = heuristic_stats_[slot];
  const Timer clock;
  ++s.calls;
  std::vector<Index> binaries;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    if (original_.col_lower[u] == 0.0 && original_.col_upper[u] == 1.0) binaries.push_back(j);
  }
  if (binaries.empty()) return;
  std::vector<double> reference;
  if (have_incumbent_) {
    reference = incumbent_x_;
  } else {
    reference = relaxation;
    for (const Index j : binaries) {
      const auto u = static_cast<std::size_t>(j);
      reference[u] = reference[u] >= 0.5 ? 1.0 : 0.0;
    }
  }
  for (int round = 0; round < tol::kLocalBranchingRounds; ++round) {
    if (control_ != nullptr && control_->interruption_requested()) break;
    if (schedule_.seconds_budgets && limits_.time_exhausted(timer_.elapsed_seconds())) break;
    // The neighbourhood row: + x_j where the reference is 0, - x_j where it is 1, at most
    // k minus the reference's support.
    const Index n = original_.num_cols();
    const Index m = original_.num_rows();
    Model sub = original_;
    sub.resize_rows(m + 1);
    double support = 0.0;
    SparseMatrix matrix(m + 1, n);
    for (Index j = 0; j < n; ++j) {
      const ColumnView column = original_.matrix.column(j);
      for (Index k = 0; k < column.size; ++k)
        matrix.add_entry(column.rows[k], j, column.values[k]);
    }
    for (const Index j : binaries) {
      const bool one = reference[static_cast<std::size_t>(j)] >= 0.5;
      matrix.add_entry(m, j, one ? -1.0 : 1.0);
      if (one) support += 1.0;
    }
    matrix.finalize(0.0);
    sub.matrix = std::move(matrix);
    sub.row_upper[static_cast<std::size_t>(m)] =
        static_cast<double>(tol::kLocalBranchingRadius) - support;
    Logger quiet(nullptr);
    const Solution found = solve_branch_and_bound(
        sub,
        sub_mip_options(
            options_, tol::kLocalBranchingNodes,
            sub_mip_seconds(limits_, timer_.elapsed_seconds(), schedule_.seconds_budgets)),
        quiet, control_);
    s.work += found.nodes;
    if (!claims_a_point(found.status) || found.col_value.empty()) break;
    std::vector<double> point(found.col_value.begin(),
                              found.col_value.begin() + static_cast<std::ptrdiff_t>(n));
    if (!offer_from(slot, point)) break;  // no better point in this neighbourhood
    reference = std::move(point);         // recentre on the improvement
  }
  s.seconds += clock.elapsed_seconds();
}

}  // namespace sankhya::mip
