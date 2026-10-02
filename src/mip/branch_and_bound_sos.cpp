// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound on semi-continuous columns and special ordered sets (#754).
//
// References, written from the literature:
//   Beale and Tomlin, "Special facilities in a general mathematical programming system for
//     non-convex problems using ordered sets of variables", Proc. 5th IFORS Conference
//     (1970), 447-454 - SOS1 and SOS2, and branching on a set by splitting it at the
//     weighted centre of its members instead of on one member. The semi-continuous split
//     below is the same idea for a single variable whose domain is {0} or [l, u]: branch on
//     the gap in the domain, not on a value inside it
//   Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 5 - pseudocosts and
//     the product score, used here per set as they are per column
//
// THE SEARCH RUNS ON A RELAXATION (sc_sos_spec.hpp). The node LPs see every semi-continuous
// column in [0, u] and every set's members free of each other, so an LP optimum may sit at
// 0 < x < l, or spread a set over members that may not all be nonzero. Such a node is split
// in two, each side excluding the point and the two together keeping every point that meets
// the condition:
//
//   semi-continuous x, l:     x <= 0                 |   x >= l
//   SOS1 at split r:          x_k = 0 for k > r      |   x_k = 0 for k <= r
//   SOS2 at split r:          x_k = 0 for k > r      |   x_k = 0 for k < r
//
// (positions k in weight order). An SOS2 point is nonzero on at most {i, i+1}: with i+1 <= r
// it is on the left, with i >= r on the right. r is Beale and Tomlin's: the last position
// whose weight is at most the weighted centre sum w_k |x_k| / sum |x_k|, clamped so that
// both sides cut the point off (the first nonzero member <= r < the last, for SOS2 strictly
// between them). A branch therefore fixes at least one more nonzero member to 0, and the
// tree is finite.
//
// These branchings come before integer branching: a node whose point breaks a condition is
// split on it even when its integer columns are fractional too. Which condition is chosen is
// the pseudocost product score over the violated items, with each item's pseudocost the
// average objective gain per unit of |x| a side removed; an item never branched on yet
// borrows the average of those that have been, and 1 when none has.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../util/profiler.hpp"
#include "branch_and_bound_internal.hpp"
#include "core/sc_sos.hpp"

namespace sankhya::mip {

ScSosSpec ScSosSpec::take_from(Model* model) {
  ScSosSpec spec;
  for (const Index j : model->semicontinuous) {
    const auto u = static_cast<std::size_t>(j);
    spec.sc_columns.push_back(j);
    spec.sc_lower.push_back(model->col_lower[u]);
    model->col_lower[u] = std::min(0.0, model->col_lower[u]);
  }
  spec.sets = std::move(model->sos);
  model->semicontinuous.clear();
  model->sos.clear();
  return spec;
}

double ScSosSpec::violation(const std::vector<double>& x) const {
  double worst = 0.0;
  for (std::size_t k = 0; k < sc_columns.size(); ++k) {
    const double v = x[static_cast<std::size_t>(sc_columns[k])];
    worst = std::max(worst, semicontinuous_violation(v, sc_lower[k]));
  }
  for (const SosSet& set : sets) worst = std::max(worst, sos_violation(set, x.data()));
  return worst;
}

void BranchAndBound::attach_sc_sos(const ScSosSpec* spec) {
  sc_sos_ = spec;
  const auto items = static_cast<std::size_t>(spec->num_items());
  sc_sos_down_sum_.assign(items, 0.0);
  sc_sos_up_sum_.assign(items, 0.0);
  sc_sos_down_count_.assign(items, 0);
  sc_sos_up_count_.assign(items, 0);
  // CONFLICT ANALYSIS learns bound disjunctions on INTEGER columns (conflict.hpp): a literal
  // x <= v becomes x >= v + 1 when it is negated. The x <= 0 of a semi-continuous or SOS
  // branching is on a column that need not be integer, where that negation is wrong.
  if (conflicts_enabled_ || conflict_cutoff_) {
    logger_.info("Semi-continuous / SOS (#754): conflict analysis is off in this search");
  }
  conflicts_enabled_ = false;
  conflict_cutoff_ = false;
  // pool_complete partitions an integral node over its INTEGER columns alone, which is not
  // a partition of the points that meet the sets.
  if (pool_complete_) {
    logger_.info("Semi-continuous / SOS (#754): pool_complete is off in this search");
  }
  pool_complete_ = false;
}

void BranchAndBound::record_sc_sos_pseudocost(const TreeNode& node, double node_bound) {
  if (node.sc_sos_item < 0 || !(node.sc_sos_amount > 0.0)) return;
  const double parent_lp = std::isnan(node.parent_lp_bound) ? node.bound : node.parent_lp_bound;
  const double gain = node_bound - parent_lp;
  if (!std::isfinite(gain)) return;
  const auto item = static_cast<std::size_t>(node.sc_sos_item);
  const double per_unit = std::max(gain, 0.0) / node.sc_sos_amount;
  if (node.sc_sos_up) {
    sc_sos_up_sum_[item] += per_unit;
    ++sc_sos_up_count_[item];
  } else {
    sc_sos_down_sum_[item] += per_unit;
    ++sc_sos_down_count_[item];
  }
}

namespace {

/// The average pseudocost over the items that have one, 1 when none has.
double average_pseudocost(const std::vector<double>& sums, const std::vector<Count>& counts) {
  double total = 0.0;
  Count seen = 0;
  for (std::size_t k = 0; k < sums.size(); ++k) {
    if (counts[k] == 0) continue;
    total += sums[k] / static_cast<double>(counts[k]);
    ++seen;
  }
  return seen > 0 ? total / static_cast<double>(seen) : 1.0;
}

}  // namespace

bool BranchAndBound::branch_sc_sos(Index node_index, const Solution& relaxation,
                                   double node_bound, double prune_bound, const WarmStart& warm,
                                   const qp::QpIpmWarmStart& qp_warm) {
  const std::vector<double>& x = relaxation.col_value;
  const double tolerance = integrality_tolerance_;
  const Index sc_count = static_cast<Index>(sc_sos_->sc_columns.size());

  // The changes that set column c to 0 at this node, none when it already is.
  const auto zero = [&](Index c, std::vector<DomainChange>* changes) {
    const auto u = static_cast<std::size_t>(c);
    if (working_.col_upper[u] > 0.0) changes->push_back(DomainChange{c, true, 0.0});
    if (working_.col_lower[u] < 0.0) changes->push_back(DomainChange{c, false, 0.0});
  };

  struct Choice {
    Index item = -1;
    double down_amount = 0.0;
    double up_amount = 0.0;
    std::vector<DomainChange> down;
    std::vector<DomainChange> up;
  };
  Choice best;
  double best_score = -1.0;
  const double average_down = average_pseudocost(sc_sos_down_sum_, sc_sos_down_count_);
  const double average_up = average_pseudocost(sc_sos_up_sum_, sc_sos_up_count_);
  const auto consider = [&](Choice&& choice) {
    const auto k = static_cast<std::size_t>(choice.item);
    const double pc_down =
        sc_sos_down_count_[k] > 0
            ? sc_sos_down_sum_[k] / static_cast<double>(sc_sos_down_count_[k])
            : average_down;
    const double pc_up = sc_sos_up_count_[k] > 0
                             ? sc_sos_up_sum_[k] / static_cast<double>(sc_sos_up_count_[k])
                             : average_up;
    constexpr double kEpsilon = 1e-6;
    const double score = std::max(pc_down * choice.down_amount, kEpsilon) *
                         std::max(pc_up * choice.up_amount, kEpsilon);
    if (score > best_score) {
      best_score = score;
      best = std::move(choice);
    }
  };

  for (Index k = 0; k < sc_count; ++k) {
    const Index j = sc_sos_->sc_columns[static_cast<std::size_t>(k)];
    const double lower = sc_sos_->sc_lower[static_cast<std::size_t>(k)];
    const double v = x[static_cast<std::size_t>(j)];
    if (semicontinuous_violation(v, lower) <= tolerance) continue;
    Choice choice;
    choice.item = k;
    choice.down_amount = v;
    choice.up_amount = lower - v;
    choice.down.push_back(DomainChange{j, true, 0.0});
    choice.up.push_back(DomainChange{j, false, lower});
    consider(std::move(choice));
  }

  for (std::size_t s = 0; s < sc_sos_->sets.size(); ++s) {
    const SosSet& set = sc_sos_->sets[s];
    if (sos_violation(set, x.data()) <= tolerance) continue;
    const std::size_t n = set.columns.size();
    std::vector<double> magnitude(n);
    std::size_t first = n;
    std::size_t last = 0;
    double mass = 0.0;
    double moment = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      magnitude[p] = std::fabs(x[static_cast<std::size_t>(set.columns[p])]);
      mass += magnitude[p];
      moment += set.weights[p] * magnitude[p];
      if (magnitude[p] > tolerance) {
        first = std::min(first, p);
        last = p;
      }
    }
    // Violated means two members apart (SOS1) or three positions apart (SOS2) are nonzero.
    if (first >= n || last <= first || (set.type == 2 && last < first + 2)) continue;
    const double centre = moment / mass;
    std::size_t split = 0;
    for (std::size_t p = 0; p < n; ++p) {
      if (set.weights[p] <= centre) split = p;
    }
    // SOS1: first <= split < last. SOS2: first < split < last.
    const std::size_t low = set.type == 1 ? first : first + 1;
    split = std::clamp(split, low, last - 1);
    Choice choice;
    choice.item = sc_count + static_cast<Index>(s);
    for (std::size_t p = 0; p < n; ++p) {
      const bool left_zeroes = p > split;
      const bool right_zeroes = set.type == 1 ? p <= split : p < split;
      if (left_zeroes) {
        zero(set.columns[p], &choice.down);
        choice.down_amount += magnitude[p];
      }
      if (right_zeroes) {
        zero(set.columns[p], &choice.up);
        choice.up_amount += magnitude[p];
      }
    }
    // Each side zeroes a member that is nonzero at this point, and a member already fixed
    // at 0 is not nonzero, so neither list can be empty here.
    if (choice.down.empty() || choice.up.empty()) continue;
    consider(std::move(choice));
  }

  if (best.item < 0) return false;
  if (best.item < sc_count) {
    ++sc_branches_;
  } else {
    ++sos_branches_;
  }

  // Read before nodes_ grows: `nodes_[node_index]` is a reference into it.
  const Index depth = nodes_[static_cast<std::size_t>(node_index)].depth;
  const double estimate = estimate_from(x, node_bound);
  const auto warm_cuts = cut_layout();  // #497: the rows `warm` is over, when rows move
  leave();

  // A side with several changes hangs them on never-opened link nodes, the way
  // link_strong_fixes() does; enter() walks the chain to the root either way.
  const auto add_child = [&](const std::vector<DomainChange>& changes, bool up, double amount) {
    Index tail = node_index;
    for (std::size_t k = 0; k + 1 < changes.size(); ++k) {
      TreeNode link;
      link.parent = tail;
      link.has_change = true;
      link.change = changes[k];
      link.bound = prune_bound;
      link.depth = depth;
      nodes_.push_back(std::move(link));
      tail = static_cast<Index>(nodes_.size() - 1);
    }
    TreeNode child;
    child.parent = tail;
    child.has_change = true;
    child.change = changes.back();
    child.bound = prune_bound;
    child.parent_lp_bound = node_bound;
    child.depth = depth + 1;
    child.warm = warm;
    child.qp_warm = qp_warm;
    child.warm_cuts = warm_cuts;
    child.estimate = estimate;
    child.sc_sos_item = best.item;
    child.sc_sos_up = up;
    child.sc_sos_amount = amount;
    nodes_.push_back(std::move(child));
    push_open(static_cast<Index>(nodes_.size() - 1));
  };
  add_child(best.down, false, best.down_amount);
  add_child(best.up, true, best.up_amount);
  return true;
}

void BranchAndBound::report_sc_sos() const {
  if (sc_sos_ == nullptr) return;
  logger_.info(
      "Semi-continuous / SOS (#754): {} semi-continuous column(s), {} set(s); {} node(s) "
      "branched on a semi-continuous column, {} on a set",
      sc_sos_->sc_columns.size(), sc_sos_->sets.size(), sc_branches_, sos_branches_);
  if (Profiler* profiler = logger_.profiler(); profiler != nullptr) {
    profiler->count("semi-continuous branchings", static_cast<std::int64_t>(sc_branches_));
    profiler->count("SOS branchings", static_cast<std::int64_t>(sos_branches_));
  }
}

}  // namespace sankhya::mip
