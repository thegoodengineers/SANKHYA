// SPDX-License-Identifier: Apache-2.0
// SANKHYA - semi-continuous columns and special ordered sets (#754): validation and the
// violation measures. See sc_sos.hpp.
//
// Reference: Beale and Tomlin, "Special facilities in a general mathematical programming
// system for non-convex problems using ordered sets of variables", Proc. 5th IFORS
// Conference (1970), 447-454 - the two set types and what "adjacent" means for type 2.

#include "core/sc_sos.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

#include <fmt/format.h>

namespace sankhya {

double semicontinuous_violation(double x, double lower) {
  if (x <= 0.0 || x >= lower) return 0.0;
  return std::min(x, lower - x);
}

double sos_violation(const SosSet& set, const double* x) {
  const std::size_t n = set.columns.size();
  const auto magnitude = [&](std::size_t k) {
    return std::fabs(x[static_cast<std::size_t>(set.columns[k])]);
  };
  if (set.type == 1) {
    double largest = 0.0;
    double second = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      const double v = magnitude(k);
      if (v > largest) {
        second = largest;
        largest = v;
      } else if (v > second) {
        second = v;
      }
    }
    return second;
  }
  // Type 2: the members outside the best adjacent pair must vanish. prefix[k] is the
  // largest magnitude among members 0..k-1, suffix[k] among members k..n-1.
  if (n <= 2) return 0.0;
  std::vector<double> prefix(n + 1, 0.0);
  std::vector<double> suffix(n + 1, 0.0);
  for (std::size_t k = 0; k < n; ++k) prefix[k + 1] = std::max(prefix[k], magnitude(k));
  for (std::size_t k = n; k-- > 0;) suffix[k] = std::max(suffix[k + 1], magnitude(k));
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k + 1 < n; ++k)
    best = std::min(best, std::max(prefix[k], suffix[k + 2]));
  return best;
}

double semicontinuous_and_sos_violation(const Model& model, const double* x) {
  double worst = 0.0;
  for (const Index j : model.semicontinuous) {
    const auto u = static_cast<std::size_t>(j);
    worst = std::max(worst, semicontinuous_violation(x[u], model.col_lower[u]));
  }
  for (const SosSet& set : model.sos) worst = std::max(worst, sos_violation(set, x));
  return worst;
}

SosSet sorted_by_weight(const SosSet& set) {
  std::vector<std::size_t> order(set.columns.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return set.weights[a] < set.weights[b];
  });
  SosSet sorted;
  sorted.type = set.type;
  sorted.name = set.name;
  sorted.priority = set.priority;
  for (const std::size_t k : order) {
    sorted.columns.push_back(set.columns[k]);
    sorted.weights.push_back(set.weights[k]);
  }
  return sorted;
}

std::vector<char> semicontinuous_mask(const Model& model) {
  std::vector<char> mask;
  if (model.semicontinuous.empty()) return mask;
  mask.assign(static_cast<std::size_t>(model.num_cols()), 0);
  for (const Index j : model.semicontinuous) {
    if (j >= 0 && j < model.num_cols()) mask[static_cast<std::size_t>(j)] = 1;
  }
  return mask;
}

std::string validate_semicontinuous_and_sos(const Model& model) {
  const Index n = model.num_cols();
  Index previous = -1;
  for (const Index j : model.semicontinuous) {
    if (j < 0 || j >= n)
      return fmt::format("semi-continuous column index {} is out of range", j);
    if (j <= previous) return "the semi-continuous column list is not ascending and unique";
    previous = j;
    const double lower = model.col_lower[static_cast<std::size_t>(j)];
    // x = 0 or l <= x <= u. A negative l would put 0 inside the run range, and the files this
    // reads (MPS SC, LP semi-continuous) define the variable for l >= 0 only.
    if (!(lower >= 0.0)) {
      return fmt::format(
          "semi-continuous column {} has lower bound {:g}; a semi-continuous column needs a "
          "lower bound of at least 0",
          j, lower);
    }
  }
  for (std::size_t s = 0; s < model.sos.size(); ++s) {
    const SosSet& set = model.sos[s];
    const std::string label = set.name.empty() ? fmt::format("#{}", s) : set.name;
    if (set.type != 1 && set.type != 2) {
      return fmt::format("special ordered set {} has type {}; only 1 and 2 exist", label,
                         static_cast<int>(set.type));
    }
    if (set.columns.size() != set.weights.size()) {
      return fmt::format("special ordered set {} has {} members and {} weights", label,
                         set.columns.size(), set.weights.size());
    }
    if (std::isnan(set.priority))
      return fmt::format("special ordered set {} has a NaN priority", label);
    std::vector<Index> seen = set.columns;
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
      return fmt::format("special ordered set {} lists a column twice", label);
    }
    for (std::size_t k = 0; k < set.columns.size(); ++k) {
      const Index j = set.columns[k];
      if (j < 0 || j >= n) {
        return fmt::format("special ordered set {} names column index {}, out of range", label,
                           j);
      }
      if (!std::isfinite(set.weights[k])) {
        return fmt::format("special ordered set {} has a weight that is not finite", label);
      }
      if (k > 0 && !(set.weights[k] > set.weights[k - 1])) {
        return fmt::format(
            "special ordered set {} has weights that are not strictly increasing; the weights "
            "define the members' order, and two equal weights leave it undefined",
            label);
      }
    }
  }
  return {};
}

}  // namespace sankhya
