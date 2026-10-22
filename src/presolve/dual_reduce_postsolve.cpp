// SPDX-License-Identifier: Apache-2.0
// SANKHYA - forcing rows and dominated columns: the replay (dual_reduce.hpp has the
// derivation). Records are undone newest first, each on the model as it stood when it fired.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "presolve/dual_reduce.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {

namespace {

template <typename T>
std::vector<T> scatter(const std::vector<T>& reduced, const std::vector<Index>& to_input,
                       Index size, T fill) {
  if (reduced.size() != to_input.size()) return {};
  std::vector<T> full(static_cast<std::size_t>(size), fill);
  for (std::size_t j = 0; j < to_input.size(); ++j) {
    full[static_cast<std::size_t>(to_input[j])] = reduced[j];
  }
  return full;
}

BasisStatus at_bound(double x, double lower, double upper) {
  if (lower == upper) return BasisStatus::kFixed;
  if (!is_finite_bound(upper)) return BasisStatus::kAtLower;
  if (!is_finite_bound(lower)) return BasisStatus::kAtUpper;
  return std::fabs(x - lower) <= std::fabs(x - upper) ? BasisStatus::kAtLower
                                                      : BasisStatus::kAtUpper;
}

std::atomic<std::int64_t> applied_count{0};

}  // namespace

std::int64_t dual_reductions_applied_for_testing() {
  return applied_count.load(std::memory_order_relaxed);
}

Solution undo_dual_reduce(const DualReduceResult& result, const Model& input,
                          const Solution& reduced) {
  applied_count.fetch_add(1, std::memory_order_relaxed);
  Solution out = reduced;
  const Index n = input.num_cols();
  const Index m = input.num_rows();
  out.exact_col_value.clear();
  out.exact_repair_farkas.clear();
  out.col_ranging_lower.clear();
  out.col_ranging_upper.clear();
  out.row_ranging_lower.clear();
  out.row_ranging_upper.clear();
  out.exact_col_sensitivity.clear();
  out.exact_row_sensitivity.clear();
  out.iis_rows.clear();
  out.iis_col_lo.clear();
  out.iis_col_hi.clear();
  out.iis_witnesses.clear();
  out.safe_multipliers.clear();
  out.safe_multipliers_exact.clear();
  out.safe_column_bounds.clear();

  std::vector<double> x = scatter(reduced.col_value, result.col_to_input, n, 0.0);
  std::vector<double> y = scatter(reduced.row_dual, result.row_to_input, m, 0.0);
  std::vector<BasisStatus> cs =
      scatter(reduced.col_status, result.col_to_input, n, BasisStatus::kUnknown);
  std::vector<BasisStatus> rs =
      scatter(reduced.row_status, result.row_to_input, m, BasisStatus::kUnknown);
  const bool has_x = !x.empty();
  const bool has_y = !y.empty();
  const bool has_basis = !cs.empty() && !rs.empty();

  for (auto it = result.records.rbegin(); it != result.records.rend(); ++it) {
    const DualReduction& rec = *it;
    switch (rec.kind) {
      case DualReduction::Kind::kRedundantRow: {
        const auto ur = static_cast<std::size_t>(rec.row);
        if (has_y) y[ur] = 0.0;
        if (has_basis) rs[ur] = BasisStatus::kBasic;
        break;
      }
      case DualReduction::Kind::kDominatedColumn: {
        const auto u = static_cast<std::size_t>(rec.column);
        if (has_x) x[u] = rec.value;
        if (has_basis) cs[u] = at_bound(rec.value, input.col_lower[u], input.col_upper[u]);
        break;
      }
      case DualReduction::Kind::kForcingRow: {
        const auto ur = static_cast<std::size_t>(rec.row);
        for (std::size_t p = 0; p < rec.columns.size(); ++p) {
          const auto u = static_cast<std::size_t>(rec.columns[p]);
          if (has_x) x[u] = rec.values[p];
          if (has_basis) {
            cs[u] = at_bound(rec.values[p], input.col_lower[u], input.col_upper[u]);
          }
        }
        if (!has_y) break;
        // y_r = max_j t_j (min_j when forced to the upper bound), with zero in the running
        // when the row's other side is open; the winning column goes basic.
        const bool take_max = rec.at_lower;
        double best = rec.equality ? (take_max ? -kInfinity : kInfinity) : 0.0;
        std::ptrdiff_t winner = -1;
        for (std::size_t p = 0; p < rec.columns.size(); ++p) {
          const auto u = static_cast<std::size_t>(rec.columns[p]);
          double rest = 0.0;
          const ColumnView view = input.matrix.column(rec.columns[p]);
          for (Index q = 0; q < view.size; ++q) {
            if (view.rows[q] == rec.row) continue;
            rest += view.values[q] * y[static_cast<std::size_t>(view.rows[q])];
          }
          const double t = (input.col_cost[u] - rest) / rec.coefficients[p];
          if (take_max ? t > best : t < best) {
            best = t;
            winner = static_cast<std::ptrdiff_t>(p);
          }
        }
        y[ur] = best;
        if (has_basis) {
          if (winner < 0) {
            rs[ur] = BasisStatus::kBasic;
          } else {
            cs[static_cast<std::size_t>(rec.columns[static_cast<std::size_t>(winner)])] =
                BasisStatus::kBasic;
            rs[ur] = rec.equality
                         ? BasisStatus::kFixed
                         : (rec.at_lower ? BasisStatus::kAtLower : BasisStatus::kAtUpper);
          }
        }
        break;
      }
    }
  }

  out.col_value = std::move(x);
  out.row_dual = y;
  out.col_status = std::move(cs);
  out.row_status = std::move(rs);
  if (has_y) {
    out.col_dual.assign(static_cast<std::size_t>(n), 0.0);
    for (Index j = 0; j < n; ++j) {
      double d = input.col_cost[static_cast<std::size_t>(j)];
      const ColumnView view = input.matrix.column(j);
      for (Index p = 0; p < view.size; ++p) {
        d -= view.values[p] * y[static_cast<std::size_t>(view.rows[p])];
      }
      out.col_dual[static_cast<std::size_t>(j)] = d;
    }
  } else {
    out.col_dual.clear();
  }
  // A ray or a Farkas vector of the reduced model, extended by zeros on what was removed: a
  // candidate solve() checks against the model before anyone sees it.
  if (!reduced.primal_ray.empty()) {
    out.primal_ray = scatter(reduced.primal_ray, result.col_to_input, n, 0.0);
  }
  if (!reduced.farkas_dual.empty()) {
    out.farkas_dual = scatter(reduced.farkas_dual, result.row_to_input, m, 0.0);
  }
  for (auto& member : out.pool) {
    member.col_value = scatter(member.col_value, result.col_to_input, n, 0.0);
    if (member.col_value.empty()) continue;
    for (const DualReduction& rec : result.records) {
      if (rec.kind == DualReduction::Kind::kDominatedColumn) {
        member.col_value[static_cast<std::size_t>(rec.column)] = rec.value;
      } else if (rec.kind == DualReduction::Kind::kForcingRow) {
        for (std::size_t p = 0; p < rec.columns.size(); ++p) {
          member.col_value[static_cast<std::size_t>(rec.columns[p])] = rec.values[p];
        }
      }
    }
  }
  out.recompute_quality(input);
  return out;
}

}  // namespace sankhya::presolve
