// SPDX-License-Identifier: Apache-2.0
// SANKHYA - column substitution from equality rows: the replay (substitute.hpp has the
// derivation and the citations).
//
// THE RECORDS ARE UNDONE IN STRICT REVERSE ORDER. Record p was taken on the model as it
// stood when p fired; every column its row held then and every row its column met then is
// either in the reduced model or was removed by a later record, which is undone first. So
// each step reads only values the steps before it have already restored, and a chain of
// substitutions needs nothing beyond what one substitution needs.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "presolve/dual_reduce.hpp"
#include "presolve/substitute.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {

namespace {

/// Is x on the bound `bound` that this record's carried box introduced (and the partner's
/// own box did not have)?
bool on_carried(double x, double bound, double own, bool lower) {
  if (!is_finite_bound(bound)) return false;
  if (is_finite_bound(own) && (lower ? bound <= own : bound >= own)) return false;
  return std::fabs(x - bound) <= tol::kPrimalFeasibility * std::max(1.0, std::fabs(bound));
}

double row_value(const Substitution& rec, const std::vector<double>& x) {
  double rest = 0.0;
  for (const auto& [k, a] : rec.row_entries) rest += a * x[static_cast<std::size_t>(k)];
  return (rec.rhs - rest) / rec.coefficient;
}

/// The nonbasic status of a column at value x in [lower, upper].
BasisStatus at_bound(double x, double lower, double upper) {
  if (lower == upper) return BasisStatus::kFixed;
  if (!is_finite_bound(upper)) return BasisStatus::kAtLower;
  if (!is_finite_bound(lower)) return BasisStatus::kAtUpper;
  return std::fabs(x - lower) <= std::fabs(x - upper) ? BasisStatus::kAtLower
                                                      : BasisStatus::kAtUpper;
}

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

/// Only the primal: every eliminated column from its row, newest record first. `ray` takes
/// the row's right-hand side as zero, which turns the same formula into a direction.
void undo_primal(const SubstitutionResult& result, std::vector<double>* x, bool ray) {
  for (auto it = result.records.rbegin(); it != result.records.rend(); ++it) {
    if (it->column < 0) continue;
    Substitution rec = *it;
    if (ray) rec.rhs = 0.0;
    (*x)[static_cast<std::size_t>(rec.column)] = row_value(rec, *x);
  }
}

/// A Farkas vector over the rows: y_r chosen so the eliminated column's combination is zero,
/// the stationarity condition with no cost.
void undo_farkas(const SubstitutionResult& result, std::vector<double>* y) {
  for (auto it = result.records.rbegin(); it != result.records.rend(); ++it) {
    double sum = 0.0;
    if (it->column >= 0) {
      for (const auto& [i, a] : it->column_entries)
        sum += a * (*y)[static_cast<std::size_t>(i)];
      sum = -sum / it->coefficient;
    }
    (*y)[static_cast<std::size_t>(it->row)] = sum;
  }
}

}  // namespace

Solution undo_substitution(const SubstitutionResult& result, const Model& input,
                           const Solution& reduced) {
  Solution out = reduced;
  const Index n = input.num_cols();
  const Index m = input.num_rows();
  // Vectors measured on the reduced model that this replay does not rebuild are dropped
  // rather than left the wrong length; every one of them is computed after postsolve.
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
    const Substitution& rec = *it;
    const auto ur = static_cast<std::size_t>(rec.row);
    if (rec.column < 0) {  // an emptied row
      if (has_y) y[ur] = 0.0;
      if (has_basis) rs[ur] = BasisStatus::kBasic;
      continue;
    }
    const auto ue = static_cast<std::size_t>(rec.column);
    if (has_x) x[ue] = row_value(rec, x);
    // Does the partner press on a bound only x_e's box gave it? Then x_e is the column on its
    // bound, and the row's dual moves to zero the partner's reduced cost instead.
    bool partner_carries = false;
    const auto uk = static_cast<std::size_t>(rec.carried ? rec.partner : 0);
    const bool partner_nonbasic = rec.carried && (!has_basis || cs[uk] != BasisStatus::kBasic);
    const auto carried_lower = [&] {
      return has_x && on_carried(x[uk], rec.partner_new_lower, rec.partner_lower, true);
    };
    const auto carried_upper = [&] {
      return has_x && on_carried(x[uk], rec.partner_new_upper, rec.partner_upper, false);
    };
    if (has_y) {
      // d = c - A^T y on the model before the record: zero for x_e first.
      double sum = rec.cost;
      for (const auto& [i, a] : rec.column_entries) sum -= a * y[static_cast<std::size_t>(i)];
      double y_r = sum / rec.coefficient;
      if (partner_nonbasic) {
        // The partner's reduced cost then, which is the one the reduced model reported
        // (substitute.hpp): its sign says which of the partner's bounds it presses on. A
        // bound its own box has as well keeps x_e basic; only a carried one moves the dual.
        double d_k = rec.partner_cost - rec.partner_coefficient * y_r;
        for (const auto& [i, a] : rec.partner_entries)
          d_k -= a * y[static_cast<std::size_t>(i)];
        const double pressing = input.sense_multiplier() * d_k;
        partner_carries =
            (pressing > 0.0 && carried_lower()) || (pressing < 0.0 && carried_upper());
        if (partner_carries) y_r += d_k / rec.partner_coefficient;
      }
      y[ur] = y_r;
    } else if (partner_nonbasic) {
      partner_carries = carried_lower() || carried_upper();
    }
    if (has_basis) {
      rs[ur] = BasisStatus::kFixed;  // an equality row
      if (partner_carries) {
        cs[static_cast<std::size_t>(rec.partner)] = BasisStatus::kBasic;
        cs[ue] = at_bound(has_x ? x[ue] : 0.0, rec.lower, rec.upper);
      } else {
        cs[ue] = BasisStatus::kBasic;
      }
    }
  }

  out.col_value = std::move(x);
  out.row_dual = y;
  out.col_status = std::move(cs);
  out.row_status = std::move(rs);
  if (has_y) {
    // Every reduced cost from the input model itself, d = c - A^T y.
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
  if (!reduced.primal_ray.empty()) {
    out.primal_ray = scatter(reduced.primal_ray, result.col_to_input, n, 0.0);
    if (!out.primal_ray.empty()) undo_primal(result, &out.primal_ray, true);
  }
  if (!reduced.farkas_dual.empty()) {
    out.farkas_dual = scatter(reduced.farkas_dual, result.row_to_input, m, 0.0);
    if (!out.farkas_dual.empty()) undo_farkas(result, &out.farkas_dual);
  }
  for (auto& member : out.pool) {
    member.col_value = scatter(member.col_value, result.col_to_input, n, 0.0);
    if (!member.col_value.empty()) undo_primal(result, &member.col_value, false);
  }
  out.recompute_quality(input);
  return out;
}

namespace {
std::atomic<std::int64_t> applied_count{0};
}  // namespace

std::int64_t substitutions_applied_for_testing() {
  return applied_count.load(std::memory_order_relaxed);
}

Solution solve_substituted(const Model& model, const Options& options, Logger& logger,
                           const std::function<Solution(const Model&)>& simplex) {
  if (!options.get_bool("presolve")) return simplex(model);
  const auto with_substitution = [&](const Model& lp) -> Solution {
    if (!options.get_bool("presolve_substitution")) return simplex(lp);
    const SubstitutionResult substituted = substitute(lp, options, logger);
    if (!substituted.reduced_anything()) return simplex(lp);
    applied_count.fetch_add(1, std::memory_order_relaxed);
    return undo_substitution(substituted, lp, simplex(substituted.model));
  };
  // The forcing-row and dominated-column stage first (dual_reduce.hpp): it only removes,
  // so the substitution sees a smaller model; undone last.
  if (!options.get_bool("presolve_dual_reductions")) return with_substitution(model);
  const DualReduceResult reduced = dual_reduce(model, options, logger);
  if (!reduced.reduced_anything()) return with_substitution(model);
  return undo_dual_reduce(reduced, model, with_substitution(reduced.model));
}

}  // namespace sankhya::presolve
