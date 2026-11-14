// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover cuts (#419, #1158). The references and the validity argument are on
// the declaration; this file is the bookkeeping: reading a base inequality as a single-node
// flow set, the cover, the (lifted) inequality, and the walk over aggregated rows.

#include "flow_cover_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "row_aggregation.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

/// Violation, relative to the cut's right-hand side, below which the cut is not worth a row.
constexpr double kMinViolation = 1e-4;
/// Largest ratio between the largest and smallest coefficient a cut may carry, the same
/// numerical filter the other families apply before proposing a cut.
constexpr double kMaxDynamism = 1e6;
constexpr double kInf = std::numeric_limits<double>::infinity();

/// x <= capacity * y, read off a two-nonzero row.
struct Vub {
  Index indicator = -1;
  double capacity = 0.0;
};

/// One column of a flow row as an arc: a flow x' = scale * (x - shift), or
/// scale * (shift - x) when complemented, in [0, capacity], switched by `indicator`
/// (-1: always on), on the inflow or the outflow side of the row.
struct Arc {
  Index column = -1;
  double scale = 0.0;
  double shift = 0.0;
  bool complemented = false;
  Index indicator = -1;
  double capacity = kInf;
  double flow = 0.0;    ///< x' at the LP point
  double y_star = 1.0;  ///< the indicator at the LP point; 1 when always on
  bool inflow = true;
};

/// A base inequality read as a single-node flow set: sum of inflows - outflows <= b.
struct FlowSet {
  std::vector<Arc> arcs;
  double b = 0.0;
};

bool is_binary(const Model& model, Index j, const std::vector<double>& lower,
               const std::vector<double>& upper) {
  const auto u = static_cast<std::size_t>(j);
  return model.col_type[u] == VarType::kInteger && lower[u] == 0.0 && upper[u] == 1.0;
}

/// The variable upper bounds x <= u y the model states as rows: two nonzeros, a continuous
/// column with lower bound 0 and a binary, and a bound of 0 on the side that reads
/// a x - a' y <= 0 with a, a' > 0. A column with several keeps the tightest.
std::vector<Vub> find_vubs(const Model& model, const std::vector<double>& col_lower,
                           const std::vector<double>& col_upper, std::vector<char>* is_vub_row,
                           int* rows_found) {
  std::vector<Vub> vub(static_cast<std::size_t>(model.num_cols()));
  const CsrView by_row(model.matrix);
  for (Index i = 0; i < model.num_rows(); ++i) {
    const ColumnView row = by_row.row(i);
    if (row.size != 2) continue;
    const auto ui = static_cast<std::size_t>(i);
    for (int side = 0; side < 2; ++side) {
      const double bound = side == 0 ? model.row_upper[ui] : model.row_lower[ui];
      if (!is_finite_bound(bound) || bound != 0.0) continue;
      const double sign = side == 0 ? 1.0 : -1.0;  // sign * (a x + a' y) <= 0
      for (int k = 0; k < 2; ++k) {
        const Index x = row.rows[k];
        const Index y = row.rows[1 - k];
        const double ax = sign * row.values[k];
        const double ay = sign * row.values[1 - k];
        if (!(ax > tol::kZeroDrop && ay < -tol::kZeroDrop)) continue;
        const auto ux = static_cast<std::size_t>(x);
        if (model.col_type[ux] == VarType::kInteger || col_lower[ux] != 0.0) continue;
        if (!is_binary(model, y, col_lower, col_upper)) continue;
        const double capacity = -ay / ax;
        Vub& slot = vub[ux];
        if (slot.indicator < 0 || capacity < slot.capacity) slot = Vub{y, capacity};
        if ((*is_vub_row)[ui] == 0) ++*rows_found;
        (*is_vub_row)[ui] = 1;
      }
    }
  }
  return vub;
}

/// The superadditive lifting function (flow_cover_cuts.hpp, LIFTING): a staircase of period
/// `u1`, flat at h lambda then rising with slope 1 over the last lambda of each period.
double staircase(double z, double u1, double lambda) {
  const double periods = std::floor(z / u1);
  const double within = z - periods * u1;
  return lambda * periods + std::max(0.0, within - (u1 - lambda));
}

/// Add `sign` times the arc's flow x' to the LEFT side of `cut`, the constant of the
/// shift going to the right.
void add_flow(const Arc& arc, double sign, Cut* cut, double* rhs) {
  const auto u = static_cast<std::size_t>(arc.column);
  if (arc.complemented) {
    // x' = scale * (shift - x)
    cut->coeff[u] -= sign * arc.scale;
    *rhs -= sign * arc.scale * arc.shift;
  } else {
    // x' = scale * (x - shift)
    cut->coeff[u] += sign * arc.scale;
    *rhs += sign * arc.scale * arc.shift;
  }
}

/// `base` (sum values_j x_j <= rhs) as a flow set, or nothing: a free column has no flow to
/// speak of, and a set with no switched inflow is not this family's.
std::optional<FlowSet> read_flow_set(const Model& model, const Solution& solution,
                                     const std::vector<double>& col_lower,
                                     const std::vector<double>& col_upper,
                                     const std::vector<Vub>& vub, const BaseRow& base) {
  FlowSet set;
  set.b = base.rhs;
  bool any_switched_inflow = false;
  for (std::size_t k = 0; k < base.columns.size(); ++k) {
    const Index j = base.columns[k];
    const auto uj = static_cast<std::size_t>(j);
    const double coef = base.values[k];
    if (std::fabs(coef) <= tol::kZeroDrop) continue;  // an eliminated column
    const double lo = col_lower[uj];
    const double hi = col_upper[uj];
    const double x = solution.col_value[uj];
    Arc arc;
    arc.column = j;
    arc.scale = std::fabs(coef);
    if (is_binary(model, j, col_lower, col_upper)) {
      // A binary in the flow row is its own switch: x' = |a| z with capacity |a|.
      arc.indicator = j;
      arc.capacity = arc.scale;
      arc.flow = arc.scale * x;
      arc.y_star = x;
      arc.inflow = coef > 0.0;
    } else if (model.col_type[uj] != VarType::kInteger && vub[uj].indicator >= 0) {
      double capacity = vub[uj].capacity;
      if (is_finite_bound(hi)) capacity = std::min(capacity, hi);
      arc.indicator = vub[uj].indicator;
      arc.capacity = arc.scale * capacity;
      arc.flow = arc.scale * x;
      arc.y_star = solution.col_value[static_cast<std::size_t>(arc.indicator)];
      arc.inflow = coef > 0.0;
      if (arc.inflow) any_switched_inflow = true;
    } else if (is_finite_bound(lo)) {
      // Always on, measured from its lower bound; a general integer is a flow too.
      arc.shift = lo;
      set.b -= coef * lo;
      arc.capacity = is_finite_bound(hi) ? arc.scale * (hi - lo) : kInf;
      arc.flow = arc.scale * (x - lo);
      arc.inflow = coef > 0.0;
    } else if (is_finite_bound(hi)) {
      // Bounded above only: x = hi - x'', which turns the arc around.
      arc.complemented = true;
      arc.shift = hi;
      set.b -= coef * hi;
      arc.capacity = kInf;
      arc.flow = arc.scale * (hi - x);
      arc.inflow = coef < 0.0;
    } else {
      return std::nullopt;
    }
    set.arcs.push_back(arc);
  }
  if (!any_switched_inflow || !std::isfinite(set.b)) return std::nullopt;
  return set;
}

/// The flow cover cut of `set` at the LP point: the cover, then the unlifted and the
/// lifted inequality, the more violated kept. `violation` is relative to the right-hand
/// side; nothing when no cover with lambda > 0 exists, the cut is not violated, or it fails
/// the dynamism filter. `covers` counts a cover found, `lifted` says the lifted form won.
///
/// `always_on_in_cover` false leaves the always-on inflows out of the cover's first pass
/// (they still enter if the switched arcs alone do not cover b). An always-on arc costs the
/// cover nothing at the LP point but adds its whole capacity to lambda, and a loose bound
/// from presolve's propagation or probing makes lambda swallow every (u_j - lambda)^+ term:
/// on modglob that took the round-1 covers from 58 to 48 and with them the proof (#1158).
/// The caller separates both covers and keeps the more violated cut.
std::optional<Cut> separate_flow_set(const Model& model, const FlowSet& set, bool lift,
                                     bool always_on_in_cover, int* covers, bool* lifted,
                                     double* violation) {
  const std::vector<Arc>& arcs = set.arcs;
  const double b = set.b;
  const Index n = model.num_cols();
  // THE COVER: the inflows whose switch is on at the LP point (at least a half; an
  // always-on arc counts as on), by switch value then capacity then column so a rerun
  // builds the same cover, extended by the next-best inflows until the capacity exceeds
  // b. Only arcs with a finite capacity can be in it.
  std::vector<std::size_t> ranked;
  for (std::size_t k = 0; k < arcs.size(); ++k) {
    if (arcs[k].inflow && std::isfinite(arcs[k].capacity)) ranked.push_back(k);
  }
  std::sort(ranked.begin(), ranked.end(), [&arcs](std::size_t p, std::size_t q) {
    if (arcs[p].y_star != arcs[q].y_star) return arcs[p].y_star > arcs[q].y_star;
    if (arcs[p].capacity != arcs[q].capacity) return arcs[p].capacity > arcs[q].capacity;
    return arcs[p].column < arcs[q].column;
  });
  std::vector<char> in_cover(arcs.size(), 0);
  double covered = 0.0;
  for (const std::size_t k : ranked) {
    if (arcs[k].y_star < 0.5) continue;
    if (!always_on_in_cover && arcs[k].indicator < 0) continue;
    in_cover[k] = 1;
    covered += arcs[k].capacity;
  }
  const double margin = 1e-9 * std::max(1.0, std::fabs(b));
  double lambda = covered - b;
  for (const std::size_t k : ranked) {
    if (lambda > margin) break;
    if (in_cover[k]) continue;
    in_cover[k] = 1;
    covered += arcs[k].capacity;
    lambda = covered - b;
  }
  if (lambda <= margin) return std::nullopt;
  ++*covers;

  // u1, the largest capacity of a cover arc above lambda, is the lifting function's
  // period; with none the function is z itself and lifting adds nothing over leaving the
  // arcs out.
  double u1 = 0.0;
  for (std::size_t k = 0; k < arcs.size(); ++k) {
    if (in_cover[k] && arcs[k].capacity > lambda) u1 = std::max(u1, arcs[k].capacity);
  }

  // THE INEQUALITY, in the model's own columns, and its violation at the LP point
  // relative to its right-hand side (NaN when it fails the filter). Lifted, the arcs
  // outside the cover get the LIFTING coefficients, and every outflow sits on the right
  // as its own flow, which the lifting's validity argument needs.
  const auto build = [&](bool lifting, Cut* cut, int* lifted_arcs) {
    cut->family = CutFamily::kFlowCover;
    cut->coeff.assign(static_cast<std::size_t>(n), 0.0);
    double rhs = b;
    double lhs_star = 0.0;
    double rhs_star = b;
    *lifted_arcs = 0;
    for (std::size_t k = 0; k < arcs.size(); ++k) {
      const Arc& arc = arcs[k];
      if (arc.inflow) {
        if (!in_cover[k]) {
          // Outside the cover: zero unlifted; lifted, x' - (u - g(u)) y, kept only when
          // it is positive at the LP point (zero is valid too, so the choice is free).
          if (!lifting || !std::isfinite(arc.capacity)) continue;
          const double beta = arc.capacity - staircase(arc.capacity, u1, lambda);
          if (arc.flow - beta * arc.y_star <= tol::kZeroDrop) continue;
          add_flow(arc, 1.0, cut, &rhs);
          if (arc.indicator >= 0) {
            cut->coeff[static_cast<std::size_t>(arc.indicator)] -= beta;
          } else {
            rhs += beta;  // always on: the switch is the constant 1
          }
          lhs_star += arc.flow - beta * arc.y_star;
          ++*lifted_arcs;
          continue;
        }
        add_flow(arc, 1.0, cut, &rhs);
        lhs_star += arc.flow;
        const double extra = std::max(0.0, arc.capacity - lambda);
        if (extra > tol::kZeroDrop && arc.indicator >= 0) {
          // (u_j - lambda)^+ (1 - y_j): the constant to the right, -y_j on the left.
          cut->coeff[static_cast<std::size_t>(arc.indicator)] -= extra;
          rhs -= extra;
          lhs_star += extra * (1.0 - arc.y_star);
        }
      } else if (!lifting && lambda * arc.y_star < arc.flow) {
        // An outflow on the right as lambda times its switch, the smaller form here.
        if (arc.indicator >= 0) {
          cut->coeff[static_cast<std::size_t>(arc.indicator)] -= lambda;
        } else {
          rhs += lambda;
        }
        rhs_star += lambda * arc.y_star;
      } else {
        // An outflow on the right as the flow itself.
        add_flow(arc, -1.0, cut, &rhs);
        rhs_star += arc.flow;
      }
    }
    cut->rhs = rhs;
    const double v = (lhs_star - rhs_star) / std::max(1.0, std::fabs(rhs_star));
    if (v <= kMinViolation) return std::numeric_limits<double>::quiet_NaN();
    double largest = 0.0;
    double smallest = kInf;
    for (const double c : cut->coeff) {
      const double magnitude = std::fabs(c);
      if (magnitude <= tol::kZeroDrop) continue;
      largest = std::max(largest, magnitude);
      smallest = std::min(smallest, magnitude);
    }
    if (largest == 0.0 || largest / smallest > kMaxDynamism) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return v;
  };

  Cut plain;
  int unused = 0;
  const double plain_violation = build(false, &plain, &unused);
  Cut lifted_cut;
  int lifted_arcs = 0;
  const double lifted_violation = lift && u1 > 0.0 ? build(true, &lifted_cut, &lifted_arcs)
                                                   : std::numeric_limits<double>::quiet_NaN();
  // The lifted cut when it lifted something and is the more violated; NaN compares false.
  if (lifted_arcs > 0 && !(lifted_violation <= plain_violation)) {
    if (std::isnan(lifted_violation)) return std::nullopt;
    *lifted = true;
    *violation = lifted_violation;
    return lifted_cut;
  }
  if (std::isnan(plain_violation)) return std::nullopt;
  *lifted = false;
  *violation = plain_violation;
  return plain;
}

}  // namespace

std::vector<Cut> generate_flow_cover_cuts(const Model& model, const Solution& solution,
                                          const std::vector<double>& col_lower,
                                          const std::vector<double>& col_upper,
                                          FlowCoverStats* stats, bool lift) {
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (n == 0 || m == 0) return cuts;
  if (static_cast<Index>(solution.col_value.size()) != n ||
      static_cast<Index>(col_lower.size()) != n || static_cast<Index>(col_upper.size()) != n) {
    return cuts;
  }
  FlowCoverStats local;
  FlowCoverStats& s = stats != nullptr ? *stats : local;
  std::vector<char> is_vub_row(static_cast<std::size_t>(m), 0);
  const std::vector<Vub> vub = find_vubs(model, col_lower, col_upper, &is_vub_row, &s.vub_rows);
  if (s.vub_rows == 0) return cuts;  // no switched flow anywhere: nothing this family is for

  // AGGREGATION (row_aggregation.hpp): a switched flow is never eliminated, its variable
  // upper bound being the structure separated on; the stocks and unswitched flows between
  // rows are what the walk follows.
  std::vector<char> keep(static_cast<std::size_t>(n), 0);
  for (Index j = 0; j < n; ++j) {
    if (vub[static_cast<std::size_t>(j)].indicator >= 0) keep[static_cast<std::size_t>(j)] = 1;
  }
  const std::vector<RowEntries> rows = rows_of(model);
  std::vector<char> used(static_cast<std::size_t>(m), 0);

  for (Index i = 0; i < m; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    if (is_vub_row[ui]) continue;
    const RowEntries& row = rows[ui];
    if (row.columns.size() < 2) continue;
    // Both senses of the row are flow sets: sum a x <= upper, and sum -a x <= -lower.
    for (int side = 0; side < 2; ++side) {
      const double bound = side == 0 ? model.row_upper[ui] : model.row_lower[ui];
      if (!is_finite_bound(bound)) continue;
      const double sign = side == 0 ? 1.0 : -1.0;
      BaseRow base;
      base.columns = row.columns;
      base.values.reserve(row.values.size());
      for (const double v : row.values) base.values.push_back(sign * v);
      base.rhs = sign * bound;
      base.multipliers = {{i, sign}};
      std::fill(used.begin(), used.end(), 0);
      used[ui] = 1;

      std::optional<Cut> best;
      double best_violation = 0.0;
      bool best_lifted = false;
      int best_depth = 0;
      for (int depth = 0; depth <= tol::kRowAggregationDepth; ++depth) {
        const std::optional<FlowSet> set =
            read_flow_set(model, solution, col_lower, col_upper, vub, base);
        if (set) {
          if (depth == 0) ++s.flow_rows;
          // Both covers when the set has an always-on inflow that could be in one (see
          // separate_flow_set); the second counts no cover of its own in the statistics.
          const bool has_always_on =
              std::any_of(set->arcs.begin(), set->arcs.end(), [](const Arc& arc) {
                return arc.inflow && arc.indicator < 0 && std::isfinite(arc.capacity);
              });
          for (int pass = 0; pass < (has_always_on ? 2 : 1); ++pass) {
            bool lifted = false;
            double violation = 0.0;
            int second_covers = 0;
            if (std::optional<Cut> cut =
                    separate_flow_set(model, *set, lift, /*always_on_in_cover=*/pass == 0,
                                      pass == 0 ? &s.covers : &second_covers, &lifted,
                                      &violation)) {
              if (!best || violation > best_violation) {
                best = std::move(cut);
                best_violation = violation;
                best_lifted = lifted;
                best_depth = depth;
              }
            }
          }
        } else if (depth == 0) {
          break;  // not a flow row: nothing to walk from
        }
        if (depth == tol::kRowAggregationDepth) break;
        const Index eliminate =
            column_to_eliminate(model, solution, col_lower, col_upper, base, &keep);
        if (eliminate < 0) break;
        const ColumnView column = model.matrix.column(eliminate);
        bool extended = false;
        for (Index k = 0; k < column.size && !extended; ++k) {
          const Index other = column.rows[k];
          const auto uo = static_cast<std::size_t>(other);
          if (used[uo] || is_vub_row[uo]) continue;
          BaseRow trial = base;
          if (!aggregate_row(model, rows, other, eliminate, &trial)) continue;
          used[uo] = 1;
          base = std::move(trial);
          extended = true;
        }
        if (!extended) break;
      }
      if (!best) continue;
      cuts.push_back(std::move(*best));
      ++s.cuts;
      if (best_lifted) ++s.lifted;
      if (best_depth > 0) ++s.aggregated;
      s.deepest = std::max(s.deepest, best_depth);
    }
  }
  return cuts;
}

}  // namespace sankhya::mip
