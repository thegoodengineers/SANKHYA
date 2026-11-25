// SPDX-License-Identifier: Apache-2.0
// SANKHYA - forcing rows and dominated columns by implied dual bounds: the reductions. The
// derivation and the citations are in dual_reduce.hpp; the replay in dual_reduce_postsolve.cpp.

#include "presolve/dual_reduce.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {

namespace {

using Entry = std::pair<Index, double>;

constexpr int kMaxPasses = 8;
/// Rounds of dual bound propagation through the free columns per pass.
constexpr int kDualPropagationRounds = 4;

bool finite(double v) {
  return is_finite_bound(v);
}

struct Work {
  std::vector<std::vector<Entry>> rows;  ///< live entries by row
  std::vector<std::vector<Entry>> cols;  ///< the same entries by column
  std::vector<Index> col_count;          ///< live entries per column
  std::vector<double> col_lower, col_upper, cost, row_lower, row_upper;
  std::vector<char> row_dead, col_dead;
  double offset = 0.0;
  double sense = 1.0;  ///< costs are held in the minimisation sense: sense * col_cost

  /// Fix column j at v: its contribution becomes a constant in every live row and the
  /// objective, and the column goes.
  void fix(Index j, double v) {
    const auto u = static_cast<std::size_t>(j);
    for (const auto& [i, a] : cols[u]) {
      const auto ui = static_cast<std::size_t>(i);
      if (row_dead[ui] != 0) continue;
      if (finite(row_lower[ui])) row_lower[ui] -= a * v;
      if (finite(row_upper[ui])) row_upper[ui] -= a * v;
      auto& row = rows[ui];
      for (std::size_t p = 0; p < row.size(); ++p) {
        if (row[p].first == j) {
          row.erase(row.begin() + static_cast<std::ptrdiff_t>(p));
          break;
        }
      }
    }
    offset += sense * cost[u] * v;
    col_lower[u] = col_upper[u] = v;
    col_dead[u] = 1;
    col_count[u] = 0;
  }

  void kill_row(Index i) {
    const auto ui = static_cast<std::size_t>(i);
    for (const auto& [j, a] : rows[ui]) --col_count[static_cast<std::size_t>(j)];
    rows[ui].clear();
    row_dead[ui] = 1;
  }
};

/// [inf, sup] of row i's activity over the live columns' boxes.
struct Activity {
  double inf = 0.0, sup = 0.0;
  bool inf_finite = true, sup_finite = true;
};

Activity activity_of(const Work& w, Index i) {
  Activity act;
  for (const auto& [j, a] : w.rows[static_cast<std::size_t>(i)]) {
    const auto u = static_cast<std::size_t>(j);
    const double at_inf = a > 0.0 ? w.col_lower[u] : w.col_upper[u];
    const double at_sup = a > 0.0 ? w.col_upper[u] : w.col_lower[u];
    if (finite(at_inf))
      act.inf += a * at_inf;
    else
      act.inf_finite = false;
    if (finite(at_sup))
      act.sup += a * at_sup;
    else
      act.sup_finite = false;
  }
  return act;
}

/// Whether fixing every column of row i at the bound that attains its activity bound moves
/// no column by more than the feasibility tolerance: `gap`, the distance between the
/// activity bound and the row bound, taken in each column's own units, gap / |a_rj|, and
/// held to the tolerance a column bound is. The gap alone against an absolute tolerance
/// reads a row whose bound is itself below that tolerance as forcing: R391 of
/// scaled_ship04l is sum a_j x_j = 8.8e-8 with x >= 0 and |a_j| down to 5.8e-11, a demand
/// its columns meet far from zero, and with them fixed at zero the answer came back
/// 2.7e-4 under the optimum (#1213's stress run).
bool gap_fixes_columns(const Work& w, Index i, double gap, bool at_sup, double feasibility) {
  for (const auto& [j, a] : w.rows[static_cast<std::size_t>(i)]) {
    const auto u = static_cast<std::size_t>(j);
    const double bound = (a > 0.0) == at_sup ? w.col_upper[u] : w.col_lower[u];
    if (std::fabs(gap) > feasibility * std::fabs(a) * std::max(1.0, std::fabs(bound))) {
      return false;
    }
  }
  return true;
}

/// What one column of row i moving by the feasibility tolerance changes the activity by, at
/// most: the largest |a_rj| * max(1, |bound|) over the row. A row is redundant when the box
/// keeps it inside its bounds to within the tolerance, and that tolerance is the row's own
/// only as far as this scale: 1e-8 x0 + 1e-8 x1 >= 2e-8 with x in [0, 1] is short by 2e-8
/// at x = 0, inside an absolute 1e-7, and is x0 + x1 >= 2.
double row_scale(const Work& w, Index i) {
  double scale = 0.0;
  for (const auto& [j, a] : w.rows[static_cast<std::size_t>(i)]) {
    const auto u = static_cast<std::size_t>(j);
    double size = 1.0;
    if (finite(w.col_lower[u])) size = std::max(size, std::fabs(w.col_lower[u]));
    if (finite(w.col_upper[u])) size = std::max(size, std::fabs(w.col_upper[u]));
    scale = std::max(scale, std::fabs(a) * size);
  }
  return scale;
}

/// Forcing and redundant rows over the live rows; true when something fired.
bool reduce_rows(Work* w, double feasibility, DualReduceResult* out) {
  bool changed = false;
  const auto m = static_cast<Index>(w->rows.size());
  for (Index i = 0; i < m; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    if (w->row_dead[ui] != 0 || w->rows[ui].empty()) continue;
    const Activity act = activity_of(*w, i);
    const double lo = w->row_lower[ui];
    const double hi = w->row_upper[ui];
    const double tol_lo = feasibility * std::max(1.0, std::fabs(lo));
    const double tol_hi = feasibility * std::max(1.0, std::fabs(hi));
    // Redundant: the box alone keeps the row inside its bounds, to within the tolerance at
    // the row's own scale (row_scale). Dual zero.
    const double slack = feasibility * row_scale(*w, i);
    const bool lower_slack =
        !finite(lo) || (act.inf_finite && act.inf >= lo - std::min(tol_lo, slack));
    const bool upper_slack =
        !finite(hi) || (act.sup_finite && act.sup <= hi + std::min(tol_hi, slack));
    if (lower_slack && upper_slack) {
      DualReduction rec;
      rec.kind = DualReduction::Kind::kRedundantRow;
      rec.row = i;
      out->records.push_back(std::move(rec));
      ++out->redundant_rows;
      w->kill_row(i);
      changed = true;
      continue;
    }
    // Forcing: the activity bound meets the row bound, so every column sits at the bound
    // that attains it - to within the tolerance, on the row and on each column it fixes
    // (gap_fixes_columns). A bound crossed by more than the tolerance is an infeasibility
    // the engine reports; it is left alone here.
    bool force_lower = finite(lo) && act.sup_finite && std::fabs(act.sup - lo) <= tol_lo &&
                       gap_fixes_columns(*w, i, act.sup - lo, true, feasibility);
    bool force_upper = finite(hi) && act.inf_finite && std::fabs(act.inf - hi) <= tol_hi &&
                       gap_fixes_columns(*w, i, act.inf - hi, false, feasibility);
    if (!force_lower && !force_upper) continue;
    if (force_lower && force_upper) force_upper = false;  // either reading fixes the same
    DualReduction rec;
    rec.kind = DualReduction::Kind::kForcingRow;
    rec.row = i;
    rec.at_lower = force_lower;
    rec.equality = finite(lo) && finite(hi) && lo == hi;
    for (const auto& [j, a] : w->rows[ui]) {
      const auto u = static_cast<std::size_t>(j);
      const bool at_upper = force_lower ? a > 0.0 : a < 0.0;
      rec.columns.push_back(j);
      rec.coefficients.push_back(a);
      rec.values.push_back(at_upper ? w->col_upper[u] : w->col_lower[u]);
    }
    w->kill_row(i);
    for (std::size_t p = 0; p < rec.columns.size(); ++p) w->fix(rec.columns[p], rec.values[p]);
    out->records.push_back(std::move(rec));
    ++out->forcing_rows;
    changed = true;
  }
  return changed;
}

/// What the row types and the column singletons say of each row's dual, before the free
/// columns propagate it, and the same with the tightest singleton left out: the box a
/// singleton is tested against is the one that does not rest on that singleton
/// (reduce_columns).
struct SingletonBounds {
  std::vector<double> lo, hi;            ///< row type and every singleton
  std::vector<double> lo_else, hi_else;  ///< without the singleton that set lo (hi)
  std::vector<Index> lo_from, hi_from;   ///< that singleton, or -1
};

/// The box [y_lo, y_hi] every dual feasible y lies in (dual_reduce.hpp), from the row
/// types, the column singletons and the free columns.
void dual_box(const Work& w, std::vector<double>* y_lo, std::vector<double>* y_hi,
              SingletonBounds* singletons) {
  const auto m = static_cast<Index>(w.rows.size());
  const auto n = static_cast<Index>(w.cols.size());
  y_lo->assign(static_cast<std::size_t>(m), -kInfinity);
  y_hi->assign(static_cast<std::size_t>(m), kInfinity);
  for (Index i = 0; i < m; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    if (w.row_dead[ui] != 0) continue;
    const bool has_lo = finite(w.row_lower[ui]);
    const bool has_hi = finite(w.row_upper[ui]);
    if (has_lo && !has_hi) (*y_lo)[ui] = 0.0;
    if (has_hi && !has_lo) (*y_hi)[ui] = 0.0;
  }
  // A live column j with u_j = +inf says sum a_ij y_i <= c_j, with l_j = -inf >= c_j. The
  // singletons bound their one row outright; the free columns are run as the equality
  // sum a_ij y_i = c_j, each term bounded from the others', for a few rounds.
  singletons->lo_else = *y_lo;
  singletons->hi_else = *y_hi;
  singletons->lo_from.assign(static_cast<std::size_t>(m), -1);
  singletons->hi_from.assign(static_cast<std::size_t>(m), -1);
  // The tightest bound on each side, the column it came from, and what the side was
  // without that column: the previous tightest, row type included.
  const auto tighten = [&](Index i, double bound, bool upper, Index from) {
    const auto ui = static_cast<std::size_t>(i);
    if (upper) {
      if (bound < (*y_hi)[ui]) {
        singletons->hi_else[ui] = (*y_hi)[ui];
        (*y_hi)[ui] = bound;
        singletons->hi_from[ui] = from;
      } else {
        singletons->hi_else[ui] = std::min(singletons->hi_else[ui], bound);
      }
    } else if (bound > (*y_lo)[ui]) {
      singletons->lo_else[ui] = (*y_lo)[ui];
      (*y_lo)[ui] = bound;
      singletons->lo_from[ui] = from;
    } else {
      singletons->lo_else[ui] = std::max(singletons->lo_else[ui], bound);
    }
  };
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (w.col_dead[u] != 0 || w.col_count[u] != 1) continue;
    const bool no_upper = !finite(w.col_upper[u]);
    const bool no_lower = !finite(w.col_lower[u]);
    if (!no_upper && !no_lower) continue;
    for (const auto& [i, a] : w.cols[u]) {
      if (w.row_dead[static_cast<std::size_t>(i)] != 0) continue;
      const double bound = w.cost[u] / a;
      // a y <= c: y <= c/a when a > 0, y >= c/a when a < 0; a y >= c the other way round.
      if (no_upper) tighten(i, bound, a > 0.0, j);
      if (no_lower) tighten(i, bound, a < 0.0, j);
    }
  }
  singletons->lo = *y_lo;
  singletons->hi = *y_hi;
  for (int round = 0; round < kDualPropagationRounds; ++round) {
    bool moved = false;
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (w.col_dead[u] != 0 || w.col_count[u] < 2) continue;
      if (finite(w.col_lower[u]) || finite(w.col_upper[u])) continue;
      // sum a_i y_i = c: a_k y_k = c - rest, rest in [rest_lo, rest_hi].
      double rest_lo = 0.0, rest_hi = 0.0;
      int lo_inf = 0, hi_inf = 0;
      for (const auto& [i, a] : w.cols[u]) {
        const auto ui = static_cast<std::size_t>(i);
        if (w.row_dead[ui] != 0) continue;
        const double at_lo = a > 0.0 ? (*y_lo)[ui] : (*y_hi)[ui];
        const double at_hi = a > 0.0 ? (*y_hi)[ui] : (*y_lo)[ui];
        if (finite(at_lo))
          rest_lo += a * at_lo;
        else
          ++lo_inf;
        if (finite(at_hi))
          rest_hi += a * at_hi;
        else
          ++hi_inf;
      }
      if (lo_inf > 1 && hi_inf > 1) continue;
      for (const auto& [i, a] : w.cols[u]) {
        const auto ui = static_cast<std::size_t>(i);
        if (w.row_dead[ui] != 0) continue;
        const double at_lo = a > 0.0 ? (*y_lo)[ui] : (*y_hi)[ui];
        const double at_hi = a > 0.0 ? (*y_hi)[ui] : (*y_lo)[ui];
        // The others' range: this term taken out of the sums.
        const bool others_lo_finite = lo_inf == (finite(at_lo) ? 0 : 1);
        const bool others_hi_finite = hi_inf == (finite(at_hi) ? 0 : 1);
        const double others_lo = rest_lo - (finite(at_lo) ? a * at_lo : 0.0);
        const double others_hi = rest_hi - (finite(at_hi) ? a * at_hi : 0.0);
        // a y_k in [c - others_hi, c - others_lo].
        const double term_lo = others_hi_finite ? w.cost[u] - others_hi : -kInfinity;
        const double term_hi = others_lo_finite ? w.cost[u] - others_lo : kInfinity;
        const double new_lo = a > 0.0 ? (finite(term_lo) ? term_lo / a : -kInfinity)
                                      : (finite(term_hi) ? term_hi / a : -kInfinity);
        const double new_hi = a > 0.0 ? (finite(term_hi) ? term_hi / a : kInfinity)
                                      : (finite(term_lo) ? term_lo / a : kInfinity);
        if (new_lo >
            (*y_lo)[ui] + tol::kPresolveBoundMinStep * std::max(1.0, std::fabs(new_lo))) {
          (*y_lo)[ui] = new_lo;
          moved = true;
        }
        if (new_hi <
            (*y_hi)[ui] - tol::kPresolveBoundMinStep * std::max(1.0, std::fabs(new_hi))) {
          (*y_hi)[ui] = new_hi;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }
}

/// Dominated columns over the live columns; true when one was fixed.
bool reduce_columns(Work* w, double dual_tolerance, DualReduceResult* out) {
  std::vector<double> y_lo, y_hi;
  SingletonBounds singletons;
  dual_box(*w, &y_lo, &y_hi, &singletons);
  // AN EMPTY BOX IS NOT A BOX (#1337). y_lo > y_hi on a row says no y is dual feasible: the
  // LP has no optimum, it is unbounded or infeasible, and which of the two is the engine's
  // to find on the model as it stands (nothing here has shown a feasible point). The ranges
  // below are read off the intervals, and off an inverted one they are not ranges: with
  // y <= 0 from the row and y >= 1/4 from a singleton with no lower bound, a column of the
  // row came out "negative over the box" and was fixed at its upper bound, the one bound
  // that kept an unbounded LP bounded, and the engine then called that optimal. The bounds
  // only ever tighten, so an interval inverted anywhere in the propagation still is here.
  for (std::size_t i = 0; i < y_lo.size(); ++i) {
    if (y_lo[i] > y_hi[i]) return false;
  }
  bool changed = false;
  const auto n = static_cast<Index>(w->cols.size());
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (w->col_dead[u] != 0 || w->col_count[u] == 0) continue;
    // d_j = c_j - sum a_ij y_i over the box: [d_lo, d_hi].
    //
    // A SINGLETON WITH A BOUND MISSING IS TESTED WITHOUT ITS OWN WORD (#1337). Fixing x_j
    // removes its dual constraint, so the box the fixing rests on must hold without that
    // constraint. For every other column it does: the box is built from the singletons and
    // the free columns, which stay. Such a singleton is one of the box's sources, and the
    // free columns carry its bound round and back onto its own row tighter than it went in:
    // with y0 <= -24 from x2 >= -2 (cost -6, coefficient 1/4), and free columns saying
    // y0 = 16 y1 and y0 = 2 y1, the rounds gave y0 <= -192, x2 came out "positive over the
    // box" and was fixed at -2. Without x2's constraint y = 0 is dual feasible: the LP with
    // x2 fixed is bounded, the LP as given is not, and the engine called the first optimal.
    // So such a column is read against its row's bound from the row type and the OTHER
    // singletons only, with no propagation. It is then fixed only when another source bounds
    // its row strictly tighter than it does, so no fixing ever loosens the box.
    const bool source =
        w->col_count[u] == 1 && (!finite(w->col_lower[u]) || !finite(w->col_upper[u]));
    double d_lo = w->cost[u], d_hi = w->cost[u];
    bool lo_finite = true, hi_finite = true;
    for (const auto& [i, a] : w->cols[u]) {
      const auto ui = static_cast<std::size_t>(i);
      if (w->row_dead[ui] != 0) continue;
      double lo = y_lo[ui];
      double hi = y_hi[ui];
      if (source) {
        lo = singletons.lo_from[ui] == j ? singletons.lo_else[ui] : singletons.lo[ui];
        hi = singletons.hi_from[ui] == j ? singletons.hi_else[ui] : singletons.hi[ui];
      }
      const double term_max = a > 0.0 ? hi : lo;  // of a y_i
      const double term_min = a > 0.0 ? lo : hi;
      if (finite(term_max))
        d_lo -= a * term_max;
      else
        lo_finite = false;
      if (finite(term_min))
        d_hi -= a * term_min;
      else
        hi_finite = false;
    }
    const double tol = dual_tolerance * std::max(1.0, std::fabs(w->cost[u]));
    double value = 0.0;
    if (lo_finite && d_lo > tol && finite(w->col_lower[u])) {
      value = w->col_lower[u];
    } else if (hi_finite && d_hi < -tol && finite(w->col_upper[u])) {
      value = w->col_upper[u];
    } else {
      continue;
    }
    DualReduction rec;
    rec.kind = DualReduction::Kind::kDominatedColumn;
    rec.column = j;
    rec.value = value;
    out->records.push_back(std::move(rec));
    ++out->dominated_columns;
    w->fix(j, value);
    changed = true;
  }
  return changed;
}

}  // namespace

DualReduceResult dual_reduce(const Model& model, const Options& options, Logger& logger) {
  DualReduceResult out;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (model.has_integrality() || model.has_quadratic_objective() ||
      model.has_semicontinuous_or_sos() || n == 0 || m == 0) {
    return out;
  }
  const double feasibility = options.get_double("primal_feasibility_tolerance");
  const double dual_tolerance = options.get_double("dual_feasibility_tolerance");

  Work w;
  w.rows.resize(static_cast<std::size_t>(m));
  w.cols.resize(static_cast<std::size_t>(n));
  w.col_count.assign(static_cast<std::size_t>(n), 0);
  w.col_lower = model.col_lower;
  w.col_upper = model.col_upper;
  w.sense = model.sense_multiplier();
  w.cost = model.col_cost;
  for (double& c : w.cost) c *= w.sense;
  w.row_lower = model.row_lower;
  w.row_upper = model.row_upper;
  w.row_dead.assign(static_cast<std::size_t>(m), 0);
  w.col_dead.assign(static_cast<std::size_t>(n), 0);
  w.offset = model.objective_offset;
  for (Index j = 0; j < n; ++j) {
    const ColumnView view = model.matrix.column(j);
    for (Index p = 0; p < view.size; ++p) {
      if (view.values[p] == 0.0) continue;
      w.rows[static_cast<std::size_t>(view.rows[p])].emplace_back(j, view.values[p]);
      w.cols[static_cast<std::size_t>(j)].emplace_back(view.rows[p], view.values[p]);
      ++w.col_count[static_cast<std::size_t>(j)];
    }
  }
  for (int pass = 0; pass < kMaxPasses; ++pass) {
    const bool rows_moved = reduce_rows(&w, feasibility, &out);
    const bool cols_moved = reduce_columns(&w, dual_tolerance, &out);
    if (!rows_moved && !cols_moved) break;
  }
  if (!out.reduced_anything()) return out;

  Model& reduced = out.model;
  reduced.name = model.name;
  reduced.source_path = model.source_path;
  reduced.sense = model.sense;
  reduced.objective_offset = w.offset;
  std::vector<Index> col_new(static_cast<std::size_t>(n), -1);
  for (Index j = 0; j < n; ++j) {
    if (w.col_dead[static_cast<std::size_t>(j)] != 0) continue;
    col_new[static_cast<std::size_t>(j)] = static_cast<Index>(out.col_to_input.size());
    out.col_to_input.push_back(j);
  }
  for (Index i = 0; i < m; ++i) {
    if (w.row_dead[static_cast<std::size_t>(i)] != 0) continue;
    out.row_to_input.push_back(i);
  }
  const auto nn = static_cast<Index>(out.col_to_input.size());
  const auto mm = static_cast<Index>(out.row_to_input.size());
  reduced.resize_columns(nn);
  reduced.resize_rows(mm);
  const bool col_names = model.col_names.size() == static_cast<std::size_t>(n);
  const bool row_names = model.row_names.size() == static_cast<std::size_t>(m);
  if (col_names) reduced.col_names.resize(static_cast<std::size_t>(nn));
  if (row_names) reduced.row_names.resize(static_cast<std::size_t>(mm));
  for (Index j = 0; j < nn; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const auto from = static_cast<std::size_t>(out.col_to_input[u]);
    reduced.col_cost[u] = w.sense * w.cost[from];
    reduced.col_lower[u] = w.col_lower[from];
    reduced.col_upper[u] = w.col_upper[from];
    reduced.col_type[u] = VarType::kContinuous;
    if (col_names) reduced.col_names[u] = model.col_names[from];
  }
  reduced.matrix.reset(mm, nn);
  for (Index i = 0; i < mm; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const auto from = static_cast<std::size_t>(out.row_to_input[u]);
    reduced.row_lower[u] = w.row_lower[from];
    reduced.row_upper[u] = w.row_upper[from];
    if (row_names) reduced.row_names[u] = model.row_names[from];
    for (const auto& [k, a] : w.rows[from]) {
      reduced.matrix.add_entry(i, col_new[static_cast<std::size_t>(k)], a);
    }
  }
  reduced.matrix.finalize(0.0);
  logger.info(
      "Dual reductions: {} rows -> {}, {} columns -> {} ({} forcing rows, {} redundant rows, "
      "{} dominated columns)",
      m, mm, n, nn, out.forcing_rows, out.redundant_rows, out.dominated_columns);
  return out;
}

}  // namespace sankhya::presolve
