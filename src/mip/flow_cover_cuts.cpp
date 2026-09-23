// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover cuts (#419). The references and the validity argument are on the
// declaration.

#include "flow_cover_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

}  // namespace

std::vector<Cut> generate_flow_cover_cuts(const Model& model, const Solution& solution,
                                          const std::vector<double>& col_lower,
                                          const std::vector<double>& col_upper,
                                          FlowCoverStats* stats) {
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

  const CsrView by_row(model.matrix);
  for (Index i = 0; i < m; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    if (is_vub_row[ui]) continue;
    const ColumnView row = by_row.row(i);
    if (row.size < 2) continue;
    // Both senses of the row are flow sets: sum a x <= upper, and sum -a x <= -lower.
    for (int side = 0; side < 2; ++side) {
      const double bound = side == 0 ? model.row_upper[ui] : model.row_lower[ui];
      if (!is_finite_bound(bound)) continue;
      const double sign = side == 0 ? 1.0 : -1.0;
      double b = sign * bound;
      std::vector<Arc> arcs;
      bool readable = true;
      bool any_switched_inflow = false;
      for (Index k = 0; k < row.size && readable; ++k) {
        const Index j = row.rows[k];
        const auto uj = static_cast<std::size_t>(j);
        const double coef = sign * row.values[k];
        if (std::fabs(coef) <= tol::kZeroDrop) continue;
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
          b -= coef * lo;
          arc.capacity = is_finite_bound(hi) ? arc.scale * (hi - lo) : kInf;
          arc.flow = arc.scale * (x - lo);
          arc.inflow = coef > 0.0;
        } else if (is_finite_bound(hi)) {
          // Bounded above only: x = hi - x'', which turns the arc around.
          arc.complemented = true;
          arc.shift = hi;
          b -= coef * hi;
          arc.capacity = kInf;
          arc.flow = arc.scale * (hi - x);
          arc.inflow = coef < 0.0;
        } else {
          readable = false;  // a free column has no flow to speak of
        }
        arcs.push_back(arc);
      }
      if (!readable || !any_switched_inflow || !std::isfinite(b)) continue;
      ++s.flow_rows;

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
      if (lambda <= margin) continue;
      ++s.covers;

      // THE INEQUALITY, in the model's own columns, and its two sides at the LP point.
      Cut cut;
      cut.family = CutFamily::kFlowCover;
      cut.coeff.assign(static_cast<std::size_t>(n), 0.0);
      double rhs = b;
      double lhs_star = 0.0;
      double rhs_star = b;
      for (std::size_t k = 0; k < arcs.size(); ++k) {
        const Arc& arc = arcs[k];
        if (arc.inflow) {
          if (!in_cover[k]) continue;  // outside the cover: zero, the unlifted coefficient
          add_flow(arc, 1.0, &cut, &rhs);
          lhs_star += arc.flow;
          const double extra = std::max(0.0, arc.capacity - lambda);
          if (extra > tol::kZeroDrop && arc.indicator >= 0) {
            // (u_j - lambda)^+ (1 - y_j): the constant to the right, -y_j on the left.
            cut.coeff[static_cast<std::size_t>(arc.indicator)] -= extra;
            rhs -= extra;
            lhs_star += extra * (1.0 - arc.y_star);
          }
        } else if (lambda * arc.y_star < arc.flow) {
          // An outflow on the right as lambda times its switch, the smaller form here.
          if (arc.indicator >= 0) {
            cut.coeff[static_cast<std::size_t>(arc.indicator)] -= lambda;
          } else {
            rhs += lambda;
          }
          rhs_star += lambda * arc.y_star;
        } else {
          // An outflow on the right as the flow itself.
          add_flow(arc, -1.0, &cut, &rhs);
          rhs_star += arc.flow;
        }
      }
      const double violation = (lhs_star - rhs_star) / std::max(1.0, std::fabs(rhs_star));
      if (violation <= kMinViolation) continue;
      double largest = 0.0;
      double smallest = kInf;
      for (const double c : cut.coeff) {
        const double magnitude = std::fabs(c);
        if (magnitude <= tol::kZeroDrop) continue;
        largest = std::max(largest, magnitude);
        smallest = std::min(smallest, magnitude);
      }
      if (largest == 0.0 || largest / smallest > kMaxDynamism) continue;
      cut.rhs = rhs;
      cuts.push_back(std::move(cut));
      ++s.cuts;
    }
  }
  return cuts;
}

}  // namespace sankhya::mip
