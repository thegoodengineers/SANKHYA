// SPDX-License-Identifier: Apache-2.0
// SANKHYA - flow cover cuts (#419). See flow_cover_cuts.hpp for the inequality, the citation
// and the scope this file deliberately stops at.

#include "flow_cover_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

/// The binary and capacity of one column's variable upper bound x_j <= u_j * y_j.
struct VariableUpperBound {
  Index binary_col = -1;
  double capacity = 0.0;  ///< u_j > 0
};

/// Row bound must be zero (relative to its own scale) for a row to be a clean x_j <= u_j y_j;
/// anything else is a shifted variable bound, a different structure this family does not
/// attempt (the issue's textbook shape has no such shift).
constexpr double kVubRhsTolerance = 1e-9;
/// Violation, relative to the cut's right-hand side, below which a flow cover is not worth a
/// row (the same convention mir_cuts.cpp uses).
constexpr double kMinViolation = 1e-4;
/// A cover whose excess is not clearly positive is numerically meaningless.
constexpr double kMinExcess = 1e-7;
/// Largest ratio between the largest and smallest coefficient a cut may carry, matched to
/// mir_cuts.cpp's own filter so a bad cut is not even proposed.
constexpr double kMaxDynamism = 1e6;

[[nodiscard]] bool is_binary(const Model& model, const std::vector<double>& col_lower,
                             const std::vector<double>& col_upper, Index col) {
  const auto j = static_cast<std::size_t>(col);
  return model.col_type[j] == VarType::kInteger && std::fabs(col_lower[j]) <= tol::kZeroDrop &&
         std::fabs(col_upper[j] - 1.0) <= tol::kZeroDrop;
}

/// One row of the model in row-wise form: (column, coefficient) pairs.
struct RowEntries {
  std::vector<Index> columns;
  std::vector<double> values;
};

std::vector<RowEntries> rows_of(const Model& model) {
  std::vector<RowEntries> rows(static_cast<std::size_t>(model.num_rows()));
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      RowEntries& row = rows[static_cast<std::size_t>(column.rows[k])];
      row.columns.push_back(j);
      row.values.push_back(column.values[k]);
    }
  }
  return rows;
}

/// For every continuous column, the tightest variable-upper-bound row that gives it a linked
/// binary: a row with exactly two nonzeros, one a positive coefficient on the continuous
/// column, the other a negative coefficient on a binary column, and a right-hand side of zero
/// (either `<= 0` with no lower bound, or `>= 0` with no upper bound, since both state the
/// same halfspace once oriented). When more than one such row exists for a column, the
/// smallest capacity (the tightest bound) is kept - it is still an exact row of the model, so
/// using it is always valid.
std::vector<std::optional<VariableUpperBound>> find_variable_upper_bounds(
    const Model& model, const std::vector<RowEntries>& rows,
    const std::vector<double>& col_lower, const std::vector<double>& col_upper) {
  std::vector<std::optional<VariableUpperBound>> found(
      static_cast<std::size_t>(model.num_cols()));
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto ui = static_cast<std::size_t>(i);
    const RowEntries& row = rows[ui];
    if (row.columns.size() != 2) continue;
    for (int orientation = 0; orientation < 2; ++orientation) {
      const double bound = orientation == 0 ? model.row_upper[ui] : model.row_lower[ui];
      const double other_bound = orientation == 0 ? model.row_lower[ui] : model.row_upper[ui];
      if (is_finite_bound(other_bound)) continue;  // must be one-sided
      if (std::fabs(bound) > kVubRhsTolerance) continue;
      const double sign = orientation == 0 ? 1.0 : -1.0;
      for (int slot = 0; slot < 2; ++slot) {
        const Index col_a = row.columns[static_cast<std::size_t>(slot)];
        const Index col_b = row.columns[static_cast<std::size_t>(1 - slot)];
        const double coef_a = sign * row.values[static_cast<std::size_t>(slot)];
        const double coef_b = sign * row.values[static_cast<std::size_t>(1 - slot)];
        const auto ua = static_cast<std::size_t>(col_a);
        if (model.col_type[ua] == VarType::kInteger) continue;  // col_a must be the flow var
        // The theorem is stated for x_j >= 0 with no other bound; a nonzero lower bound
        // would need the same bound-substitution mir_cuts.cpp does, which this family does
        // not attempt (flow_cover_cuts.hpp's scope note).
        if (std::fabs(col_lower[ua]) > tol::kZeroDrop) continue;
        if (coef_a <= tol::kZeroDrop) continue;
        if (coef_b >= -tol::kZeroDrop)
          continue;  // must be a negative coefficient on the binary
        if (!is_binary(model, col_lower, col_upper, col_b)) continue;
        const double capacity = -coef_b / coef_a;
        if (!(capacity > 0.0) || !std::isfinite(capacity)) continue;
        auto& slot_found = found[ua];
        if (!slot_found.has_value() || capacity < slot_found->capacity) {
          slot_found = VariableUpperBound{col_b, capacity};
        }
      }
    }
  }
  return found;
}

/// One flow column of a capacity row, in the row's own orientation (coefficient already
/// sign-flipped so it is positive, and scaled so U = coefficient * capacity is the effective
/// capacity of z_j = coefficient * x_j).
struct FlowTerm {
  Index flow_col = -1;
  Index binary_col = -1;
  double coefficient = 0.0;         ///< a_j > 0 in the oriented row
  double capacity = 0.0;            ///< u_j, the column's own variable upper bound
  double effective_capacity = 0.0;  ///< U_j = a_j * u_j
  double z_star = 0.0;              ///< a_j * x*_j, the scaled LP value
  double y_star = 0.0;
};

/// How many rows a single-node flow relaxation may aggregate into a starting row before the
/// search gives up on it (flow_cover_cuts.hpp: "small", deliberately less than mir_cuts.cpp's
/// 6, which discharges continuous slack rather than reaching a flow shape at all).
constexpr int kMaxAggregation = 3;

/// The qualifying flow terms of a dense row `coeff` (a valid "sum coeff[j] x_j <= rhs"
/// inequality) and the first column, in index order, that keeps it from being a clean
/// single-node flow row - a positive coefficient on a continuous column with its own
/// variable upper bound is qualifying, anything else nonzero is not. Deterministic: column
/// order is fixed, so which column is picked to eliminate next never depends on iteration
/// order elsewhere.
struct RowScan {
  std::vector<FlowTerm> terms;
  Index disqualifying = -1;
};

RowScan scan_flow_terms(const Model& model, const Solution& solution,
                        const std::vector<std::optional<VariableUpperBound>>& vubs,
                        const std::vector<double>& coeff) {
  RowScan scan;
  for (Index j = 0; j < static_cast<Index>(coeff.size()); ++j) {
    const auto uj = static_cast<std::size_t>(j);
    const double c = coeff[uj];
    if (std::fabs(c) <= tol::kZeroDrop) continue;
    if (c > 0.0 && model.col_type[uj] != VarType::kInteger && vubs[uj].has_value()) {
      FlowTerm term;
      term.flow_col = j;
      term.binary_col = vubs[uj]->binary_col;
      term.coefficient = c;
      term.capacity = vubs[uj]->capacity;
      term.effective_capacity = c * vubs[uj]->capacity;
      term.z_star = c * solution.col_value[uj];
      term.y_star = solution.col_value[static_cast<std::size_t>(term.binary_col)];
      scan.terms.push_back(term);
    } else if (scan.disqualifying < 0) {
      scan.disqualifying = j;
    }
  }
  return scan;
}

/// Eliminate column `eliminate` from the dense inequality `coeff <= *rhs` by adding lambda
/// times an unused row that contains it (as `sum a x <= u` for lambda > 0, `sum a x >= l` for
/// lambda < 0, both of which are `<=` inequalities after the multiplication - the same
/// direction argument mir_cuts.cpp's aggregate_row uses), chosen so the column cancels
/// exactly. Rows are tried in the column's own storage order (ascending row index, since the
/// matrix is built that way), so the choice is deterministic. Returns false when no unused row
/// with a finite bound on the needed side contains the column - the aggregation for this line
/// stops there rather than guessing.
bool aggregate_eliminating(const Model& model, const std::vector<RowEntries>& rows,
                           Index eliminate, std::vector<char>* used, std::vector<double>* coeff,
                           double* rhs) {
  const auto ue = static_cast<std::size_t>(eliminate);
  const double coef_in_base = (*coeff)[ue];
  if (std::fabs(coef_in_base) <= tol::kZeroDrop) return false;
  const ColumnView column = model.matrix.column(eliminate);
  for (Index k = 0; k < column.size; ++k) {
    const Index r = column.rows[k];
    const auto ur = static_cast<std::size_t>(r);
    if ((*used)[ur]) continue;
    const double coef_in_row = column.values[k];
    if (std::fabs(coef_in_row) <= tol::kZeroDrop) continue;
    const double lambda = -coef_in_base / coef_in_row;
    const double bound = lambda > 0.0 ? model.row_upper[ur] : model.row_lower[ur];
    if (!is_finite_bound(bound)) continue;
    const RowEntries& row = rows[ur];
    for (std::size_t t = 0; t < row.columns.size(); ++t) {
      (*coeff)[static_cast<std::size_t>(row.columns[t])] += lambda * row.values[t];
    }
    *rhs += lambda * bound;
    // The eliminated column is exactly cancelled in exact arithmetic; make it so here, the
    // same tidy-up mir_cuts.cpp's aggregate_row does.
    (*coeff)[ue] = 0.0;
    (*used)[ur] = true;
    return true;
  }
  return false;
}

/// The best flow cover found for one ordering of `terms`: try every prefix that has positive
/// excess and keep whichever is most violated at the LP point. Any such cover yields a valid
/// inequality (see flow_cover_cuts.hpp), so trying several orderings only affects whether a
/// violated one is found, never correctness.
struct CoverResult {
  std::vector<std::size_t> members;  // indices into `terms`
  double excess = 0.0;
  double violation = 0.0;
};

/// The finished cut for one chosen cover: the cover's members with the flow cover coefficients,
/// plus the partial lifting of leftover terms whose own variable upper bound is already tight
/// at the LP point (flow_cover_cuts.hpp). Returns nullopt only on a non-finite result.
struct BuiltCut {
  Cut cut;
  int lifted_terms = 0;
};

std::optional<BuiltCut> build_cut(Index n, const std::vector<FlowTerm>& terms,
                                  const CoverResult& cover, double rhs) {
  std::vector<char> in_cover(terms.size(), 0);
  for (const std::size_t member : cover.members) in_cover[member] = 1;

  Cut cut;
  cut.coeff.assign(static_cast<std::size_t>(n), 0.0);
  double cut_rhs = rhs;
  for (std::size_t k = 0; k < terms.size(); ++k) {
    if (!in_cover[k]) continue;
    const FlowTerm& t = terms[k];
    cut.coeff[static_cast<std::size_t>(t.flow_col)] += t.coefficient;
    const double slack_capacity = t.effective_capacity - cover.excess;
    if (slack_capacity > tol::kZeroDrop) {
      cut.coeff[static_cast<std::size_t>(t.binary_col)] -= slack_capacity;
      cut_rhs -= slack_capacity;
    }
  }
  // No row-sense "undo" is needed here (unlike mir_cuts.cpp's y = hi - x complementing):
  // z_j := sign * a_j * x_j (folded into `coeff` before this point, including any single-node
  // aggregation) was a direct substitution, not a flip of the inequality's direction, so
  // t.coefficient is already the correct coefficient on the ORIGINAL x_j in the final cut.

  // LIFTING (partial, see flow_cover_cuts.hpp): a leftover flow column's own variable upper
  // bound (coefficient * x_j - effective_capacity * y_j <= 0) is a valid inequality in its
  // own right; adding any non-negative multiple to a valid inequality keeps it valid. It is
  // folded in exactly when doing so costs nothing at the current point - the bound is already
  // tight there - so the round's separation never loses violation by carrying the extra
  // columns.
  int lifted_terms = 0;
  for (std::size_t k = 0; k < terms.size(); ++k) {
    if (in_cover[k]) continue;
    const FlowTerm& t = terms[k];
    const double vub_slack = t.effective_capacity * t.y_star - t.z_star;
    if (vub_slack > tol::kZeroDrop) continue;  // would cost violation; leave it out
    cut.coeff[static_cast<std::size_t>(t.flow_col)] += t.coefficient;
    cut.coeff[static_cast<std::size_t>(t.binary_col)] -= t.effective_capacity;
    ++lifted_terms;
  }

  if (!std::isfinite(cut_rhs)) return std::nullopt;
  for (const double c : cut.coeff) {
    if (!std::isfinite(c)) return std::nullopt;
  }
  cut.rhs = cut_rhs;
  return BuiltCut{std::move(cut), lifted_terms};
}

std::optional<CoverResult> best_cover_in_order(const std::vector<FlowTerm>& terms,
                                               const std::vector<std::size_t>& order,
                                               double b) {
  std::optional<CoverResult> best;
  double running_capacity = 0.0;
  double running_z = 0.0;
  std::vector<std::size_t> members;
  for (const std::size_t idx : order) {
    members.push_back(idx);
    running_capacity += terms[idx].effective_capacity;
    running_z += terms[idx].z_star;
    const double excess = running_capacity - b;
    if (excess <= kMinExcess) continue;  // not yet a cover
    // sum_{C} z*_j + sum_{C+} (U_j - excess) (1 - y*_j), evaluated for THIS prefix as C.
    double lhs = running_z;
    double max_abs = 1.0;  // the coefficient 1 on every z_j in C
    double min_abs = 1.0;
    for (const std::size_t member : members) {
      const FlowTerm& term = terms[member];
      const double slack_capacity = term.effective_capacity - excess;
      if (slack_capacity > tol::kZeroDrop) {
        lhs += slack_capacity * (1.0 - term.y_star);
        max_abs = std::max(max_abs, slack_capacity);
        min_abs = std::min(min_abs, slack_capacity);
      }
    }
    if (max_abs / min_abs > kMaxDynamism) continue;
    const double violation = (lhs - b) / std::max(1.0, std::fabs(b));
    if (violation <= kMinViolation) continue;
    if (!best || violation > best->violation) {
      best = CoverResult{members, excess, violation};
    }
  }
  return best;
}

}  // namespace

std::vector<Cut> generate_flow_cover_cuts(const Model& model, const Solution& solution,
                                          const std::vector<double>& col_lower,
                                          const std::vector<double>& col_upper,
                                          FlowCoverStats* stats) {
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  if (static_cast<Index>(solution.col_value.size()) != n || n == 0) return cuts;
  if (static_cast<Index>(col_lower.size()) != n || static_cast<Index>(col_upper.size()) != n) {
    return cuts;
  }
  const std::vector<RowEntries> rows = rows_of(model);
  const std::vector<std::optional<VariableUpperBound>> vubs =
      find_variable_upper_bounds(model, rows, col_lower, col_upper);
  if (stats != nullptr) {
    for (const auto& v : vubs) {
      if (v.has_value()) ++stats->vub_rows_found;
    }
  }

  std::vector<char> used(static_cast<std::size_t>(model.num_rows()), 0);
  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto ui = static_cast<std::size_t>(i);
    const RowEntries& row = rows[ui];
    if (row.columns.size() < 2) continue;

    for (int side = 0; side < 2; ++side) {
      const double bound = side == 0 ? model.row_upper[ui] : model.row_lower[ui];
      if (!is_finite_bound(bound)) continue;
      const double sign = side == 0 ? 1.0 : -1.0;

      // The starting inequality, in dense form: sum coeff[j] x_j <= rhs. A SINGLE-NODE FLOW
      // RELAXATION (flow_cover_cuts.hpp) may extend it by aggregating in other rows, up to
      // kMaxAggregation deep, whenever exactly one column blocks the flow-cover shape and an
      // unused row can be found to cancel it. The best cut over every depth tried is kept.
      std::vector<double> coeff(static_cast<std::size_t>(n), 0.0);
      for (std::size_t k = 0; k < row.columns.size(); ++k) {
        coeff[static_cast<std::size_t>(row.columns[k])] = sign * row.values[k];
      }
      double rhs = sign * bound;
      std::fill(used.begin(), used.end(), 0);
      used[ui] = 1;

      std::optional<BuiltCut> best;
      double best_violation = 0.0;
      int best_depth = 0;
      for (int depth = 0; depth <= kMaxAggregation; ++depth) {
        RowScan scan = scan_flow_terms(model, solution, vubs, coeff);
        if (scan.disqualifying < 0 && scan.terms.size() >= 2) {
          const std::vector<FlowTerm>& terms = scan.terms;
          double total_capacity = 0.0;
          for (const FlowTerm& t : terms) total_capacity += t.effective_capacity;
          if (total_capacity - rhs > kMinExcess) {
            // Three deterministic orderings: capacity, LP flow, and LP activation, each
            // descending. Any resulting prefix with positive excess is a valid cover
            // (flow_cover_cuts.hpp); the orderings only change whether the search happens to
            // find a violated one.
            std::vector<std::size_t> by_capacity(terms.size()), by_flow(terms.size()),
                by_binary(terms.size());
            for (std::size_t k = 0; k < terms.size(); ++k) {
              by_capacity[k] = by_flow[k] = by_binary[k] = k;
            }
            std::sort(by_capacity.begin(), by_capacity.end(),
                      [&](std::size_t a, std::size_t b2) {
                        return terms[a].effective_capacity > terms[b2].effective_capacity;
                      });
            std::sort(by_flow.begin(), by_flow.end(), [&](std::size_t a, std::size_t b2) {
              return terms[a].z_star > terms[b2].z_star;
            });
            std::sort(by_binary.begin(), by_binary.end(), [&](std::size_t a, std::size_t b2) {
              return terms[a].y_star > terms[b2].y_star;
            });

            std::optional<CoverResult> cover;
            for (const auto& order : {by_capacity, by_flow, by_binary}) {
              auto result = best_cover_in_order(terms, order, rhs);
              if (result && (!cover || result->violation > cover->violation)) {
                cover = std::move(result);
              }
            }
            if (cover && cover->violation > best_violation) {
              if (std::optional<BuiltCut> candidate = build_cut(n, terms, *cover, rhs)) {
                best = std::move(candidate);
                best_violation = cover->violation;
                best_depth = depth;
              }
            }
          }
        }
        if (depth == kMaxAggregation) break;
        if (scan.disqualifying < 0) break;  // a clean row with nothing left to eliminate
        if (!aggregate_eliminating(model, rows, scan.disqualifying, &used, &coeff, &rhs)) break;
      }
      if (best) {
        if (stats != nullptr) {
          ++stats->cuts;
          if (best_depth > 0) ++stats->aggregated_cuts;
          stats->deepest = std::max(stats->deepest, best_depth);
          stats->lifted_terms += best->lifted_terms;
          if (best->lifted_terms > 0) ++stats->lifted_cuts;
        }
        cuts.push_back(std::move(best->cut));
      }
    }
  }
  return cuts;
}

}  // namespace sankhya::mip
