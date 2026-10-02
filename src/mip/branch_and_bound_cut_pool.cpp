// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cut pool of branch and cut (#497): ageing the cut rows the cut rounds
// (branch_and_bound_cuts.cpp) appended, removing aged rows from the node LP, holding them in
// the pool, and putting a pooled cut back in force when a node's LP point violates it.
//
// See THE POOL in branch_and_bound_cuts.cpp for the whole design; in brief: with
// `mip_cut_pooling` an aged row is deleted from working_ and appended again when violated,
// cut_lp_rows_ says which pooled cut each cut row is, and an open node's basis is remapped
// onto the rows of the moment when the node is entered. Under write_certificate rows are
// freed in place instead and nothing is deleted.
//
// Achterberg, "Constraint integer programming", PhD thesis, TU Berlin (2007), ch. 8 (row
// ageing and the cut pool).

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "branch_and_bound_internal.hpp"
#include "cut_pool_audit.hpp"

namespace sankhya::mip {

namespace {

double activity_at(const Cut& cut, const std::vector<double>& x) {
  double activity = 0.0;
  for (std::size_t j = 0; j < cut.coeff.size() && j < x.size(); ++j)
    activity += cut.coeff[j] * x[j];
  return activity;
}

/// Row `row` of `model` is `cut`, as append_pool_rows() writes it: the same entries above
/// the drop tolerance, nothing else, and `<=` its right-hand side. For the test seam only.
bool row_is(const Model& model, Index row, const Cut& cut) {
  if (row < 0 || row >= model.num_rows()) return false;
  const auto r = static_cast<std::size_t>(row);
  if (model.row_lower[r] != -kInfinity || model.row_upper[r] != cut.rhs) return false;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double want =
        u < cut.coeff.size() && std::fabs(cut.coeff[u]) > tol::kZeroDrop ? cut.coeff[u] : 0.0;
    double have = 0.0;
    const ColumnView view = model.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) {
      if (view.rows[k] == row) have += view.values[k];
    }
    if (have != want) return false;
  }
  return true;
}

}  // namespace

CutPoolHook& cut_pool_audit_for_testing() {
  static CutPoolHook hook;
  return hook;
}

std::shared_ptr<const std::vector<std::size_t>> BranchAndBound::cut_layout() {
  if (!cut_removal_) return nullptr;
  if (!cut_layout_)
    cut_layout_ = std::make_shared<const std::vector<std::size_t>>(cut_lp_rows_);
  return cut_layout_;
}

void BranchAndBound::age_cut_rows(Solution* relaxation) {
  if (first_cut_row_ < 0) return;
  if (relaxation->row_status.size() != static_cast<std::size_t>(working_.num_rows())) return;
  // Positions in cut_lp_rows_ of the rows that age out here, when they are to be removed.
  std::vector<std::size_t> removed;
  for (std::size_t p = 0; p < cut_lp_rows_.size(); ++p) {
    const std::size_t k = cut_lp_rows_[p];
    if (cut_row_free_[k]) continue;
    const auto row = static_cast<std::size_t>(first_cut_row_) + p;
    if (relaxation->row_status[row] == BasisStatus::kBasic) {
      if (++cut_row_slack_[k] >= cut_age_limit_) {
        cut_row_free_[k] = true;
        ++cut_rows_aged_out_;
        if (cut_removal_) {
          removed.push_back(p);
        } else {
          working_.row_lower[row] = -kInfinity;
          working_.row_upper[row] = kInfinity;
        }
      }
    } else {
      cut_row_slack_[k] = 0;
    }
  }
  if (!removed.empty()) remove_cut_rows(removed, relaxation);
}

// REMOVING A ROW WHOSE LOGICAL IS BASIC (#497). Let B be the basis matrix of `relaxation`
// over [A | -I]. The logical of row i is the column -e_i, whose only entry is in row i, so
// expanding det(B) along that column gives det(B) = +-det(B'), where B' is B without row i and
// without that column: the remaining basic columns are a basis of the LP without row i. The
// duals are unchanged (a basic logical prices to zero, so y_i = 0), the point is unchanged,
// and so the solution, cut down to the remaining rows, is optimal for the smaller LP. Its
// bound can only be weaker than the LP's with the row, and the row is a valid cut, so the
// smaller LP is still a relaxation of the node: nothing the search proves from it is wrong.
void BranchAndBound::remove_cut_rows(const std::vector<std::size_t>& positions,
                                     Solution* relaxation) {
  const Index old_rows = working_.num_rows();
  const Index cols = working_.num_cols();
  const auto rows_u = static_cast<std::size_t>(old_rows);
  std::vector<char> gone(rows_u, 0);
  for (const std::size_t p : positions) {
    const std::size_t row = static_cast<std::size_t>(first_cut_row_) + p;
    assert(row < rows_u);
    assert(relaxation->row_status[row] == BasisStatus::kBasic);
    gone[row] = 1;
  }
  std::vector<Index> renumbered(rows_u, -1);
  Index kept = 0;
  for (std::size_t i = 0; i < rows_u; ++i) {
    if (gone[i] == 0) renumbered[i] = kept++;
  }

  SparseMatrix matrix(kept, cols);
  for (Index j = 0; j < cols; ++j) {
    const ColumnView view = working_.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) {
      const Index to = renumbered[static_cast<std::size_t>(view.rows[k])];
      if (to >= 0) matrix.add_entry(to, j, view.values[k]);
    }
  }
  matrix.finalize();
  // Every row-indexed vector of the same length loses the same entries.
  const auto compact = [&](auto& values) {
    if (values.size() != rows_u) return;
    std::size_t to = 0;
    for (std::size_t i = 0; i < rows_u; ++i) {
      if (gone[i] != 0) continue;
      if (to != i) values[to] = std::move(values[i]);
      ++to;
    }
    values.resize(to);
  };
  compact(working_.row_lower);
  compact(working_.row_upper);
  compact(working_.row_names);
  working_.matrix = std::move(matrix);
  compact(relaxation->row_activity);
  compact(relaxation->row_dual);
  compact(relaxation->row_status);
  compact(relaxation->farkas_dual);
  compact(relaxation->row_ranging_lower);
  compact(relaxation->row_ranging_upper);

  const std::vector<std::size_t> before = cut_lp_rows_;
  std::size_t to = 0;
  for (std::size_t p = 0, next = 0; p < cut_lp_rows_.size(); ++p) {
    if (next < positions.size() && positions[next] == p) {
      ++next;
      continue;
    }
    cut_lp_rows_[to++] = cut_lp_rows_[p];
  }
  cut_lp_rows_.resize(to);
  cut_layout_.reset();
  assert(working_.matrix.num_rows() == working_.num_rows());
  assert(static_cast<Index>(cut_lp_rows_.size()) == working_.num_rows() - first_cut_row_);
  remap_warm_start(&current_warm_, before);
  cut_rows_removed_ += static_cast<Count>(positions.size());
  scaling_ = build_node_scaling(working_, node_options_);
  if (const CutPoolHook& hook = cut_pool_audit_for_testing(); hook) {
    for (const std::size_t p : positions) {
      CutPoolEvent event;
      event.kind = CutPoolEvent::Kind::kRemoved;
      event.pool_index = before[p];
      event.activity = activity_at(pool_cuts_[before[p]], relaxation->col_value);
      event.rhs = pool_cuts_[before[p]].rhs;
      event.lp_rows = working_.num_rows();
      hook(event);
    }
  }
}

// A BASIS OVER OTHER CUT ROWS (#497). An open node keeps the basis its parent's LP ended on,
// over the cut rows of that moment; by the time the node is entered, rows may have been
// removed and others appended. Rows kept keep their statuses. A row appended since gets its
// logical basic: B extended by a row and that logical is nonsingular (the same argument as
// above, backwards), and it is how every stored basis is extended when a cut round appends
// rows. A row removed since whose logical was basic simply goes, as above. A row removed
// whose logical was NONBASIC leaves one basic variable too many, so one basic column of that
// cut, the one with the largest coefficient in it, is made nonbasic at a bound in its stead.
// That basis may be singular or dual infeasible at the node; the simplex's basis repair (#34)
// swaps dependent columns for logicals when it factorizes a warm start, and the dual simplex
// starts from a dual infeasible basis by boxing (dual_simplex.cpp), so a poor choice costs
// iterations and never a wrong answer. A basis that cannot be remapped at all is dropped,
// and the node is solved cold.
void BranchAndBound::remap_warm_start(WarmStart* warm, const std::vector<std::size_t>& from) {
  if (warm->empty() || from == cut_lp_rows_) return;
  const auto base =
      static_cast<std::size_t>(first_cut_row_ >= 0 ? first_cut_row_ : working_.num_rows());
  const auto drop = [&] {
    *warm = WarmStart{};
    ++warm_starts_dropped_;
  };
  if (warm->row_status.size() != base + from.size() ||
      warm->col_status.size() != static_cast<std::size_t>(working_.num_cols())) {
    drop();
    return;
  }
  // Where each pooled cut sat among `from`'s rows, or -1.
  std::vector<std::ptrdiff_t> where(pool_cuts_.size(), -1);
  for (std::size_t q = 0; q < from.size(); ++q) {
    if (from[q] >= pool_cuts_.size()) {  // a cut a rollback took back since: cannot happen
      drop();
      return;
    }
    where[from[q]] = static_cast<std::ptrdiff_t>(q);
  }
  std::vector<BasisStatus> rows(base + cut_lp_rows_.size(), BasisStatus::kBasic);
  std::copy(warm->row_status.begin(),
            warm->row_status.begin() + static_cast<std::ptrdiff_t>(base), rows.begin());
  std::vector<char> kept(from.size(), 0);
  for (std::size_t p = 0; p < cut_lp_rows_.size(); ++p) {
    const std::ptrdiff_t q = where[cut_lp_rows_[p]];
    if (q < 0) continue;  // appended since: its logical is basic
    rows[base + p] = warm->row_status[base + static_cast<std::size_t>(q)];
    kept[static_cast<std::size_t>(q)] = 1;
  }
  for (std::size_t q = 0; q < from.size(); ++q) {
    if (kept[q] != 0 || warm->row_status[base + q] == BasisStatus::kBasic) continue;
    const Cut& cut = pool_cuts_[from[q]];
    std::size_t best = cut.coeff.size();
    double largest = 0.0;
    for (std::size_t j = 0; j < cut.coeff.size() && j < warm->col_status.size(); ++j) {
      const double a = std::fabs(cut.coeff[j]);
      if (warm->col_status[j] == BasisStatus::kBasic && a > tol::kZeroDrop && a > largest) {
        largest = a;
        best = j;
      }
    }
    if (best == cut.coeff.size()) {
      drop();
      return;
    }
    // The bound the global box has; seed_basis() moves it to the node's own bound.
    warm->col_status[best] = is_finite_bound(global_lower_[best]) ? BasisStatus::kAtLower
                             : is_finite_bound(global_upper_[best])
                                 ? BasisStatus::kAtUpper
                                 : BasisStatus::kNonbasicFree;
    ++warm_columns_demoted_;
  }
  warm->row_status = std::move(rows);
}

// #497: Achterberg, "Constraint Integer Programming" (thesis, TU Berlin, 2007), ch. 8 (the
// cut pool and row ageing). A freed cut row whose cut the node's LP point violates by more
// than the cut filter's violation tolerance gets its right-hand side back; the node is then
// re-solved from its own optimal basis, in which a freed row's logical is basic, so the
// start is dual feasible and only the re-imposed rows are primal infeasible - the same
// warm start a tree cut round uses. A row whose logical is not basic is left alone: making
// it basic would give the start one basic variable too many.
bool BranchAndBound::reactivate_pooled_cuts(Solution* relaxation) {
  if (first_cut_row_ < 0) return false;
  if (relaxation->row_status.size() != static_cast<std::size_t>(working_.num_rows())) {
    return false;
  }
  std::vector<std::size_t> reactivated;
  for (std::size_t k = 0; k < pool_cuts_.size(); ++k) {
    if (!cut_row_free_[k]) continue;
    const auto row = static_cast<std::size_t>(first_cut_row_) + k;
    if (relaxation->row_status[row] != BasisStatus::kBasic) continue;
    const Cut& cut = pool_cuts_[k];
    assert(cut.coeff.size() <= relaxation->col_value.size());
    double activity = 0.0;
    for (std::size_t j = 0; j < cut.coeff.size(); ++j) {
      activity += cut.coeff[j] * relaxation->col_value[j];
    }
    if (activity - cut.rhs <= tol::kCutViolationTolerance) continue;
    working_.row_lower[row] = -kInfinity;
    working_.row_upper[row] = cut.rhs;
    cut_row_free_[k] = false;
    cut_row_slack_[k] = 0;
    reactivated.push_back(k);
  }
  if (reactivated.empty()) return false;

  current_warm_ = basis_of(*relaxation);
  Solution after = solve_node();
  if (after.status == SolveStatus::kOptimal) {
    *relaxation = std::move(after);
    cuts_reactivated_ += static_cast<Count>(reactivated.size());
    return true;
  }
  // The re-solve did not finish: free the rows again and keep the node's optimal relaxation
  // without them, which is a weaker bound but a correct one (as a tree cut round rolls back).
  logger_.verbose("cut pool: re-imposing {} row(s) induced {}; rolled back", reactivated.size(),
                  to_string(after.status));
  for (const std::size_t k : reactivated) {
    const auto row = static_cast<std::size_t>(first_cut_row_) + k;
    working_.row_lower[row] = -kInfinity;
    working_.row_upper[row] = kInfinity;
    cut_row_free_[k] = true;
    cut_row_slack_[k] = cut_age_limit_;
  }
  current_warm_ = basis_of(*relaxation);
  return false;
}

// #497 with removal: the pool is checked before any new cut is separated at the node (the
// caller runs this right after the node LP), and a removed cut the point violates is
// appended again exactly as it was first appended - the same coefficients, the same
// right-hand side, valid for the whole tree then and still - with its logical basic. That is
// the tree cut round's warm start: the node's optimal basis extended by basic logicals is
// dual feasible, and only the appended rows are primal infeasible.
bool BranchAndBound::readd_pooled_cuts(Solution* relaxation) {
  if (first_cut_row_ < 0) return false;
  if (relaxation->row_status.size() != static_cast<std::size_t>(working_.num_rows())) {
    return false;
  }
  std::vector<std::size_t> violated;
  std::vector<double> activities;  // at the point that brought each back, for the test seam
  for (std::size_t k = 0; k < pool_cuts_.size(); ++k) {
    if (!cut_row_free_[k]) continue;
    const Cut& cut = pool_cuts_[k];
    assert(cut.coeff.size() <= relaxation->col_value.size());
    const double activity = activity_at(cut, relaxation->col_value);
    if (activity - cut.rhs > tol::kCutViolationTolerance) {
      violated.push_back(k);
      activities.push_back(activity);
    }
  }
  if (violated.empty()) return false;

  Model before = working_;
  auto before_scaling = scaling_;
  const std::size_t lp_before = cut_lp_rows_.size();
  append_pool_rows(violated);
  for (const std::size_t k : violated) {
    cut_row_free_[k] = false;
    cut_row_slack_[k] = 0;
  }
  resize_warm_starts(working_.num_rows());
  current_warm_ = basis_of(*relaxation);
  current_warm_.row_status.resize(static_cast<std::size_t>(working_.num_rows()),
                                  BasisStatus::kBasic);
  Solution after = solve_node();
  if (after.status == SolveStatus::kOptimal) {
    *relaxation = std::move(after);
    cuts_reactivated_ += static_cast<Count>(violated.size());
    cut_rows_readded_ += static_cast<Count>(violated.size());
    if (const CutPoolHook& hook = cut_pool_audit_for_testing(); hook) {
      for (std::size_t i = 0; i < violated.size(); ++i) {
        CutPoolEvent event;
        event.kind = CutPoolEvent::Kind::kReadded;
        event.pool_index = violated[i];
        event.activity = activities[i];
        event.rhs = pool_cuts_[violated[i]].rhs;
        event.lp_rows = working_.num_rows();
        event.row = first_cut_row_ + static_cast<Index>(lp_before + i);
        event.row_is_the_cut = row_is(working_, event.row, pool_cuts_[violated[i]]) &&
                               cut_lp_rows_[lp_before + i] == violated[i];
        hook(event);
      }
    }
    return true;
  }
  // As reactivate_pooled_cuts(): the rows go back to the pool, and the node's optimal
  // relaxation without them stands, a weaker bound but a correct one.
  logger_.verbose("cut pool: appending {} pooled row(s) induced {}; rolled back",
                  violated.size(), to_string(after.status));
  working_ = std::move(before);
  scaling_ = std::move(before_scaling);
  cut_lp_rows_.resize(lp_before);
  cut_layout_.reset();
  for (const std::size_t k : violated) {
    cut_row_free_[k] = true;
    cut_row_slack_[k] = cut_age_limit_;
  }
  resize_warm_starts(working_.num_rows());
  current_warm_ = basis_of(*relaxation);
  return false;
}

}  // namespace sankhya::mip
