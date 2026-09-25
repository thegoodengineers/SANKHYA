// SPDX-License-Identifier: Apache-2.0
// SANKHYA - certifying cut rows for a MILP certificate (#518). See cut_derivation.hpp for
// the two derivation forms and their references.

#include "cut_derivation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <span>

#include <fmt/format.h>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

bool is_exact_integer(double v) {
  return std::isfinite(v) && v == std::trunc(v) &&
         std::fabs(v) <= tol::kCertificateExactInteger;
}

/// Everything one derivation needs: the rows the cut may use and, after them, one row per
/// split cut of the batch holding its disjunction. Those rows are free except while their
/// own cut is being proved, so no other cut's assumption can enter a proof.
struct Problem {
  SparseMatrix matrix;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::span<const double> col_lower;
  std::span<const double> col_upper;
  Index rows_seen = 0;  ///< rows before the disjunction rows
};

/// Prove `cut` (a.x <= rhs) from minimise-space multipliers `y` over `problem`: the
/// combination completed over the box. Returns an upper bound on what the exact combination
/// proves, +inf when it proves nothing, and the multipliers and implied bounds it used.
double prove_side(const Problem& problem, const Cut& cut, std::vector<double> y,
                  SafeBoundDetail* used) {
  std::vector<double> cost(cut.coeff.size());
  for (std::size_t j = 0; j < cost.size(); ++j) cost[j] = -cut.coeff[j];
  SafeBoundProblem p;
  p.matrix = &problem.matrix;
  p.cost = cost;
  p.row_lower = problem.row_lower;
  p.row_upper = problem.row_upper;
  p.col_lower = problem.col_lower;
  p.col_upper = problem.col_upper;
  const SafeBound bound = safe_dual_bound(p, y, true, used);
  if (!std::isfinite(bound.value)) return kInfinity;
  return -bound.value;  // -a.x >= value, so a.x <= -value; negation is exact
}

CutProof::Side side_of(const SafeBoundDetail& used, Index rows_seen, Index disjunction_row) {
  CutProof::Side side;
  for (Index i = 0; i < rows_seen; ++i) {
    const double v = used.multipliers[static_cast<std::size_t>(i)];
    if (v != 0.0) side.rows.emplace_back(i, v);
  }
  if (disjunction_row >= 0) {
    side.assumption = used.multipliers[static_cast<std::size_t>(disjunction_row)];
  }
  for (ImpliedBound b : used.implied) {
    if (b.row == disjunction_row) b.row = rows_seen;
    side.implied.push_back(b);
  }
  return side;
}

/// The multiplier lambda >= 0 on row `row`'s upper side that minimises what
/// lambda * row + completion proves for the cut, among the row's breakpoints a_j / A_j
/// (the bound is convex and piecewise linear in lambda, so a breakpoint attains it).
double best_row_multiplier(const CsrView& by_row, const Problem& problem, const Cut& cut,
                           Index row) {
  const ColumnView entries = by_row.row(row);
  const double b = problem.row_upper[static_cast<std::size_t>(row)];
  std::map<Index, double> weight;  // A_j on the row's columns
  for (Index p = 0; p < entries.size; ++p) weight[entries.rows[p]] = entries.values[p];
  std::vector<double> candidates;
  for (const auto& [j, a] : weight) {
    const double c = cut.coeff[static_cast<std::size_t>(j)];
    if (a != 0.0 && c / a > 0.0) candidates.push_back(c / a);
  }
  const auto value_at = [&](double lambda) {
    double total = lambda * b;
    for (const auto& [j, a] : weight) {
      const auto u = static_cast<std::size_t>(j);
      const double e = cut.coeff[u] - lambda * a;
      if (e == 0.0) continue;
      const double bound = e > 0.0 ? problem.col_upper[u] : problem.col_lower[u];
      total += e * bound;
    }
    return total;
  };
  double best = 0.0;
  double best_value = kInfinity;
  constexpr std::size_t kMostCandidates = 2000;  // O(k^2) below; covers are short
  if (candidates.size() > kMostCandidates) candidates.resize(kMostCandidates);
  for (const double lambda : candidates) {
    const double v = value_at(lambda);
    if (std::isfinite(v) && v < best_value) {
      best_value = v;
      best = lambda;
    }
  }
  return best;
}

std::string family_summary(const std::map<int, std::pair<Count, Count>>& tally) {
  std::string out;
  for (const auto& [family, counts] : tally) {
    if (!out.empty()) out += "; ";
    out += fmt::format("{} {} derived", cut_family_name(static_cast<CutFamily>(family)),
                       counts.first);
    if (counts.second > 0) out += fmt::format(", {} dropped", counts.second);
  }
  return out.empty() ? "no cuts" : out;
}

}  // namespace

CutCertification certify_cuts(const Model& rows, std::span<const double> col_lower,
                              std::span<const double> col_upper, std::vector<Cut>* cuts) {
  CutCertification result;
  const Index m = rows.num_rows();
  const Index n = rows.num_cols();
  // One disjunction row per split cut, after the rows the cuts may use.
  std::vector<Index> disjunction_row(cuts->size(), -1);
  Index extra = 0;
  for (std::size_t c = 0; c < cuts->size(); ++c) {
    const auto& d = (*cuts)[c].derivation;
    if (d && d->kind == CutDerivation::Kind::kSplit) disjunction_row[c] = m + extra++;
  }
  Problem problem;
  problem.rows_seen = m;
  problem.col_lower = col_lower;
  problem.col_upper = col_upper;
  problem.matrix.reset(m + extra, n);
  for (Index j = 0; j < n; ++j) {
    const ColumnView col = rows.matrix.column(j);
    for (Index k = 0; k < col.size; ++k)
      problem.matrix.add_entry(col.rows[k], j, col.values[k]);
  }
  for (std::size_t c = 0; c < cuts->size(); ++c) {
    if (disjunction_row[c] < 0) continue;
    for (const auto& [j, v] : (*cuts)[c].derivation->disjunction) {
      problem.matrix.add_entry(disjunction_row[c], j, v);
    }
  }
  problem.matrix.finalize(0.0);  // every entry the checker's rows have, however small
  problem.row_lower.assign(rows.row_lower.begin(), rows.row_lower.end());
  problem.row_upper.assign(rows.row_upper.begin(), rows.row_upper.end());
  problem.row_lower.resize(static_cast<std::size_t>(m + extra), -kInfinity);
  problem.row_upper.resize(static_cast<std::size_t>(m + extra), kInfinity);
  std::optional<CsrView> by_row;

  std::map<int, std::pair<Count, Count>> tally;
  std::vector<Cut> kept;
  kept.reserve(cuts->size());
  const auto total = static_cast<std::size_t>(m + extra);
  for (std::size_t c = 0; c < cuts->size(); ++c) {
    Cut& cut = (*cuts)[c];
    auto& counts = tally[static_cast<int>(cut.family)];
    const auto fail = [&](std::string reason) {
      ++counts.second;
      ++result.dropped;
      if (result.first_failure.empty()) {
        result.first_failure = fmt::format("{} cut: {}", cut_family_name(cut.family), reason);
      }
    };
    for (double& a : cut.coeff) {
      if (std::fabs(a) <= tol::kZeroDrop) a = 0.0;
    }
    const std::shared_ptr<const CutDerivation> d = cut.derivation;
    if (!d) {
      fail("its family states no derivation");
      continue;
    }
    auto proof = std::make_shared<CutProof>();
    proof->rows_seen = m;
    if (d->kind != CutDerivation::Kind::kSplit) {
      // CHVATAL-GOMORY: integer coefficients on integer columns, floor of what the
      // combination proves.
      bool integral = true;
      for (Index j = 0; j < n && integral; ++j) {
        const double a = cut.coeff[static_cast<std::size_t>(j)];
        integral =
            a == 0.0 || (rows.col_type[static_cast<std::size_t>(j)] == VarType::kInteger &&
                         is_exact_integer(a));
      }
      if (!integral) {
        fail("a rounding needs integer coefficients on integer columns");
        continue;
      }
      std::vector<std::pair<Index, double>> w = d->rows[0];
      if (d->kind == CutDerivation::Kind::kSingleRowRounding) {
        if (d->row < 0 || d->row >= m ||
            !is_finite_bound(problem.row_upper[static_cast<std::size_t>(d->row)])) {
          fail("the row it rounds has no upper side");
          continue;
        }
        if (!by_row) by_row.emplace(problem.matrix);
        w = {{d->row, best_row_multiplier(*by_row, problem, cut, d->row)}};
      }
      std::vector<double> y(total, 0.0);
      for (const auto& [i, v] : w) {
        if (i >= 0 && i < m) y[static_cast<std::size_t>(i)] -= v;
      }
      SafeBoundDetail used;
      const double proved = prove_side(problem, cut, std::move(y), &used);
      if (!std::isfinite(proved) || std::floor(proved) > cut.rhs) {
        fail(fmt::format("the rounding proves {:.17g}, not {:.17g}", std::floor(proved),
                         cut.rhs));
        continue;
      }
      proof->side[0] = side_of(used, m, -1);
    } else {
      // A SPLIT: each side of the disjunction proves the cut; the weaker of the two is
      // what the certificate states.
      bool valid = !d->disjunction.empty() && is_exact_integer(d->disjunction_rhs) &&
                   is_exact_integer(d->disjunction_rhs + 1.0);
      for (const auto& [j, v] : d->disjunction) {
        valid = valid && j >= 0 && j < n && is_exact_integer(v) && v != 0.0 &&
                rows.col_type[static_cast<std::size_t>(j)] == VarType::kInteger;
      }
      if (!valid) {
        fail("its disjunction is not an integer split");
        continue;
      }
      const auto r = static_cast<std::size_t>(disjunction_row[c]);
      double proved = -kInfinity;
      for (int s = 0; s < 2; ++s) {
        problem.row_lower[r] = s == 0 ? -kInfinity : d->disjunction_rhs + 1.0;
        problem.row_upper[r] = s == 0 ? d->disjunction_rhs : kInfinity;
        std::vector<double> y(total, 0.0);
        for (const auto& [i, v] : d->rows[s]) {
          if (i >= 0 && i < m) y[static_cast<std::size_t>(i)] -= v;
        }
        y[r] = -d->mu[s];
        SafeBoundDetail used;
        const double side = prove_side(problem, cut, std::move(y), &used);
        proved = std::max(proved, side);
        if (std::isfinite(side)) proof->side[s] = side_of(used, m, disjunction_row[c]);
      }
      problem.row_lower[r] = -kInfinity;
      problem.row_upper[r] = kInfinity;
      const double slack = tol::kCertificateCutSlack * std::max(1.0, std::fabs(cut.rhs));
      if (!std::isfinite(proved) || proved > cut.rhs + slack) {
        fail(fmt::format("the split proves {:.17g}, the cut says {:.17g}", proved, cut.rhs));
        continue;
      }
      if (proved > cut.rhs) {
        cut.rhs = proved;
        ++result.relaxed;
      }
      proof->split = true;
      proof->disjunction = d->disjunction;
      proof->disjunction_rhs = d->disjunction_rhs;
    }
    cut.proof = std::move(proof);
    ++counts.first;
    ++result.derived;
    kept.push_back(std::move(cut));
  }
  *cuts = std::move(kept);
  result.summary = family_summary(tally);
  return result;
}

std::shared_ptr<const CutDerivation> gmi_derivation(
    const Model& model, const detail::ReconstructedTableauRow& tableau) {
  if (!tableau.basic_is_structural) return nullptr;
  const double f0 = tableau.rhs - std::floor(tableau.rhs);
  if (!(f0 > 0.0 && f0 < 1.0)) return nullptr;
  auto d = std::make_shared<CutDerivation>();
  d->kind = CutDerivation::Kind::kSplit;
  // The disjunction: x_B + sum_j pi_j t_j <= floor(beta) | >= floor(beta) + 1, over the
  // integer nonbasic columns at an integral bound, t_j their distance from that bound, pi_j
  // the rounding of the tableau entry that Gomory's formula picks (down when f_j <= f0).
  d->disjunction.emplace_back(tableau.basic_index, 1.0);
  double rhs = std::floor(tableau.rhs);
  for (std::size_t j = 0; j < tableau.structural_coefs.size(); ++j) {
    const BasisStatus status = tableau.structural_status[j];
    if (status != BasisStatus::kAtLower && status != BasisStatus::kAtUpper) continue;
    if (model.col_type[j] != VarType::kInteger) continue;
    const double s = status == BasisStatus::kAtLower ? 1.0 : -1.0;
    const double bound = std::round(s > 0.0 ? model.col_lower[j] : model.col_upper[j]);
    if (!std::isfinite(bound) ||
        std::fabs(bound - (s > 0.0 ? model.col_lower[j] : model.col_upper[j])) >
            tol::kIntegrality) {
      continue;  // Gomory treats this column as continuous
    }
    const double alpha = s * tableau.structural_coefs[j];
    const double fj = alpha - std::floor(alpha);
    const double pi = fj <= f0 ? std::floor(alpha) : std::ceil(alpha);
    if (pi == 0.0) continue;
    d->disjunction.emplace_back(static_cast<Index>(j), s * pi);
    rhs += s * pi * bound;
  }
  d->disjunction_rhs = rhs;
  // The rows: the nonnegativity of each nonbasic logical t_i, weighted by what the cut's
  // coefficient on it exceeds the tableau's (the file header's derivation).
  for (std::size_t i = 0; i < tableau.logical_coefs.size(); ++i) {
    const BasisStatus status = tableau.logical_status[i];
    double s = 1.0;
    if (status == BasisStatus::kAtUpper) {
      s = -1.0;
    } else if (status != BasisStatus::kAtLower && status != BasisStatus::kFixed) {
      continue;
    }
    const double alpha = s * tableau.logical_coefs[i];
    double gamma = 0.0;
    if (status != BasisStatus::kFixed && std::fabs(alpha) > tol::kZeroDrop) {
      gamma = alpha > 0.0 ? alpha / f0 : -alpha / (1.0 - f0);
    }
    const double down = gamma - alpha / f0;
    const double up = gamma + alpha / (1.0 - f0);
    if (down != 0.0) d->rows[0].emplace_back(static_cast<Index>(i), -s * down);
    if (up != 0.0) d->rows[1].emplace_back(static_cast<Index>(i), -s * up);
  }
  d->mu[0] = 1.0 / f0;
  d->mu[1] = -1.0 / (1.0 - f0);
  return d;
}

}  // namespace sankhya::mip
