// SPDX-License-Identifier: Apache-2.0
// SANKHYA - IIS deletion filter.
//
// Reference: Chinneck, J.W. and Dravnieks, E.W., "Locating minimal infeasible constraint
// sets in linear programs", ORSA Journal on Computing 3(2) (1991), pp. 157-168.
//
// The algorithm is O(k) solves where k is the cardinality of the Farkas certificate support
// (the rows and column bounds the engine named). In practice k << m, so each re-solve is on
// a small, quickly-decided infeasible system.

#include "core/iis.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "sankhya/certificate.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"
#include "util/profiler.hpp"

namespace sankhya {
namespace {

// THE WITNESS IS JUDGED THE WAY THE VERIFIER JUDGES IT (#475). A trial solve hands back a
// point the status guard calls feasible, and the guard measures a row's violation against the
// largest term of its activity sum. tools/verify_solution.py measures a witness against
// max(1, |activity|, |the row's bounds|), which is the smaller scale whenever the terms
// cancel: on Netlib klein2, from the interior point's certificate, four witnesses violated
// another IIS row by 1.0e-7 to 2.2e-7 in that measure, and the file was rejected. So every
// witness is checked here by the verifier's rule before irreducibility is claimed.
/// How far, in units of the primal tolerance in the verifier's measure, the other elements
/// are pulled inward when a witness is solved for again: the first that yields a witness
/// that holds is kept. A larger pull leaves more room for the solver's own tolerance, which
/// is measured against the terms of a row and not against its activity.
constexpr double kWitnessMargins[] = {0.5, 5.0, 50.0};

struct IisElement {
  int kind = 0;  // 0 a row, 1 a column lower bound, 2 a column upper bound
  Index index = 0;
};

double relative_violation(double value, double lower, double upper) {
  const double below = is_finite_bound(lower) ? lower - value : 0.0;
  const double above = is_finite_bound(upper) ? value - upper : 0.0;
  const double worst = std::max({below, above, 0.0});
  double scale = std::max(1.0, std::fabs(value));
  if (is_finite_bound(lower)) scale = std::max(scale, std::fabs(lower));
  if (is_finite_bound(upper)) scale = std::max(scale, std::fabs(upper));
  return worst / scale;
}

/// Does `point` satisfy every element of `iis` but `self` to `tolerance`, and violate `self`
/// by more than it? The verifier's test, element for element.
bool witness_holds(const Model& model, const std::vector<IisElement>& iis, std::size_t self,
                   const std::vector<double>& point, double tolerance) {
  if (point.size() != static_cast<std::size_t>(model.num_cols())) return false;
  std::vector<double> activity(static_cast<std::size_t>(model.num_rows()), 0.0);
  model.matrix.multiply_add(point.data(), activity.data());
  for (std::size_t e = 0; e < iis.size(); ++e) {
    const auto u = static_cast<std::size_t>(iis[e].index);
    double v = 0.0;
    if (iis[e].kind == 0) {
      v = relative_violation(activity[u], model.row_lower[u], model.row_upper[u]);
    } else if (iis[e].kind == 1) {
      v = relative_violation(point[u], model.col_lower[u], kInfinity);
    } else {
      v = relative_violation(point[u], -kInfinity, model.col_upper[u]);
    }
    if (e == self ? v <= tolerance : v > tolerance) return false;
  }
  return true;
}

/// A witness for `self` from the IIS `sub` with `self` removed and every other element pulled
/// inward by `margin` times max(1, |bound|) in the verifier's measure, so that the point the
/// solve stops at, within the solver's own tolerance of those bounds, is inside them by the
/// verifier's. Empty when that system is not found feasible.
std::vector<double> tightened_witness(const Model& sub, const std::vector<IisElement>& iis,
                                      std::size_t self, double margin,
                                      const Options& sub_opts) {
  Model trial = sub;
  const auto inward = [margin](double bound) {
    return margin * std::max(1.0, std::fabs(bound));
  };
  for (std::size_t e = 0; e < iis.size(); ++e) {
    const auto u = static_cast<std::size_t>(iis[e].index);
    if (iis[e].kind == 0) {
      double& lo = trial.row_lower[u];
      double& hi = trial.row_upper[u];
      if (e == self) {
        lo = -kInfinity;
        hi = kInfinity;
        continue;
      }
      if (is_finite_bound(lo)) lo += inward(lo);
      if (is_finite_bound(hi)) hi -= inward(hi);
      if (is_finite_bound(lo) && is_finite_bound(hi) && lo > hi) lo = hi = 0.5 * (lo + hi);
    } else if (iis[e].kind == 1) {
      double& lo = trial.col_lower[u];
      lo = e == self ? -kInfinity : lo + inward(lo);
    } else {
      double& hi = trial.col_upper[u];
      hi = e == self ? kInfinity : hi - inward(hi);
    }
    if (trial.col_lower[u] > trial.col_upper[u] && iis[e].kind != 0) {
      trial.col_lower[u] = trial.col_upper[u] = 0.5 * (trial.col_lower[u] + trial.col_upper[u]);
    }
  }
  const Solution solved = solve(trial, sub_opts);
  if (solved.status != SolveStatus::kOptimal && solved.status != SolveStatus::kFeasible) {
    return {};
  }
  return solved.col_value;
}

}  // namespace

void compute_iis(const Model& model, Solution* solution, const Options& options,
                 Logger& logger) {
  if (solution->status != SolveStatus::kInfeasible) return;
  if (solution->farkas_dual.empty()) return;
  if (!options.get_bool("compute_iis")) return;
  ProfileScope timed(logger.profiler(), "iis");  // #285, after the reasons not to run

  const Index m = model.num_rows();
  const Index n = model.num_cols();
  const std::vector<double>& y = solution->farkas_dual;

  // --- Step 1: identify candidate constraints from the Farkas certificate support ---------
  //
  // A multiplier y_i != 0 means row i's bound contributes to the aggregate S = sum y_i * b_i.
  // The column aggregate d = A'y determines which column bounds bound M = max d.x from above.
  // A candidate column lower bound j requires d[j] < 0 and col_lower[j] > -inf.
  // A candidate column upper bound j requires d[j] > 0 and col_upper[j] < +inf.

  // Zero is judged column by column against the terms each sum was built from, the rule
  // farkas_proves_infeasible() applies (#762), so every bound the verified certificate leans
  // on is a candidate here.
  double largest_y = 0.0;
  for (const double v : y) largest_y = std::max(largest_y, std::fabs(v));
  std::vector<double> d(static_cast<std::size_t>(n), 0.0);
  std::vector<double> threshold(static_cast<std::size_t>(n), 0.0);
  for (Index j = 0; j < n; ++j) {
    const ColumnView col = model.matrix.column(j);
    const auto u = static_cast<std::size_t>(j);
    for (Index k = 0; k < col.size; ++k) {
      const double multiplier = y[static_cast<std::size_t>(col.rows[k])];
      d[u] += col.values[k] * multiplier;
      if (multiplier != 0.0) {
        threshold[u] =
            std::max(threshold[u], tol::kZeroDrop * largest_y * std::fabs(col.values[k]));
      }
    }
  }

  std::vector<bool> row_in(static_cast<std::size_t>(m), false);
  std::vector<bool> col_lo_in(static_cast<std::size_t>(n), false);
  std::vector<bool> col_hi_in(static_cast<std::size_t>(n), false);

  Index k_rows = 0;
  Index k_cols = 0;
  for (Index i = 0; i < m; ++i) {
    if (y[static_cast<std::size_t>(i)] != 0.0) {
      row_in[static_cast<std::size_t>(i)] = true;
      ++k_rows;
    }
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (d[u] < -threshold[u] && is_finite_bound(model.col_lower[u])) {
      col_lo_in[u] = true;
      ++k_cols;
    }
    if (d[u] > threshold[u] && is_finite_bound(model.col_upper[u])) {
      col_hi_in[u] = true;
      ++k_cols;
    }
  }

  logger.info("IIS: deletion filter on {} row(s) and {} bound(s) from the Farkas certificate",
              k_rows, k_cols);

  // --- Step 2: build the restricted sub-model (only candidates active) -------------------
  //
  // The Farkas certificate proves that the candidate constraints alone are infeasible. Free
  // all non-candidates so that subsequent re-solves test only the candidates.

  Model sub = model;
  for (Index i = 0; i < m; ++i) {
    if (!row_in[static_cast<std::size_t>(i)]) {
      sub.row_lower[static_cast<std::size_t>(i)] = -kInfinity;
      sub.row_upper[static_cast<std::size_t>(i)] = +kInfinity;
    }
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (!col_lo_in[u]) sub.col_lower[u] = -kInfinity;
    if (!col_hi_in[u]) sub.col_upper[u] = +kInfinity;
  }

  // Zero the cost vector so trial solves are pure feasibility checks, not optimization.
  // Otherwise a sub-solve may return kUnbounded instead of kInfeasible.
  sub.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  sub.objective_offset = 0.0;

  // Sub-options: suppress logging, disable IIS recursion, use the dual simplex (fastest at
  // proving infeasibility from a dual-feasible start), disable presolve (the sub-model has
  // few rows and simple structure).
  Options sub_opts = options;
  sub_opts.set_bool("compute_iis", false);
  sub_opts.set_bool("log_to_console", false);
  sub_opts.set_string("algorithm", "dual-simplex");
  sub_opts.set_bool("presolve", false);
  // Each sub-solve is on a tiny system; cap at 5 s to avoid hanging on pathological models.
  const double parent_limit = options.get_double("time_limit");
  const double sub_limit =
      (parent_limit > 0.0 && std::isfinite(parent_limit)) ? std::min(parent_limit, 5.0) : 5.0;
  sub_opts.set_double("time_limit", sub_limit);
  // Prevent trial solves from appending to the user's progress file.
  sub_opts.set_string("progress_out", "");

  // --- Step 3: deletion filter -----------------------------------------------------------
  //
  // For each candidate, free it in the current working sub-model and re-solve. If still
  // infeasible the candidate is redundant (drop it permanently). If feasible the candidate
  // is necessary (restore it in the working model).

  // Returns true when the candidate is redundant (sub-problem still infeasible without it).
  // Inconclusive statuses (time limit, numerical error) are treated conservatively: the
  // candidate is kept (not dropped), and a warning is emitted so the caller knows the
  // result may not be fully irreducible.
  // THE PROOF OF NECESSITY IS THE TRIAL ITSELF. A trial that comes back feasible hands over
  // a point that satisfies the working system minus the candidate, and the final IIS is a
  // subset of the working system, so that point satisfies every IIS element but the one
  // removed. Kept per candidate and written as witnesses, it lets the verifier check
  // irreducibility with arithmetic alone (#217, third acceptance box).
  std::vector<std::vector<double>> row_witness(static_cast<std::size_t>(m));
  std::vector<std::vector<double>> lo_witness(static_cast<std::size_t>(n));
  std::vector<std::vector<double>> hi_witness(static_cast<std::size_t>(n));

  bool inconclusive = false;
  const auto is_redundant = [&](const Solution& trial, const char* kind,
                                std::size_t idx) -> bool {
    if (trial.status == SolveStatus::kInfeasible) return true;
    if (trial.status == SolveStatus::kOptimal || trial.status == SolveStatus::kFeasible)
      return false;
    logger.warning(
        "IIS: {} {} trial inconclusive ({}); candidate kept, result may not be "
        "irreducible",
        kind, idx, to_string(trial.status));
    inconclusive = true;
    return false;
  };

  // Row candidates.
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (!row_in[u]) continue;

    // Trial: free row i.
    const double saved_lo = sub.row_lower[u];
    const double saved_hi = sub.row_upper[u];
    sub.row_lower[u] = -kInfinity;
    sub.row_upper[u] = +kInfinity;

    const Solution trial = solve(sub, sub_opts);
    if (is_redundant(trial, "row", u)) {
      row_in[u] = false;
    } else {
      sub.row_lower[u] = saved_lo;
      sub.row_upper[u] = saved_hi;
      if (claims_a_point(trial.status)) row_witness[u] = trial.col_value;
    }
  }

  // Column lower bound candidates.
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (!col_lo_in[u]) continue;

    const double saved_lo = sub.col_lower[u];
    sub.col_lower[u] = -kInfinity;

    const Solution trial = solve(sub, sub_opts);
    if (is_redundant(trial, "col_lo", u)) {
      col_lo_in[u] = false;
    } else {
      sub.col_lower[u] = saved_lo;
      if (claims_a_point(trial.status)) lo_witness[u] = trial.col_value;
    }
  }

  // Column upper bound candidates.
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (!col_hi_in[u]) continue;

    const double saved_hi = sub.col_upper[u];
    sub.col_upper[u] = +kInfinity;

    const Solution trial = solve(sub, sub_opts);
    if (is_redundant(trial, "col_hi", u)) {
      col_hi_in[u] = false;
    } else {
      sub.col_upper[u] = saved_hi;
      if (claims_a_point(trial.status)) hi_witness[u] = trial.col_value;
    }
  }

  // --- Step 4: the certificate of the IIS itself ------------------------------------------
  //
  // `sub` is now exactly the IIS: every other row and bound is free. One more solve hands
  // back a Farkas vector whose support lies inside the IIS, and that vector - re-proved
  // against the caller's model, as every certificate is - replaces the full model's, so
  // the checker's existing Farkas test proves the subsystem infeasible on its own. The
  // relaxed rows carry zero multipliers and the relaxed bounds are infinite, so a vector
  // that proves `sub` infeasible proves the tighter original infeasible too; the re-proof
  // is the check on that reasoning, not a substitute for it.
  {
    const Solution final_trial = solve(sub, sub_opts);
    std::string why;
    if (final_trial.status == SolveStatus::kInfeasible &&
        final_trial.farkas_dual.size() == static_cast<std::size_t>(m) &&
        farkas_proves_infeasible(model, final_trial.farkas_dual, &why)) {
      solution->farkas_dual = final_trial.farkas_dual;
    } else {
      logger.warning(
          "IIS: the final subsystem's certificate could not be established ({}); the "
          "certificate written is the full model's, whose support may reach outside the IIS",
          final_trial.status == SolveStatus::kInfeasible ? why : to_string(final_trial.status));
      inconclusive = true;
    }
  }

  // --- Step 5: collect and report --------------------------------------------------------

  for (Index i = 0; i < m; ++i) {
    if (row_in[static_cast<std::size_t>(i)]) solution->iis_rows.push_back(i);
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (col_lo_in[u]) solution->iis_col_lo.push_back(j);
    if (col_hi_in[u]) solution->iis_col_hi.push_back(j);
  }
  // Witnesses in the same order as the elements; an element that kept its place on an
  // inconclusive trial has none, and then none are claimed at all.
  std::vector<std::vector<double>> witnesses;
  bool every_witness = true;
  for (const Index i : solution->iis_rows) {
    auto& w = row_witness[static_cast<std::size_t>(i)];
    if (w.size() != static_cast<std::size_t>(n)) every_witness = false;
    witnesses.push_back(std::move(w));
  }
  for (const Index j : solution->iis_col_lo) {
    auto& w = lo_witness[static_cast<std::size_t>(j)];
    if (w.size() != static_cast<std::size_t>(n)) every_witness = false;
    witnesses.push_back(std::move(w));
  }
  for (const Index j : solution->iis_col_hi) {
    auto& w = hi_witness[static_cast<std::size_t>(j)];
    if (w.size() != static_cast<std::size_t>(n)) every_witness = false;
    witnesses.push_back(std::move(w));
  }
  if (!every_witness) inconclusive = true;
  if (!inconclusive) {
    // Each witness is held to the verifier's own measure; one that misses it is solved for
    // again with the other elements pulled inward, and when that does not produce one that
    // holds either, irreducibility is not claimed. The certificate is untouched by this.
    const double tolerance = options.get_double("primal_feasibility_tolerance");
    std::vector<IisElement> iis;
    for (const Index i : solution->iis_rows) iis.push_back({0, i});
    for (const Index j : solution->iis_col_lo) iis.push_back({1, j});
    for (const Index j : solution->iis_col_hi) iis.push_back({2, j});
    Index repaired = 0;
    for (std::size_t e = 0; e < iis.size() && !inconclusive; ++e) {
      if (witness_holds(model, iis, e, witnesses[e], tolerance)) continue;
      bool held = false;
      for (const double share : kWitnessMargins) {
        std::vector<double> retry = tightened_witness(sub, iis, e, share * tolerance, sub_opts);
        if (witness_holds(model, iis, e, retry, tolerance)) {
          witnesses[e] = std::move(retry);
          held = true;
          break;
        }
      }
      if (held) {
        ++repaired;
        continue;
      }
      logger.warning(
          "IIS: no witness for element {} holds to {:.0e} in the verifier's measure; "
          "irreducibility is not claimed",
          e, tolerance);
      inconclusive = true;
    }
    if (repaired > 0) {
      logger.info("IIS: {} witness(es) solved again with the other elements pulled inward",
                  repaired);
    }
  }
  if (!inconclusive) solution->iis_witnesses = std::move(witnesses);
  solution->iis_inconclusive = inconclusive;

  logger.info("IIS: {} row(s), {} lower bound(s), {} upper bound(s) are irreducible",
              solution->iis_rows.size(), solution->iis_col_lo.size(),
              solution->iis_col_hi.size());
  if (inconclusive) {
    logger.warning(
        "IIS: at least one trial was inconclusive; the reported IIS may not be "
        "fully irreducible");
  }
}

}  // namespace sankhya
