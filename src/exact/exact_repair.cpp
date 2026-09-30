// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact repair of an LP basis that is optimal only to tolerance (#757). See
// exact_repair.hpp for the citations and the three steps.
//
// SIGN CONVENTIONS, minimise space, as exact_basis.hpp. x_B = B^{-1}(-N x_N) over [A | -I],
// so raising a nonbasic x_q by t moves the basic variable at position p by -alpha_p t, with
// alpha = B^{-1} a_q. The reduced cost d_q = c_q - a_q^T y is the objective's rate along it.
// A nonbasic variable at its lower bound is dual feasible with d >= 0, at its upper bound
// with d <= 0, free at 0 with d = 0; a fixed one (equal bounds) with any d.

#include "exact/exact_repair.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "exact/exact_basis.hpp"
#include "exact/rational.hpp"
#include "sankhya/types.hpp"

namespace sankhya::exact {
namespace {

using Sz = std::size_t;

/// Pivots and flips allowed before the repair declines. Bland's rule makes every phase
/// finite; this only bounds the time on a pathological model, as exact_seconds does.
constexpr int kMaxRepairSteps = 100000;

struct Declined {
  std::string why;
};
struct Failed {
  std::string why;
};

bool below(const Problem& problem, Sz k, const Rational& x) {
  return problem.has_lower[k] && x < problem.lower[k];
}
bool above(const Problem& problem, Sz k, const Rational& x) {
  return problem.has_upper[k] && x > problem.upper[k];
}

/// A nonbasic variable whose reduced cost has the wrong sign for where it sits.
bool dual_infeasible(const Problem& problem, const Basis& basis, Sz k) {
  const Nb st = basis.status[k];
  if (st == Nb::kBasic || problem.fixed(static_cast<Index>(k))) return false;
  const int s = basis.d[k].sign();
  return (st == Nb::kLower && s < 0) || (st == Nb::kUpper && s > 0) ||
         (st == Nb::kFree && s != 0);
}

class Repair {
 public:
  Repair(Problem* problem, Basis* basis, RepairResult* result, const Deadline& deadline)
      : problem_(*problem), basis_(*basis), result_(*result), deadline_(deadline) {}

  void run() {
    for (int round = 0; round < 3; ++round) {
      const bool primal = primal_infeasible_position().has_value();
      const bool dual = first_dual_infeasible().has_value();
      if (!primal && !dual) return;
      if (primal) {
        const std::vector<Rational> true_cost = problem_.cost;
        if (dual) shift_costs();
        while (const std::optional<Sz> p = primal_infeasible_position()) dual_step(*p);
        if (result_.shifted_costs > 0) {
          problem_.cost = true_cost;
          basis_.refresh(problem_);
        }
      } else {
        while (const std::optional<Sz> q = first_dual_infeasible()) primal_step(*q);
      }
    }
  }

 private:
  void count_step() {
    deadline_.check();
    if (result_.dual_pivots + result_.primal_pivots + result_.bound_flips >= kMaxRepairSteps) {
      throw Declined{fmt::format("the repair did not finish within {} steps", kMaxRepairSteps)};
    }
  }

  /// The position of the basic variable with the smallest index outside its bounds.
  [[nodiscard]] std::optional<Sz> primal_infeasible_position() const {
    std::optional<Sz> best;
    Index best_k = -1;
    for (Sz p = 0; p < basis_.basic.size(); ++p) {
      const Index k = basis_.basic[p];
      const auto u = static_cast<Sz>(k);
      if (!below(problem_, u, basis_.value[u]) && !above(problem_, u, basis_.value[u]))
        continue;
      if (best_k < 0 || k < best_k) {
        best = p;
        best_k = k;
      }
    }
    return best;
  }

  [[nodiscard]] std::optional<Sz> first_dual_infeasible() const {
    for (Sz k = 0; k < basis_.status.size(); ++k) {
      if (dual_infeasible(problem_, basis_, k)) return k;
    }
    return std::nullopt;
  }

  /// Step 2: c_k -= d_k on every dual infeasible nonbasic k, so d_k = 0. y does not change,
  /// since only nonbasic costs move.
  void shift_costs() {
    for (Sz k = 0; k < basis_.status.size(); ++k) {
      if (!dual_infeasible(problem_, basis_, k)) continue;
      problem_.cost[k] -= basis_.d[k];
      basis_.d[k] = Rational(0);
      ++result_.shifted_costs;
    }
  }

  /// One dual simplex pivot on the infeasible basic variable at position p (Bland).
  void dual_step(Sz p) {
    count_step();
    const auto leaving = static_cast<Sz>(basis_.basic[p]);
    const bool must_rise = below(problem_, leaving, basis_.value[leaving]);
    const Nb to = must_rise ? Nb::kLower : Nb::kUpper;
    Index entering = -1;
    Rational best(0);
    for (const auto& [k, a] : basis_.tableau_row(problem_, p)) {
      if (problem_.fixed(static_cast<Index>(k))) continue;
      const Nb st = basis_.status[k];
      // x_p = ... - a x_k: raising x_k moves x_p by -a, lowering it by +a.
      const bool can_rise = st == Nb::kLower || st == Nb::kFree;
      const bool can_fall = st == Nb::kUpper || st == Nb::kFree;
      const bool helps = (can_rise && ((a.sign() < 0) == must_rise)) ||
                         (can_fall && ((a.sign() > 0) == must_rise));
      if (!helps) continue;
      Rational ratio = basis_.d[k] / a;
      if (ratio.sign() < 0) ratio = -ratio;
      if (entering < 0 || ratio < best) {  // ties keep the smaller index: Bland
        best = std::move(ratio);
        entering = static_cast<Index>(k);
      }
    }
    if (entering < 0) {
      // The claim is checked on its own terms before it is made: row p of B^{-1} [A | -I]
      // is an exact consequence of the rows, x_p + sum_k alpha_k x_k = 0, and over every
      // variable's own bounds its left side must be unable to reach 0 (a Farkas proof).
      if (!row_excludes_zero(p)) {
        throw Declined{fmt::format(
            "the dual simplex found no entering variable for variable {}, but the tableau "
            "row does not prove infeasibility",
            leaving)};
      }
      // The multipliers themselves, e_p^T B^{-1} by row, go to the .sol so that
      // tools/verify_solution.py can re-derive the proof with its own arithmetic.
      std::vector<Rational> w;
      (void)basis_.tableau_row(problem_, p, &w);
      for (Sz i = 0; i < w.size(); ++i) {
        if (!w[i].is_zero())
          result_.farkas_row.emplace_back(static_cast<Index>(i), w[i].to_string());
      }
      throw Failed{fmt::format(
          "the model as read into doubles is exactly infeasible: the tableau row of variable "
          "{} cannot reach 0 over the bounds (a Farkas proof, checked)",
          leaving)};
    }
    const std::vector<Rational> alpha = basis_.tableau_column(problem_, entering);
    if (!basis_.pivot(problem_, p, entering, to, alpha, deadline_)) {
      throw Declined{"a dual simplex pivot met an exactly singular basis"};
    }
    ++result_.dual_pivots;
  }

  /// Whether x_p + sum over nonbasic k of alpha_pk x_k, which the rows force to be 0, has a
  /// range over the variables' bounds that excludes 0: the least and greatest values of each
  /// term are taken at its bounds, an infinite bound with a nonzero coefficient leaving that
  /// side unbounded.
  [[nodiscard]] bool row_excludes_zero(Sz p) const {
    std::vector<std::pair<Sz, Rational>> terms = basis_.tableau_row(problem_, p);
    terms.emplace_back(static_cast<Sz>(basis_.basic[p]), Rational(1));
    Rational least(0);
    Rational greatest(0);
    bool least_finite = true;
    bool greatest_finite = true;
    for (const auto& [k, a] : terms) {
      const bool up = a.sign() > 0;
      // The term a x_k is least at x_k's lower bound when a > 0, at its upper when a < 0.
      if (up ? problem_.has_lower[k] : problem_.has_upper[k]) {
        least += a * (up ? problem_.lower[k] : problem_.upper[k]);
      } else {
        least_finite = false;
      }
      if (up ? problem_.has_upper[k] : problem_.has_lower[k]) {
        greatest += a * (up ? problem_.upper[k] : problem_.lower[k]);
      } else {
        greatest_finite = false;
      }
    }
    return (least_finite && least.sign() > 0) || (greatest_finite && greatest.sign() < 0);
  }

  /// One primal simplex step on the dual infeasible nonbasic variable q (Bland): a pivot, or
  /// a bound flip when q reaches its own other bound first.
  void primal_step(Sz q) {
    count_step();
    const Nb st = basis_.status[q];
    const int direction = st == Nb::kLower ? 1 : st == Nb::kUpper ? -1 : -basis_.d[q].sign();
    const std::vector<Rational> alpha = basis_.tableau_column(problem_, static_cast<Index>(q));
    std::optional<Rational> best;
    Sz leave_p = alpha.size();
    Index leave_k = -1;
    Nb leave_to = Nb::kLower;
    for (Sz p = 0; p < alpha.size(); ++p) {
      if (alpha[p].is_zero()) continue;
      const Index k = basis_.basic[p];
      const auto u = static_cast<Sz>(k);
      // x_k moves at rate -direction * alpha_p.
      const int rate = -direction * alpha[p].sign();
      const Rational size = alpha[p].sign() < 0 ? -alpha[p] : alpha[p];
      std::optional<Rational> ratio;
      Nb to = Nb::kLower;
      if (rate < 0 && problem_.has_lower[u]) {
        ratio = (basis_.value[u] - problem_.lower[u]) / size;
      } else if (rate > 0 && problem_.has_upper[u]) {
        ratio = (problem_.upper[u] - basis_.value[u]) / size;
        to = Nb::kUpper;
      }
      if (!ratio.has_value()) continue;
      if (!best.has_value() || *ratio < *best || (*ratio == *best && k < leave_k)) {
        best = std::move(ratio);
        leave_p = p;
        leave_k = k;
        leave_to = to;
      }
    }
    const bool boxed = problem_.has_lower[q] && problem_.has_upper[q] && st != Nb::kFree;
    if (boxed) {
      const Rational room = problem_.upper[q] - problem_.lower[q];
      if (!best.has_value() || room < *best) {
        basis_.status[q] = st == Nb::kLower ? Nb::kUpper : Nb::kLower;
        basis_.refresh(problem_);
        ++result_.bound_flips;
        return;
      }
    }
    if (!best.has_value()) {
      throw Failed{fmt::format(
          "variable {} improves the objective without limit: the model is exactly unbounded",
          q)};
    }
    if (!basis_.pivot(problem_, leave_p, static_cast<Index>(q), leave_to, alpha, deadline_)) {
      throw Declined{"a primal simplex pivot met an exactly singular basis"};
    }
    ++result_.primal_pivots;
  }

  Problem& problem_;
  Basis& basis_;
  RepairResult& result_;
  const Deadline& deadline_;
};

BasisStatus status_of(const Problem& problem, Nb nb, Sz k) {
  switch (nb) {
    case Nb::kBasic: return BasisStatus::kBasic;
    case Nb::kFree: return BasisStatus::kNonbasicFree;
    case Nb::kLower:
    case Nb::kUpper:
      if (problem.fixed(static_cast<Index>(k))) return BasisStatus::kFixed;
      return nb == Nb::kLower ? BasisStatus::kAtLower : BasisStatus::kAtUpper;
  }
  return BasisStatus::kUnknown;
}

}  // namespace

RepairResult repair_basis_exact(const Model& model, const Solution& solution, double seconds) {
  RepairResult result;
  const Deadline deadline(seconds);
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (solution.status != SolveStatus::kOptimal) {
    result.message = "the status is not optimal";
    return result;
  }
  if (model.hessian.num_nonzeros() > 0 || model.num_integer_columns() > 0) {
    result.message = "the exact repair covers plain LPs only";
    return result;
  }
  if (static_cast<Index>(solution.col_status.size()) != n ||
      static_cast<Index>(solution.row_status.size()) != m) {
    result.message = "the engine produced no basis";
    return result;
  }
  try {
    Problem problem(model);
    Basis basis;
    const LoadResult loaded = load_basis(model, solution, problem, deadline, &basis);
    if (!loaded.message.empty()) {
      result.verdict = loaded.verdict;
      result.message = loaded.message;
      return result;
    }
    Repair(&problem, &basis, &result, deadline).run();
    const std::string why = not_optimal(problem, basis);
    if (!why.empty()) {
      result.message = "the repair stopped short of an exactly optimal basis: " + why;
      return result;
    }
    result.verdict = ExactVerdict::kVerified;
    const auto sn = static_cast<Sz>(n);
    const auto sm = static_cast<Sz>(m);
    result.col_status.resize(sn);
    result.row_status.resize(sm);
    result.col_value.resize(sn);
    result.col_dual.resize(sn);
    result.row_dual.resize(sm);
    const double sense = problem.sense;
    for (Sz j = 0; j < sn; ++j) {
      result.col_status[j] = status_of(problem, basis.status[j], j);
      result.col_value[j] = basis.value[j].to_double();
      result.col_dual[j] = basis.status[j] == Nb::kBasic ? 0.0 : sense * basis.d[j].to_double();
    }
    for (Sz i = 0; i < sm; ++i) {
      result.row_status[i] = status_of(problem, basis.status[sn + i], sn + i);
      result.row_dual[i] = sense * basis.y[i].to_double();
    }
    return result;
  } catch (const Declined& declined) {
    result.verdict = ExactVerdict::kDeclined;
    result.message = declined.why;
  } catch (const Failed& failed) {
    result.verdict = ExactVerdict::kFailed;
    result.message = failed.why;
  } catch (const ExactBudgetExceeded&) {
    result.verdict = ExactVerdict::kDeclined;
    result.message = fmt::format(
        "the exact repair ran past its {} s budget (option exact_seconds)", seconds);
  } catch (const RationalOverflow&) {
    result.verdict = ExactVerdict::kDeclined;
    result.message = "no exact rational result: a non-finite value in the data";
  }
  result.dual_pivots = result.primal_pivots = result.bound_flips = result.shifted_costs = 0;
  return result;
}

bool apply_exact_repair(const Model& model, Solution* solution, Logger& logger,
                        double seconds) {
  RepairResult repair = repair_basis_exact(model, *solution, seconds);
  solution->exact_repair_status =
      repair.verdict == ExactVerdict::kVerified   ? Solution::ExactVerification::kVerified
      : repair.verdict == ExactVerdict::kDeclined ? Solution::ExactVerification::kDeclined
                                                  : Solution::ExactVerification::kFailed;
  solution->exact_repair_message = repair.message;
  solution->exact_repair_pivots = repair.dual_pivots + repair.primal_pivots;
  solution->exact_repair_flips = repair.bound_flips;
  solution->exact_repair_farkas = std::move(repair.farkas_row);
  if (repair.verdict != ExactVerdict::kVerified) {
    if (repair.verdict == ExactVerdict::kFailed) {
      logger.warning("Exact repair FAILED: {}", repair.message);
    } else {
      logger.info("Exact repair declined: {}", repair.message);
    }
    return false;
  }
  if (!repair.changed()) {
    logger.info("Exact repair: the reported basis is already exactly optimal");
    return false;
  }
  logger.info(
      "Exact repair (#757): {} dual and {} primal simplex pivot(s) and {} bound flip(s) in "
      "exact arithmetic ({} cost(s) shifted and restored) reached an exactly optimal basis; "
      "it and its exact point and duals, rounded once to double, replace the reported ones",
      repair.dual_pivots, repair.primal_pivots, repair.bound_flips, repair.shifted_costs);
  solution->col_status = std::move(repair.col_status);
  solution->row_status = std::move(repair.row_status);
  solution->col_value = std::move(repair.col_value);
  solution->col_dual = std::move(repair.col_dual);
  solution->row_dual = std::move(repair.row_dual);
  solution->recompute_quality(model);
  return true;
}

}  // namespace sankhya::exact
