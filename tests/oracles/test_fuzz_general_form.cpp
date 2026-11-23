// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the float solver against the exact oracle on general-form models.
//
// Every other oracle sweep hands the solver the oracle's own narrow form: min c'x,
// A x >= b, 0 <= x <= u, small integer data. These sweeps hand it what a user writes
// (rich_model.hpp): rows of every kind, free and reflected and fixed columns, integer
// columns with fractional bounds and wide ranges, fractional coefficients, big-M links, a
// constant in the objective, both senses. The exact answer comes from the same instance
// rewritten onto the oracle's form.
//
// What is compared, on every instance:
//   * the status: optimal, infeasible or unbounded, exactly as the oracle has it;
//   * the objective, to 1e-6 relative (a MILP that stopped on its gap target, to that gap);
//   * the dual bound, which may not pass the exact optimum;
//   * the returned point, recomputed against the ORIGINAL model: bounds, rows and
//     integrality at the solver's own tolerances, and the objective it is worth;
//   * for an LP, the returned duals, as a certificate: the Lagrangian bound they give on
//     the original model must equal the exact optimum, which is strong duality and holds
//     for an optimal dual vector and for no other (Chvatal 1983, ch. 5). That is the
//     presolve and postsolve round trip checked from outside, duals included.
//
// A mismatch is reduced (shrink) before it is printed, so the log holds a small model.
//
// The pull-request run is modest. SANKHYA_FUZZ_SCALE multiplies every trial count and
// SANKHYA_FUZZ_SEED is added to every seed; .github/workflows/fuzz-wide.yml runs the suite
// that way, on seeds no earlier run used.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "oracles/rich_model.hpp"

namespace sankhya::oracle {
namespace {

using OptionList = std::vector<std::pair<std::string, std::string>>;

std::int64_t from_environment(const char* name, std::int64_t fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::atoll(value);
}

int scaled(int count) {
  return count *
         static_cast<int>(std::max<std::int64_t>(1, from_environment("SANKHYA_FUZZ_SCALE", 1)));
}

std::mt19937_64 seeded(std::uint64_t seed) {
  return std::mt19937_64(seed +
                         static_cast<std::uint64_t>(from_environment("SANKHYA_FUZZ_SEED", 0)));
}

std::string text(const OptionList& options) {
  if (options.empty()) return "defaults";
  std::string out;
  for (const auto& [key, value] : options) out += (out.empty() ? "" : " ") + key + "=" + value;
  return out;
}

// =========================================================================================
// The returned point and the returned duals, against the original model
// =========================================================================================

/// Empty when the solution's point satisfies the model at the solver's tolerances and is
/// worth the objective reported. The row test is the verifier's (tools/verify_solution.py):
/// a violation is measured against the largest term of its row.
std::string point_violation(const Model& model, const Solution& s) {
  const auto cols = static_cast<std::size_t>(model.num_cols());
  const auto rows = static_cast<std::size_t>(model.num_rows());
  if (s.col_value.size() != cols) return "the solution carries no point";
  std::vector<double> activity(rows, 0.0);
  std::vector<double> row_scale(rows, 1.0);
  double objective = model.objective_offset;
  for (std::size_t j = 0; j < cols; ++j) {
    const double x = s.col_value[j];
    const std::string name = "column " + std::to_string(j) + " = " + std::to_string(x);
    if (!std::isfinite(x)) return name + " is not a number";
    const double room = tol::kPrimalFeasibility * std::max(1.0, std::fabs(x));
    if (x < model.col_lower[j] - room || x > model.col_upper[j] + room) {
      return name + " is outside its bounds";
    }
    if (model.col_type[j] == VarType::kInteger &&
        std::fabs(x - std::round(x)) > tol::kIntegrality) {
      return name + " is not integral";
    }
    objective += model.col_cost[j] * x;
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      const auto i = static_cast<std::size_t>(column.rows[k]);
      const double term = column.values[k] * x;
      activity[i] += term;
      row_scale[i] = std::max(row_scale[i], std::fabs(term));
    }
  }
  for (std::size_t i = 0; i < rows; ++i) {
    const double room = tol::kPrimalFeasibility * row_scale[i];
    if (activity[i] < model.row_lower[i] - room || activity[i] > model.row_upper[i] + room) {
      return "row " + std::to_string(i) + " has activity " + std::to_string(activity[i]) +
             ", outside its bounds";
    }
  }
  if (std::fabs(objective - s.objective) > 1e-6 * std::max(1.0, std::fabs(objective))) {
    return "the point is worth " + std::to_string(objective) + ", not the reported objective";
  }
  return "";
}

/// Empty when the solution's row duals certify `expected` on the original model. In
/// minimization space, with d = c - A'y, every feasible x has
///     c'x  =  y'(Ax) + d'x  >=  sum_i min(y_i L_i, y_i U_i) + sum_j min(d_j l_j, d_j u_j),
/// and the right-hand side reaches the optimum exactly when y is an optimal dual vector. A
/// multiplier pointing at a side with no bound makes the bound minus infinity; one within
/// the dual feasibility tolerance is taken as zero there, which is the solver's own promise.
/// The reduced costs are recomputed here; the reported ones only have to agree with them.
std::string dual_violation(const Model& model, const Solution& s, double expected) {
  const auto cols = static_cast<std::size_t>(model.num_cols());
  const auto rows = static_cast<std::size_t>(model.num_rows());
  const double sigma = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  double cost_scale = 1.0;
  for (const double c : model.col_cost) cost_scale = std::max(cost_scale, std::fabs(c));
  const double negligible = tol::kDualFeasibility * cost_scale;

  double bound = 0.0;
  for (std::size_t i = 0; i < rows; ++i) {
    const double y = sigma * s.row_dual[i];
    const double side = y > 0.0 ? model.row_lower[i] : model.row_upper[i];
    if (std::isfinite(side)) {
      bound += y * side;
    } else if (!(std::fabs(y) <= negligible)) {
      return "row " + std::to_string(i) + " has multiplier " + std::to_string(s.row_dual[i]) +
             " toward a side it has no bound on";
    }
  }
  for (std::size_t j = 0; j < cols; ++j) {
    double d = sigma * model.col_cost[j];
    double term_scale = cost_scale;
    const ColumnView column = model.matrix.column(static_cast<Index>(j));
    for (Index k = 0; k < column.size; ++k) {
      const double term =
          column.values[k] * sigma * s.row_dual[static_cast<std::size_t>(column.rows[k])];
      d -= term;
      term_scale = std::max(term_scale, std::fabs(term));
    }
    if (!(std::fabs(sigma * s.col_dual[j] - d) <= 1e-6 * term_scale)) {
      return "column " + std::to_string(j) + " reports reduced cost " +
             std::to_string(s.col_dual[j]) + ", but c - A'y is " + std::to_string(sigma * d);
    }
    const double side = d > 0.0 ? model.col_lower[j] : model.col_upper[j];
    if (std::isfinite(side)) {
      bound += d * side;
    } else if (!(std::fabs(d) <= negligible)) {
      return "column " + std::to_string(j) + " has reduced cost " + std::to_string(sigma * d) +
             " toward a side it has no bound on";
    }
  }
  const double certified = sigma * bound + model.objective_offset;
  if (!(std::fabs(certified - expected) <= 1e-6 * std::max(1.0, std::fabs(expected)))) {
    return "the duals certify " + std::to_string(certified) + ", not the optimum";
  }
  return "";
}

// =========================================================================================
// One instance
// =========================================================================================

enum class Outcome { kAgreed, kAbstained, kMismatched };

struct Verdict {
  Outcome outcome = Outcome::kAgreed;
  OracleStatus exact = OracleStatus::kOptimal;
  bool duals_checked = false;
  std::string why;
};

/// `certifies` is false for an engine that cannot tell infeasible from unbounded (the plain
/// interior point): on an instance without an optimum it then only has to claim no point.
Verdict judge(const RichModel& rich, const OptionList& extra, bool certifies = true) {
  Verdict v;
  // The exact tree copies its bounds per node and solves each node from scratch, so a wide
  // general-integer model can take it thousands of nodes; past this it abstains.
  const RichExact exact = solve_exact(rich, 1000);
  v.exact = exact.status;
  if (exact.status == OracleStatus::kOverflow ||
      exact.status == OracleStatus::kIterationLimit) {
    v.outcome = Outcome::kAbstained;
    return v;
  }
  const auto mismatch = [&v](const std::string& why) {
    v.outcome = Outcome::kMismatched;
    v.why = why;
    return v;
  };
  if (exact.status == OracleStatus::kOptimal) {
    const std::string broken = exact_violation(rich, exact.x, exact.objective);
    if (!broken.empty()) return mismatch("HARNESS: the exact point does not fit: " + broken);
  }
  const Model model = to_model(rich);
  if (const std::string invalid = model.validate(); !invalid.empty()) {
    return mismatch("HARNESS: " + invalid);
  }
  Options options;
  options.set_bool("log_to_console", false);
  // No model here needs a second; a search still running after a minute is a failure the
  // status then shows, not a job that times out without saying which model.
  options.set_double("time_limit", 60.0);
  std::string error;
  for (const auto& [key, value] : extra) {
    if (!options.set_from_string(key, value, &error)) return mismatch("HARNESS: " + error);
  }
  const Solution s = solve(model, options);
  const std::string solver = std::string("solver: ") + to_string(s.status) + ", objective " +
                             std::to_string(s.objective) + ", bound " +
                             std::to_string(s.dual_bound) + " (" + s.message + ")";

  if (exact.status != OracleStatus::kOptimal) {
    const SolveStatus wanted = exact.status == OracleStatus::kInfeasible
                                   ? SolveStatus::kInfeasible
                                   : SolveStatus::kUnbounded;
    const bool claims_point =
        s.status == SolveStatus::kOptimal || s.status == SolveStatus::kFeasible;
    if (certifies ? s.status != wanted : claims_point) {
      return mismatch(std::string("exactly ") + to_string(exact.status) + "; " + solver);
    }
    return v;
  }

  const double expected = exact.objective.to_double();
  const double scale = std::max(1.0, std::fabs(expected));
  const std::string exactly =
      "the exact optimum is " + std::to_string(expected) + "; " + solver;
  if (s.status != SolveStatus::kOptimal) return mismatch(exactly);
  // A MILP that stops on its gap target is optimal within that target (#188) and says so.
  double allowed = 1e-6 * scale;
  if (rich.has_integers() && s.message.find("gap target") != std::string::npos) {
    allowed += std::max(options.get_double("mip_absolute_gap"),
                        options.get_double("mip_relative_gap") * scale);
  }
  if (!(std::fabs(s.objective - expected) <= allowed)) return mismatch(exactly);
  const double sense = rich.maximize ? -1.0 : 1.0;
  if (!(sense * (s.dual_bound - expected) <= 1e-6 * scale)) {
    return mismatch("the dual bound passes the optimum: " + exactly);
  }
  if (const std::string bad = point_violation(model, s); !bad.empty()) {
    return mismatch(bad + "; " + exactly);
  }
  if (!rich.has_integers() && s.row_dual.size() == static_cast<std::size_t>(model.num_rows()) &&
      s.col_dual.size() == static_cast<std::size_t>(model.num_cols())) {
    v.duals_checked = true;
    if (const std::string bad = dual_violation(model, s, expected); !bad.empty()) {
      return mismatch(bad + "; " + exactly);
    }
  }
  return v;
}

// =========================================================================================
// A sweep
// =========================================================================================

struct Family {
  const char* name;
  RichConfig config;
};

/// The options one instance is solved under, and whether that engine certifies.
struct Choice {
  OptionList options;
  bool certifies = true;
};

struct Tally {
  int optimal = 0;
  int infeasible = 0;
  int unbounded = 0;
  int abstained = 0;
  int mismatched = 0;
  int duals_checked = 0;
};

Tally sweep(const std::string& title, std::uint64_t seed, int count,
            const std::vector<Family>& families,
            const std::function<Choice(std::mt19937_64&)>& choose) {
  std::mt19937_64 rng = seeded(seed);
  Tally tally;
  int printed = 0;
  for (int trial = 0; trial < count; ++trial) {
    const Family& family = families[static_cast<std::size_t>(trial) % families.size()];
    const RichModel rich = random_rich(rng, family.config);
    const Choice choice = choose(rng);
    const Verdict v = judge(rich, choice.options, choice.certifies);
    tally.duals_checked += v.duals_checked ? 1 : 0;
    if (v.outcome == Outcome::kAbstained) {
      ++tally.abstained;
      continue;
    }
    if (v.outcome == Outcome::kAgreed) {
      tally.optimal += v.exact == OracleStatus::kOptimal ? 1 : 0;
      tally.infeasible += v.exact == OracleStatus::kInfeasible ? 1 : 0;
      tally.unbounded += v.exact == OracleStatus::kUnbounded ? 1 : 0;
      continue;
    }
    ++tally.mismatched;
    if (printed++ >= 3) continue;
    // Reduced under the same options, to a model that still fails for a reason of the
    // solver's (a harness failure is printed as drawn).
    const bool harness = v.why.rfind("HARNESS", 0) == 0;
    const RichModel small =
        harness ? rich : shrink(rich, [&choice](const RichModel& candidate) {
          const Verdict again = judge(candidate, choice.options, choice.certifies);
          return again.outcome == Outcome::kMismatched && again.why.rfind("HARNESS", 0) != 0;
        });
    std::cout << "\n--- MISMATCH: " << title << ", family " << family.name << ", trial "
              << trial << ", options: " << text(choice.options) << " ---\n"
              << v.why << "\nreduced to:\n"
              << small.to_text() << "where "
              << judge(small, choice.options, choice.certifies).why << "\nas drawn:\n"
              << rich.to_text();
  }
  std::cout << "\n=== " << title << " (seed " << seed << " + SANKHYA_FUZZ_SEED, " << count
            << " models) ===\n"
            << "  agreed optimal      " << tally.optimal << "\n"
            << "  agreed infeasible   " << tally.infeasible << "\n"
            << "  agreed unbounded    " << tally.unbounded << "\n"
            << "  oracle abstained    " << tally.abstained << "\n"
            << "  duals certified     " << tally.duals_checked << "\n"
            << "  MISMATCHED          " << tally.mismatched << "\n";
  return tally;
}

Choice defaults(std::mt19937_64& /*rng*/) {
  return {};
}

/// One or two entries of `pool`, merged; an entry that sets a key already set is skipped.
Choice some_of(const std::vector<Choice>& pool, std::mt19937_64& rng) {
  std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
  Choice chosen = pool[pick(rng)];
  if (rng() % 2 == 0) {
    const Choice& second = pool[pick(rng)];
    bool clash = false;
    for (const auto& added : second.options) {
      for (const auto& taken : chosen.options) clash = clash || added.first == taken.first;
    }
    if (!clash) {
      chosen.options.insert(chosen.options.end(), second.options.begin(), second.options.end());
      chosen.certifies = chosen.certifies && second.certifies;
    }
  }
  return chosen;
}

// =========================================================================================
// Linear programs
// =========================================================================================

std::vector<Family> lp_families() {
  RichConfig general;
  general.dyadic = 0.2;
  RichConfig equalities = general;
  equalities.equality_rows = 0.7;
  equalities.ranged_rows = 0.1;
  equalities.upper_rows = 0.1;
  equalities.planted = 0.9;
  RichConfig fractions = general;
  fractions.dyadic = 0.4;
  fractions.decimal = 0.3;
  fractions.fractional_bounds = 0.5;
  RichConfig wide = general;
  wide.shift = 1000;
  wide.span = 2000;
  wide.magnitude = 9;
  // Larger and sparse: singleton rows, column singletons and doubleton equations, which is
  // what presolve's reductions and their postsolve records are made of.
  RichConfig sparse = general;
  sparse.min_rows = 4;
  sparse.max_rows = 9;
  sparse.min_cols = 5;
  sparse.max_cols = 10;
  sparse.density = 0.25;
  sparse.equality_rows = 0.4;
  RichConfig boxed = general;  // every column a box: never unbounded, mostly optimal
  boxed.free_columns = 0.0;
  boxed.lower_only_columns = 0.0;
  boxed.upper_only_columns = 0.0;
  return {{"general", general},      {"equality-heavy", equalities},
          {"fractional", fractions}, {"wide boxes", wide},
          {"sparse", sparse},        {"boxed", boxed}};
}

const std::vector<Choice>& lp_switches() {
  static const std::vector<Choice> pool = {
      {{{"presolve", "false"}}},
      {{{"scaling", "false"}}},
      {{{"algorithm", "simplex"}}},
      {{{"algorithm", "simplex"}, {"ratio_test", "harris"}}},
      {{{"algorithm", "dual-simplex"}}},
      {{{"algorithm", "ipm"}}, false},
      {{{"pricing", "devex"}}},
      {{{"pricing", "dantzig"}}},
      {{{"basis_update", "forrest-tomlin"}}},
      {{{"lu_hyper_sparse", "false"}}},
      {{{"dual_ratio_test", "harris"}}},
      {{{"dual_ratio_test", "textbook"}}},
      {{{"dual_perturb_costs_at_start", "true"}}},
      {{{"presolve_dual_reductions", "false"}}},
      {{{"presolve_substitution", "false"}}},
      {{{"presolve_dual_fixing", "true"}}},
      {{{"presolve_parallel_rows", "true"}}},
      {{{"presolve_dominated_columns", "true"}}},
      {{{"presolve_implied_free", "true"}}},
      {{{"presolve_bound_propagation", "true"}}},
      {{{"algorithm_selection", "learned"}}},
      {{{"engine_race", "true"}, {"deterministic", "true"}}},
      {{{"prescale_retry", "false"}}},
      {{{"exact", "true"}}},
      {{{"ranging", "true"}}},
  };
  return pool;
}

TEST(FuzzGeneralForm, LinearProgramsUnderTheDefaults) {
  const int count = scaled(2000);
  const Tally tally = sweep("LPs, defaults", 20261123, count, lp_families(), defaults);
  EXPECT_EQ(tally.mismatched, 0);
  EXPECT_GT(tally.optimal, count / 6) << "too few optimal instances to say anything of duals";
  EXPECT_GT(tally.infeasible, count / 50);
  EXPECT_GT(tally.unbounded, count / 50);
  EXPECT_EQ(tally.duals_checked, tally.optimal) << "an optimal LP came back without duals";
}

TEST(FuzzGeneralForm, LinearProgramsUnderEveryEngineAndPresolveSwitch) {
  const int count = scaled(2000);
  const Tally tally =
      sweep("LPs, one or two switches off their defaults", 20261124, count, lp_families(),
            [](std::mt19937_64& rng) { return some_of(lp_switches(), rng); });
  EXPECT_EQ(tally.mismatched, 0);
  EXPECT_GT(tally.optimal, count / 6);
  EXPECT_GT(tally.duals_checked, count / 6);
}

// =========================================================================================
// Mixed-integer programs
// =========================================================================================

std::vector<Family> milp_families() {
  RichConfig mixed;
  mixed.integer_columns = 0.5;
  mixed.binary_columns = 0.2;
  mixed.dyadic = 0.15;
  mixed.free_columns = 0.05;
  mixed.lower_only_columns = 0.1;
  mixed.upper_only_columns = 0.05;
  RichConfig equalities = mixed;
  equalities.equality_rows = 0.6;
  equalities.ranged_rows = 0.1;
  equalities.upper_rows = 0.15;
  equalities.planted = 0.9;
  // General integers over wide ranges with fractional coefficients and fractional bounds.
  RichConfig general_integers = mixed;
  general_integers.integer_columns = 0.8;
  general_integers.binary_columns = 0.0;
  general_integers.dyadic = 0.4;
  general_integers.fractional_bounds = 0.5;
  general_integers.shift = 60;
  general_integers.span = 120;
  general_integers.max_cols = 5;
  RichConfig big_m = mixed;
  big_m.integer_columns = 0.3;
  big_m.max_cols = 4;
  big_m.big_m_links = 2;
  RichConfig sparse = mixed;
  sparse.min_rows = 3;
  sparse.max_rows = 7;
  sparse.min_cols = 5;
  sparse.max_cols = 8;
  sparse.density = 0.3;
  sparse.equality_rows = 0.35;
  RichConfig boxed = mixed;  // every column a box: the tree is finite and mostly feasible
  boxed.free_columns = 0.0;
  boxed.lower_only_columns = 0.0;
  boxed.upper_only_columns = 0.0;
  boxed.decimal = 0.2;
  return {{"mixed", mixed},
          {"equality-heavy", equalities},
          {"general integers", general_integers},
          {"big-M", big_m},
          {"sparse", sparse},
          {"boxed", boxed}};
}

const std::vector<Choice>& milp_switches() {
  static const std::vector<Choice> pool = {
      {{{"presolve", "false"}}},
      {{{"presolve_dual_fixing", "true"}}},
      {{{"presolve_parallel_rows", "true"}}},
      {{{"presolve_probing", "true"}}},
      {{{"presolve_probing", "true"}, {"presolve_probing_continuous_bounds", "true"}}},
      {{{"presolve_probing_fixed_share", "0"}}},
      {{{"presolve_dominated_columns", "true"}}},
      {{{"presolve_implied_free", "true"}}},
      {{{"presolve_implied_integer", "true"}}},
      {{{"presolve_bound_propagation", "true"}}},
      {{{"presolve_coefficient_tightening", "true"}}},
      {{{"scaling", "false"}}},
      {{{"mip_branching", "most-fractional"}}},
      {{{"mip_learned_branching", "true"}}},
      {{{"mip_node_selection", "best-bound"}}},
      {{{"mip_node_selection", "depth-first"}}},
      {{{"mip_node_selection", "best-estimate"}}},
      {{{"mip_objective_integrality", "false"}}},
      {{{"mip_node_engine", "primal"}}},
      {{{"mip_node_pricing", "dantzig"}}},
      {{{"mip_node_pricing", "dual-steepest-edge"}}},
      {{{"basis_update", "forrest-tomlin"}}},
      {{{"dual_ratio_test", "harris"}}},
      {{{"mip_cut_pooling", "false"}}},
      {{{"tree_cut_depth", "4"}}},
      {{{"mir_cmir", "false"}}},
      {{{"gmi_safety", "true"}}},
      {{{"root_cut_loop", "false"}}},
      {{{"root_cut_loop_when", "always"}}},
      {{{"enable_root_cuts", "false"}}},
      {{{"enable_clique_cuts", "false"}}},
      {{{"enable_zero_half_cuts", "false"}}},
      {{{"enable_mir_cuts", "false"}}},
      {{{"enable_flow_cover_cuts", "false"}}},
      {{{"flow_cover_lifting", "false"}}},
      {{{"cut_efficacy_test", "true"}}},
      {{{"mip_implied_bound_cuts", "true"}}},
      {{{"mip_heuristics", "true"}}},
      {{{"mip_heuristics", "false"}}},
      {{{"mip_heur_fj", "off"}}},
      {{{"mip_heur_parity", "off"}}},
      {{{"mip_local_mip", "true"}}},
      {{{"mip_dive_backtrack", "true"}}},
      {{{"mip_reduced_cost_fixing", "true"}}},
      {{{"mip_restarts", "2"}, {"mip_restart_fraction", "0"}}},
      {{{"mip_node_factor_cache", "0"}}},
      {{{"mip_objective_branching", "true"}}},
      {{{"safe_bounds", "true"}}},
      {{{"certified_gap", "false"}}},
      {{{"mip_strong_branch_fix", "true"}}},
      {{{"mip_incremental_propagation", "true"}}},
      {{{"mip_heap_open_list", "true"}}},
      {{{"mip_symmetry", "false"}}},
      {{{"mip_obbt", "true"}}},
      {{{"conflict_analysis", "true"}}},
      {{{"conflict_analysis", "true"}, {"conflict_cutoff", "true"}}},
      {{{"conflict_analysis", "true"}, {"conflict_use", "prune"}}},
      {{{"mip_threads", "1"}}},
      {{{"deterministic", "true"}}},
      {{{"mip_relative_gap", "0"}, {"mip_absolute_gap", "0"}}},
  };
  return pool;
}

TEST(FuzzGeneralForm, MixedIntegerProgramsUnderTheDefaults) {
  const int count = scaled(300);
  const Tally tally = sweep("MILPs, defaults", 20261125, count, milp_families(), defaults);
  EXPECT_EQ(tally.mismatched, 0);
  EXPECT_GT(tally.optimal, count / 6);
  EXPECT_GT(tally.infeasible, count / 50);
}

TEST(FuzzGeneralForm, MixedIntegerProgramsOnOneThread) {
  // The sequential search is the one restarts, objective branching and the deterministic
  // mode run in, and the default is four workers wherever there are four cores.
  const int count = scaled(300);
  const Tally tally = sweep("MILPs, mip_threads=1", 20261126, count, milp_families(),
                            [](std::mt19937_64&) { return Choice{{{"mip_threads", "1"}}}; });
  EXPECT_EQ(tally.mismatched, 0);
  EXPECT_GT(tally.optimal, count / 6);
}

TEST(FuzzGeneralForm, MixedIntegerProgramsUnderEverySearchAndPresolveSwitch) {
  const int count = scaled(600);
  const Tally tally =
      sweep("MILPs, one or two switches off their defaults", 20261127, count, milp_families(),
            [](std::mt19937_64& rng) { return some_of(milp_switches(), rng); });
  EXPECT_EQ(tally.mismatched, 0);
  EXPECT_GT(tally.optimal, count / 6);
}

}  // namespace
}  // namespace sankhya::oracle
