// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the PDHG feasibility pump and fix-and-propagate (#509).
//
// What is held here, each independently of the code under test:
//   - on small MILPs that have integer points, each heuristic finds one, on binaries, on
//     general integers (the auxiliary-column distance) and on a mixed model (the completion
//     LP), and the point passes this file's own check;
//   - over random MILPs, every point either returns passes that check, and on a MILP the
//     exact rational oracle proves infeasible neither returns anything;
//   - the search with both on agrees with the exact oracle's optimum;
//   - off by default, and the default path is bit for bit the path with both set off;
//   - the CUDA variants run the same checks on the device, and skip - not pass - without one.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "mip/pdhg_heuristics.hpp"
#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#endif
#ifdef SANKHYA_HAVE_OPENMP
#include <omp.h>
#endif

namespace sankhya::mip {
namespace {

/// One OpenMP thread for the test, restored after. A PDHG iteration on these tiny models is a
/// parallel region over a handful of columns (SparseMatrix::transpose_multiply_add), and on a
/// many-core box the region's overhead is thousands of times the arithmetic: at the runtime's
/// default of one thread per core the first run of this file took minutes on the 256-core
/// test box. The products are gathers whose result does not depend on the thread count (#57),
/// so this changes the time only.
class OneThread {
 public:
  // User-provided in both builds, so a guard is never an "unused variable" without OpenMP.
  OneThread() {
#ifdef SANKHYA_HAVE_OPENMP
    saved_ = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
  }
  ~OneThread() {
#ifdef SANKHYA_HAVE_OPENMP
    omp_set_num_threads(saved_);
#endif
  }
  OneThread(const OneThread&) = delete;
  OneThread& operator=(const OneThread&) = delete;

 private:
  [[maybe_unused]] int saved_ = 0;
};

/// Rows given densely; every column in [lower, upper], integer where `integer` says so.
Model make_model(const std::vector<std::vector<double>>& rows,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                 const std::vector<double>& cost, const std::vector<double>& lower,
                 const std::vector<double>& upper, const std::vector<bool>& integer) {
  Model m;
  const auto n = cost.size();
  m.col_cost = cost;
  m.col_lower = lower;
  m.col_upper = upper;
  for (std::size_t j = 0; j < n; ++j) {
    m.col_type.push_back(integer[j] ? VarType::kInteger : VarType::kContinuous);
  }
  m.matrix.reset(static_cast<Index>(rows.size()), static_cast<Index>(n));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i][j] != 0.0) {
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), rows[i][j]);
      }
    }
  }
  m.matrix.finalize();
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

std::vector<Index> integers_of(const Model& m) {
  std::vector<Index> out;
  for (Index j = 0; j < m.num_cols(); ++j) {
    if (m.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) out.push_back(j);
  }
  return out;
}

/// Independent of the code under test: integral to `integrality` (exactly, by default: the
/// heuristics snap every integer column to its integer), inside the box to 1e-9, every row
/// satisfied to the absolute tolerance offer_incumbent() applies.
bool feasible(const Model& m, const std::vector<double>& x, std::string* why,
              double integrality = 0.0) {
  if (x.size() != static_cast<std::size_t>(m.num_cols())) {
    *why = "wrong length";
    return false;
  }
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (m.col_type[j] == VarType::kInteger &&
        std::fabs(x[j] - std::round(x[j])) > integrality) {
      *why = "column " + std::to_string(j) + " fractional";
      return false;
    }
    if (x[j] < m.col_lower[j] - 1e-9 || x[j] > m.col_upper[j] + 1e-9) {
      *why = "column " + std::to_string(j) + " outside its box";
      return false;
    }
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    double activity = 0.0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      activity += m.matrix.at(i, j) * x[static_cast<std::size_t>(j)];
    }
    const auto u = static_cast<std::size_t>(i);
    if (activity < m.row_lower[u] - 1e-7 || activity > m.row_upper[u] + 1e-7) {
      *why = "row " + std::to_string(i) + " violated, activity " + std::to_string(activity);
      return false;
    }
  }
  return true;
}

PdhgHeuristicSettings settings_for(bool device) {
  PdhgHeuristicSettings s;
  s.use_device = device;
  s.completion_options.set_bool("log_to_console", false);
  return s;
}

/// Set partitioning: n binaries summing to one, distinct costs. The relaxation of a pure
/// equality with equal coefficients is fractional for the PDHG start.
Model set_partition(int n) {
  std::vector<double> cost;
  for (int j = 0; j < n; ++j) cost.push_back(1.0 + 0.01 * j);
  return make_model({std::vector<double>(static_cast<std::size_t>(n), 1.0)}, {1.0}, {1.0}, cost,
                    std::vector<double>(static_cast<std::size_t>(n), 0.0),
                    std::vector<double>(static_cast<std::size_t>(n), 1.0),
                    std::vector<bool>(static_cast<std::size_t>(n), true));
}

/// General integers, one unbounded above: 3x + 5y + 7z = 41, x - y >= 1, z <= 4.
Model general_integers() {
  return make_model({{3, 5, 7}, {1, -1, 0}}, {41.0, 1.0}, {41.0, kInfinity}, {1.0, 2.0, 1.5},
                    {0.0, 0.0, 0.0}, {kInfinity, 10.0, 4.0}, {true, true, true});
}

/// Mixed: three binaries open capacity, the continuous flow z must reach 13 and stays under
/// the capacity opened, 6a + 5b + 4c >= z; each binary costs its capacity.
Model mixed_capacity() {
  return make_model({{6, 5, 4, -1}, {0, 0, 0, 1}}, {0.0, 13.0}, {kInfinity, kInfinity},
                    {6.0, 5.0, 4.0, 0.1}, {0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 20.0},
                    {true, true, true, false});
}

void expect_found(const Model& m, const PdhgHeuristicResult& r, const std::string& what) {
  ASSERT_FALSE(r.x.empty()) << what << ": nothing found (" << r.stopped << ")";
  std::string why;
  EXPECT_TRUE(feasible(m, r.x, &why)) << what << ": " << why;
  EXPECT_TRUE(point_is_feasible(m, integers_of(m), r.x, 1e-6)) << what;
}

void expect_feasible_if_any(const Model& m, const PdhgHeuristicResult& r,
                            const std::string& what) {
  if (r.x.empty()) return;
  std::string why;
  EXPECT_TRUE(feasible(m, r.x, &why)) << what << ": " << why;
}

void finds_points_on_small_milps(bool device) {
  const std::vector<std::pair<const char*, Model>> models = {
      {"set partition", set_partition(8)},
      {"general integers", general_integers()},
      {"mixed capacity", mixed_capacity()}};
  for (const auto& [name, m] : models) {
    const std::vector<Index> ints = integers_of(m);
    const std::string label = std::string(name) + (device ? " (device)" : " (cpu)");
    // From the PDHG relaxation (no start given), where each must find a point.
    expect_found(m, pdhg_feasibility_pump(m, ints, {}, settings_for(device)), label + " pump");
    expect_found(m, fix_and_propagate(m, ints, {}, settings_for(device)),
                 label + " fix-and-propagate");
    // From the all-zero point the pump still finds one. Fix-and-propagate may not: on the
    // general-integer model x = 0 is outside the propagated domain x >= 1, and both integers
    // next to it (1 and 2) propagate the box empty at the first column, with nothing to back
    // up to - the scheme tries the neighbours of the start, not every value. Whatever it
    // returns must still be feasible.
    const std::vector<double> zero(static_cast<std::size_t>(m.num_cols()), 0.0);
    expect_found(m, pdhg_feasibility_pump(m, ints, zero, settings_for(device)),
                 label + " pump from zero");
    expect_feasible_if_any(m, fix_and_propagate(m, ints, zero, settings_for(device)),
                           label + " fix-and-propagate from zero");
  }
}

TEST(PdhgHeuristics, EachFindsAPointOnSmallMilpsOnTheCpu) {
  const OneThread one_thread;
  finds_points_on_small_milps(false);
}

TEST(PdhgHeuristics, TheMixedModelIsCompletedByTheLpNotTakenFromPdhg) {
  const OneThread one_thread;
  // The rounding's continuous column comes from a PDHG point; the pump must hand back the
  // completion LP's z, which meets z >= 13 exactly, whatever PDHG's accuracy.
  const Model m = mixed_capacity();
  const PdhgHeuristicResult r =
      pdhg_feasibility_pump(m, integers_of(m), {}, settings_for(false));
  expect_found(m, r, "pump");
  if (!r.x.empty()) {
    EXPECT_GE(r.x[3], 13.0 - 1e-7);
  }
}

TEST(PdhgHeuristics, NothingOnAModelWithNoIntegerPoint) {
  const OneThread one_thread;
  // 2x + 2y = 3 has LP points and no integer one: both must come back empty, repair included.
  const Model m =
      make_model({{2, 2}}, {3.0}, {3.0}, {1.0, 1.0}, {0.0, 0.0}, {5.0, 5.0}, {true, true});
  const auto ints = integers_of(m);
  const PdhgHeuristicResult pump = pdhg_feasibility_pump(m, ints, {}, settings_for(false));
  const PdhgHeuristicResult fix = fix_and_propagate(m, ints, {}, settings_for(false));
  EXPECT_TRUE(pump.x.empty()) << pump.stopped;
  EXPECT_TRUE(fix.x.empty()) << fix.stopped;
  EXPECT_GE(pump.perturbations, 1) << "a pump that cannot succeed must have met a cycle";
}

TEST(PdhgHeuristics, FixAndPropagateStopsWithinItsBackUpBudgetOnAnOddCycle) {
  const OneThread one_thread;
  // x1 + x2 = x1 + x3 = x2 + x3 = 1 over binaries: the relaxation has every column at 1/2 and
  // no integer point exists. Each value of the first column fixed propagates the box empty,
  // so the search must back up, stay within its budget, stop, and return nothing - the
  // Feasibility Jump repair included.
  const Model m =
      make_model({{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0},
                 {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {true, true, true});
  for (const int budget : {0, 1, 3}) {
    PdhgHeuristicSettings s = settings_for(false);
    s.backtracks = budget;
    const PdhgHeuristicResult r = fix_and_propagate(m, integers_of(m), {0.5, 0.5, 0.5}, s);
    EXPECT_LE(r.backtracks, budget);
    EXPECT_TRUE(r.x.empty()) << r.stopped;
  }
}

oracle::GeneratedLp random_milp(std::mt19937_64& rng, int trial) {
  oracle::GeneratorConfig config;
  config.max_rows = 5;
  config.max_cols = 6;
  oracle::GeneratedLp lp = oracle::random_lp(rng, config);
  lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 0);
  for (Index j = 0; j < lp.num_cols; ++j) {
    lp.integral[static_cast<std::size_t>(j)] = (j + trial) % 3 == 0 ? 0 : 1;
  }
  return lp;
}

Model as_milp(const oracle::GeneratedLp& lp) {
  Model model = oracle::to_model(lp);
  for (Index j = 0; j < lp.num_cols; ++j) {
    if (lp.integral[static_cast<std::size_t>(j)] != 0) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
    }
  }
  return model;
}

void never_an_infeasible_point(bool device, int trials) {
  std::mt19937_64 rng(5090);
  int found = 0;
  int infeasible_checked = 0;
  for (int trial = 0; trial < trials; ++trial) {
    const oracle::GeneratedLp lp = random_milp(rng, trial);
    const Model model = as_milp(lp);
    const auto ints = integers_of(model);
    PdhgHeuristicSettings s = settings_for(device);
    s.seed = static_cast<std::uint64_t>(trial);
    // Budgets cut to keep 300 trials quick; the property checked does not depend on them.
    s.pump_rounds = 10;
    s.pdhg_iterations = 2000;
    const PdhgHeuristicResult pump = pdhg_feasibility_pump(model, ints, {}, s);
    const PdhgHeuristicResult fix = fix_and_propagate(model, ints, {}, s);
    for (const PdhgHeuristicResult* r : {&pump, &fix}) {
      if (r->x.empty()) continue;
      ++found;
      std::string why;
      EXPECT_TRUE(feasible(model, r->x, &why))
          << "trial " << trial << (r == &pump ? " pump: " : " fix-and-propagate: ") << why
          << "\n"
          << lp.to_text();
    }
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status == oracle::OracleStatus::kInfeasible) {
      ++infeasible_checked;
      EXPECT_TRUE(pump.x.empty())
          << "trial " << trial << ": a pump point on an infeasible MILP";
      EXPECT_TRUE(fix.x.empty()) << "trial " << trial << ": a point on an infeasible MILP";
    }
  }
  EXPECT_GT(found, trials / 6) << "the generator should give the heuristics models to solve";
  EXPECT_GT(infeasible_checked, 0);
}

TEST(PdhgHeuristics, NeverAnInfeasiblePointOnRandomMilpsOnTheCpu) {
  const OneThread one_thread;
  never_an_infeasible_point(false, 300);
}

TEST(PdhgHeuristics, TheSearchWithBothOnAgreesWithTheExactOracle) {
  const OneThread one_thread;
  std::mt19937_64 rng(50900);
  Options options;
  options.set_bool("gpu_pump", true);
  options.set_bool("gpu_fix_and_prop", true);
  options.set_string("gpu_heur_backend", "cpu");
  options.set_int("gpu_pump_max_iter", 10);
  // They run only when the root rounding and the dives found nothing; with the dives on, that
  // was 3 of 148 searches here, and 11 of 148 with them off - most of these tiny models end at
  // a root LP that is integral or infeasible - so the other heuristics are off and 600 trials
  // run.
  options.set_bool("mip_heuristics", false);
  options.set_string("mip_heur_dive_fractional", "off");
  options.set_bool("presolve", false);
  options.set_bool("log_to_console", true);
  options.set_double("mip_relative_gap", 0.0);
  options.set_double("mip_absolute_gap", 0.0);
  int compared = 0;
  int ran = 0;
  for (int trial = 0; trial < 600; ++trial) {
    const oracle::GeneratedLp lp = random_milp(rng, trial);
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status != oracle::OracleStatus::kOptimal &&
        exact.status != oracle::OracleStatus::kInfeasible) {
      continue;
    }
    const Model model = as_milp(lp);
    testing::internal::CaptureStdout();
    const Solution got = solve(model, options);
    std::fflush(stdout);
    const std::string log = testing::internal::GetCapturedStdout();
    if (std::regex_search(log, std::regex(R"(PDHG feasibility pump \(#509, CPU\))"))) ++ran;
    if (exact.status == oracle::OracleStatus::kInfeasible) {
      EXPECT_EQ(got.status, SolveStatus::kInfeasible) << "trial " << trial << "\n"
                                                      << lp.to_text();
    } else {
      ASSERT_EQ(got.status, SolveStatus::kOptimal)
          << "trial " << trial << ": " << got.message << "\n"
          << lp.to_text();
      const double want = exact.objective.to_double();
      EXPECT_NEAR(got.objective, want, 1e-6 * std::max(1.0, std::fabs(want)))
          << "trial " << trial << "\n"
          << lp.to_text();
      std::string why;
      // The search's answer is integral to the integrality tolerance, not exactly: a node LP
      // point the dive accepts keeps its rounding error. On trial 169 the search with both
      // heuristics off returns the same x0 = 2.0000000000000013 (measured on the L4 while
      // this test was written), so the tolerance here is the search's, not the heuristics'.
      EXPECT_TRUE(feasible(model, got.col_value, &why, 1e-6))
          << "trial " << trial << ": " << why;
    }
    ++compared;
  }
  EXPECT_GT(compared, 360) << "most generated MILPs should reach a verdict in the oracle";
  EXPECT_GT(ran, 20) << "the pump ran in " << ran << " of " << compared << " searches";
}

TEST(PdhgHeuristics, OffByDefaultAndTheDefaultPathIsTheExplicitlyOffPath) {
  const OneThread one_thread;
  const Options defaults;
  EXPECT_FALSE(defaults.get_bool("gpu_pump"));
  EXPECT_FALSE(defaults.get_bool("gpu_fix_and_prop"));
  // The default search against the same search with both switches set off explicitly and
  // every budget of theirs changed: the same nodes, objective, point and message, bit for bit.
  std::mt19937_64 rng(5091);
  int searched = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const Model model = as_milp(random_milp(rng, trial));
    Options unset;
    unset.set_bool("log_to_console", false);
    unset.set_bool("presolve", false);
    Options off = unset;
    off.set_bool("gpu_pump", false);
    off.set_bool("gpu_fix_and_prop", false);
    off.set_int("gpu_pump_max_iter", 3);
    off.set_int("gpu_fix_backtrack", 0);
    off.set_string("gpu_heur_backend", "cpu");
    const Solution a = solve(model, unset);
    const Solution b = solve(model, off);
    EXPECT_EQ(a.status, b.status) << "trial " << trial;
    EXPECT_EQ(a.nodes, b.nodes) << "trial " << trial;
    EXPECT_EQ(a.iterations, b.iterations) << "trial " << trial;
    EXPECT_EQ(a.objective, b.objective) << "trial " << trial;
    EXPECT_EQ(a.col_value, b.col_value) << "trial " << trial;
    EXPECT_EQ(a.message, b.message) << "trial " << trial;
    if (a.nodes > 0) ++searched;
  }
  EXPECT_GT(searched, 5) << "the comparison should reach the search";
}

// ---- The device -----------------------------------------------------------------------------

bool no_device() {
#ifndef SANKHYA_ENABLE_CUDA
  return true;
#else
  return !gpu::device_available(nullptr);
#endif
}

TEST(PdhgHeuristicsCuda, EachFindsAPointOnSmallMilpsOnTheDevice) {
  const OneThread one_thread;
  if (no_device()) GTEST_SKIP() << "no CUDA backend or no device: skipped, not passed.";
  finds_points_on_small_milps(true);
  const Model m = set_partition(8);
  EXPECT_TRUE(pdhg_feasibility_pump(m, integers_of(m), {}, settings_for(true)).on_device);
  EXPECT_TRUE(fix_and_propagate(m, integers_of(m), {}, settings_for(true)).on_device);
}

TEST(PdhgHeuristicsCuda, NeverAnInfeasiblePointOnRandomMilpsOnTheDevice) {
  const OneThread one_thread;
  if (no_device()) GTEST_SKIP() << "no CUDA backend or no device: skipped, not passed.";
  // Fewer trials than on the CPU: a device PDHG solve on a model this small is dominated by
  // its fixed setup cost, and 120 trials took 431 s on the L4.
  never_an_infeasible_point(true, 20);
}

}  // namespace
}  // namespace sankhya::mip
