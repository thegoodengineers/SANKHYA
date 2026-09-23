// SPDX-License-Identifier: Apache-2.0
// SANKHYA - crossover from an interior point that stopped short of optimal (#474).
//
// THE CLAIM UNDER TEST IS STILL THE VERTEX. A start that is merely feasible, or an iterate
// the interior point was stopped at, is a place for the simplex to begin and nothing more:
// the answer must be optimal, sit at the published objective, be a basis, and meet the KKT
// conditions when they are measured afresh against the model - the checks
// tools/verify_solution.py makes, done here on the returned vectors. The interior point is
// stopped deliberately by iteration_limit, which counts the same on every machine, at every
// iteration before it would converge.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/crossover.hpp"

namespace sankhya {
namespace {

Model netlib(const std::string& name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (name + ".mps"))
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

Options ipm_options(bool from_nonoptimal) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("crossover_from_nonoptimal", from_nonoptimal);
  return options;
}

/// What the independent verifier checks of an optimal LP answer, measured afresh on the
/// returned vectors: primal and dual feasibility at the project tolerances, complementary
/// slackness at the verifier's, and a basis with every nonbasic entry on its bound.
void expect_a_verified_vertex(const Model& model, const Solution& s, const std::string& what) {
  Solution measured = s;
  measured.recompute_quality(model);
  EXPECT_LE(measured.primal_infeasibility_scaled, tol::kPrimalFeasibility) << what;
  EXPECT_LE(measured.dual_infeasibility_scaled, tol::kDualFeasibility) << what;
  EXPECT_LE(measured.complementarity_violation, tol::kComplementarity) << what;
  ASSERT_EQ(static_cast<Index>(s.col_status.size()), model.num_cols()) << what;
  ASSERT_EQ(static_cast<Index>(s.row_status.size()), model.num_rows()) << what;
  Index basic = 0;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double x = s.col_value[u];
    const double scale = std::max(1.0, std::fabs(x));
    switch (s.col_status[u]) {
      case BasisStatus::kBasic: ++basic; break;
      case BasisStatus::kAtLower:
      case BasisStatus::kFixed:
        EXPECT_LE(std::fabs(x - model.col_lower[u]), tol::kPrimalFeasibility * scale) << what;
        break;
      case BasisStatus::kAtUpper:
        EXPECT_LE(std::fabs(x - model.col_upper[u]), tol::kPrimalFeasibility * scale) << what;
        break;
      case BasisStatus::kNonbasicFree: EXPECT_EQ(x, 0.0) << what; break;
      case BasisStatus::kUnknown:
        ADD_FAILURE() << what << ": column " << j << " unknown";
        break;
    }
  }
  for (const BasisStatus status : s.row_status) basic += status == BasisStatus::kBasic ? 1 : 0;
  EXPECT_EQ(basic, model.num_rows()) << what << ": a basis has exactly m basic entries";
}

TEST(CrossoverFromNonoptimal, EveryEarlyStopThatPassesTheGateEndsOptimalAtAVertex) {
  // Each instance is stopped at every iteration count short of convergence. Where the stopped
  // point passes the gate (scaled infeasibility at most 1e-4 both ways) the crossover must end
  // optimal at the published objective, at a verified vertex; where it does not, the interior
  // point's answer must come back as it was - never a claim the simplex did not prove.
  struct Instance {
    const char* name;
    double published;
  };
  const Instance instances[] = {{"afiro", -464.75314286},
                                {"sc50b", -70.0},
                                {"share2b", -415.73224074},
                                {"adlittle", 225494.96316}};
  Logger quiet(nullptr);
  Index crossed = 0;
  Index stops = 0;
  for (const Instance& instance : instances) {
    const Model model = netlib(instance.name);
    const Solution full = ipm::solve_ipm(model, ipm_options(true), quiet);
    ASSERT_EQ(full.status, SolveStatus::kOptimal) << instance.name << ": " << full.message;
    for (Count stop = 1; stop < full.iterations; ++stop) {
      Options stopped = ipm_options(true);
      stopped.set_int("iteration_limit", stop);
      Solution interior = ipm::solve_ipm(model, stopped, quiet);
      ++stops;
      const std::string what = std::string(instance.name) + " stopped at " +
                               std::to_string(stop) + ": " + interior.message;
      ASSERT_EQ(interior.status, SolveStatus::kIterationLimit) << what;
      const bool usable = crossover_start_is_usable(model, interior);
      const double interior_objective = interior.objective;
      const Timer timer;
      // The pivots run without the iteration limit that stopped the interior point.
      const Solution after = crossover_when_wanted(model, std::move(interior),
                                                   ipm_options(true), quiet, nullptr, timer);
      if (!usable) {
        EXPECT_EQ(after.status, SolveStatus::kIterationLimit) << what;
        EXPECT_EQ(after.objective, interior_objective) << what;
        EXPECT_EQ(after.algorithm.find("crossover"), std::string::npos) << what;
        continue;
      }
      ++crossed;
      ASSERT_EQ(after.status, SolveStatus::kOptimal) << what << " -> " << after.message;
      EXPECT_NEAR(after.objective, instance.published,
                  1e-6 * std::max(1.0, std::fabs(instance.published)))
          << what;
      EXPECT_NE(after.algorithm.find("crossover"), std::string::npos) << what;
      EXPECT_NE(after.message.find("#474"), std::string::npos) << after.message;
      expect_a_verified_vertex(model, after, what);
    }
  }
  // The sweep has to have tested something: several early stops pass the gate.
  std::cout << "crossover from a stopped interior point: " << crossed << " of " << stops
            << " early stops passed the gate and ended optimal" << std::endl;
  EXPECT_GE(crossed, 4);
}

TEST(CrossoverFromNonoptimal, ThroughSolveAStoppedInteriorPointEndsOptimal) {
  // The whole pipeline - presolve, the interior point stopped at 8 iterations where it needs
  // 9 on afiro, the crossover, postsolve, the status guard and the certificate check - with
  // the option on, and the same run with it off for contrast. Deterministic: an iteration
  // limit, no clock. (The pivots share iteration_limit with the interior point, as they
  // always have; afiro's crossover needs 7.)
  const Model model = netlib("afiro");
  Options on = ipm_options(true);
  on.set_int("iteration_limit", 8);
  const Solution crossed = solve(model, on);
  ASSERT_EQ(crossed.status, SolveStatus::kOptimal) << crossed.message;
  EXPECT_NEAR(crossed.objective, -464.75314286, 1e-6 * 464.75314286);
  EXPECT_NE(crossed.message.find("ended iteration_limit"), std::string::npos)
      << crossed.message;
  expect_a_verified_vertex(model, crossed, "afiro through solve()");

  Options off = ipm_options(false);
  off.set_int("iteration_limit", 8);
  const Solution alone = solve(model, off);
  EXPECT_EQ(alone.status, SolveStatus::kIterationLimit) << alone.message;
  EXPECT_EQ(alone.algorithm.find("crossover"), std::string::npos) << alone.algorithm;
}

TEST(CrossoverFromNonoptimal, TheGateRefusesWhatIsNotAUsablePoint) {
  const Model model = netlib("afiro");
  Logger quiet(nullptr);
  Options stopped = ipm_options(true);
  stopped.set_int("iteration_limit", 8);
  const Solution base = ipm::solve_ipm(model, stopped, quiet);
  ASSERT_TRUE(crossover_start_is_usable(model, base)) << base.message;

  Solution interrupted = base;
  interrupted.status = SolveStatus::kInterrupted;
  EXPECT_FALSE(crossover_start_is_usable(model, interrupted)) << "the caller asked to stop";

  Solution too_far = base;
  too_far.primal_infeasibility_scaled = 10.0 * tol::kCrossoverStartInfeasibility;
  EXPECT_FALSE(crossover_start_is_usable(model, too_far));
  too_far = base;
  too_far.dual_infeasibility_scaled = 10.0 * tol::kCrossoverStartInfeasibility;
  EXPECT_FALSE(crossover_start_is_usable(model, too_far));

  Solution not_finite = base;
  not_finite.col_value[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(crossover_start_is_usable(model, not_finite));

  Solution wrong_size = base;
  wrong_size.col_value.pop_back();
  EXPECT_FALSE(crossover_start_is_usable(model, wrong_size));

  // A numerical error is a start only with an iterate attached: all-zero vectors are what a
  // numerical error carries when nothing was attached, and zero is not a computed point.
  Solution nothing_attached;
  nothing_attached.allocate_for(model);
  nothing_attached.status = SolveStatus::kNumericalError;
  nothing_attached.recompute_quality(model);
  EXPECT_FALSE(crossover_start_is_usable(model, nothing_attached));
  Solution attached = base;
  attached.status = SolveStatus::kNumericalError;
  EXPECT_TRUE(crossover_start_is_usable(model, attached));

  // And with the option off, a stopped answer is not crossed over at all.
  const Timer timer;
  const Solution left =
      crossover_when_wanted(model, base, ipm_options(false), quiet, nullptr, timer);
  EXPECT_EQ(left.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(left.objective, base.objective);
}

TEST(CrossoverFromNonoptimal, AnAttachedIterateIsWithdrawnWhenNothingUsesIt) {
  // A numerical error claims no point (#200). The iterate attached for the crossover must not
  // leave the solve on one: when the crossover is off, or refuses it, the vectors go back to
  // zero and the measured quality with them.
  const Model model = netlib("afiro");
  Logger quiet(nullptr);
  Options stopped = ipm_options(true);
  stopped.set_int("iteration_limit", 8);
  Solution attached = ipm::solve_ipm(model, stopped, quiet);
  attached.status = SolveStatus::kNumericalError;
  const Timer timer;
  Options off = ipm_options(true);
  off.set_bool("crossover", false);
  const Solution withdrawn = crossover_when_wanted(model, attached, off, quiet, nullptr, timer);
  EXPECT_EQ(withdrawn.status, SolveStatus::kNumericalError);
  for (const double v : withdrawn.col_value) EXPECT_EQ(v, 0.0);
  for (const double v : withdrawn.row_dual) EXPECT_EQ(v, 0.0);

  Solution refused = attached;
  refused.primal_infeasibility_scaled = 1.0;
  const Solution after =
      crossover_when_wanted(model, refused, ipm_options(true), quiet, nullptr, timer);
  EXPECT_EQ(after.status, SolveStatus::kNumericalError);
  for (const double v : after.col_value) EXPECT_EQ(v, 0.0);
}

TEST(CrossoverFromNonoptimal, TheTimeReserveAppliesOnlyWhenTheOptionIsOnAndTheLimitFinite) {
  Options on = ipm_options(true);
  on.set_double("time_limit", 100.0);
  EXPECT_DOUBLE_EQ(interior_point_options_before_crossover(on).get_double("time_limit"), 90.0);
  on.set_double("crossover_time_reserve", 0.25);
  EXPECT_DOUBLE_EQ(interior_point_options_before_crossover(on).get_double("time_limit"), 75.0);

  Options off = ipm_options(false);
  off.set_double("time_limit", 100.0);
  EXPECT_DOUBLE_EQ(interior_point_options_before_crossover(off).get_double("time_limit"),
                   100.0);

  const Options no_limit = ipm_options(true);
  EXPECT_EQ(interior_point_options_before_crossover(no_limit).get_double("time_limit"),
            no_limit.get_double("time_limit"));
}

}  // namespace
}  // namespace sankhya
