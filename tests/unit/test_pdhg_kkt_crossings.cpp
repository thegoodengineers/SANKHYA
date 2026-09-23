// SPDX-License-Identifier: Apache-2.0
// SANKHYA - PDHG's first crossings of the relative KKT error (#486).
//
// One run at 1e-8 yields the three times a published comparison needs. What can go wrong
// is the bookkeeping: a level recorded before it was reached, the levels out of order, a
// level reported reached when the run stopped short of it, or the numbers lost on the way
// through presolve and the polish. These cases pin each of those on committed Netlib
// instances.

#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options pdhg_options(double tolerance, bool presolve, bool polish) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", presolve);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", polish);
  options.set_double("pdhg_tolerance", tolerance);
  options.set_int("iteration_limit", 200000);
  return options;
}

Model netlib(const char* name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (std::string(name) + ".mps"))
          .string();
  EXPECT_TRUE(io::read_model(path, &model).ok) << name;
  return model;
}

TEST(PdhgKktCrossings, ThreeLevelsInOrderOnARunThatReachesThemAll) {
  const Solution s = solve(netlib("afiro"), pdhg_options(1e-8, false, false));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  ASSERT_TRUE(std::isfinite(s.kkt_1e4_seconds)) << s.message;
  ASSERT_TRUE(std::isfinite(s.kkt_1e6_seconds)) << s.message;
  ASSERT_TRUE(std::isfinite(s.kkt_1e8_seconds)) << s.message;
  EXPECT_GE(s.kkt_1e4_iterations, 0);
  EXPECT_LE(s.kkt_1e4_iterations, s.kkt_1e6_iterations);
  EXPECT_LE(s.kkt_1e6_iterations, s.kkt_1e8_iterations);
  EXPECT_LE(s.kkt_1e8_iterations, s.iterations);
  EXPECT_LE(s.kkt_1e4_seconds, s.kkt_1e6_seconds);
  EXPECT_LE(s.kkt_1e6_seconds, s.kkt_1e8_seconds);
  EXPECT_LE(s.kkt_1e8_seconds, s.solve_seconds);
}

TEST(PdhgKktCrossings, ALevelTheRunNeverReachedIsNotANumber) {
  // Stop on the caller's 1e-4 request; the run ends before 1e-8 and must say so.
  Options options = pdhg_options(1e-4, false, false);
  options.set_bool("pdhg_stop_at_request", true);
  const Solution s = solve(netlib("adlittle"), options);
  ASSERT_TRUE(s.status == SolveStatus::kOptimal || s.status == SolveStatus::kFeasible)
      << s.message;
  EXPECT_TRUE(std::isfinite(s.kkt_1e4_seconds)) << s.message;
  EXPECT_GE(s.kkt_1e4_iterations, 0);
  if (!std::isfinite(s.kkt_1e8_seconds)) {
    EXPECT_EQ(s.kkt_1e8_iterations, -1);
  } else {
    // A run that happened to cross 1e-8 anyway reports it; it cannot be earlier than 1e-4.
    EXPECT_LE(s.kkt_1e4_iterations, s.kkt_1e8_iterations);
  }
}

TEST(PdhgKktCrossings, SurviveThePresolveRoundTripAndThePolish) {
  const Solution plain = solve(netlib("sc50a"), pdhg_options(1e-8, false, false));
  const Solution presolved = solve(netlib("sc50a"), pdhg_options(1e-8, true, false));
  const Solution polished = solve(netlib("sc50a"), pdhg_options(1e-8, true, true));
  for (const Solution* s : {&plain, &presolved, &polished}) {
    ASSERT_TRUE(s->status == SolveStatus::kOptimal || s->status == SolveStatus::kFeasible)
        << s->message;
    EXPECT_TRUE(std::isfinite(s->kkt_1e4_seconds)) << s->algorithm << ": " << s->message;
    EXPECT_GE(s->kkt_1e4_iterations, 0) << s->algorithm;
  }
}

TEST(PdhgKktCrossings, TheSimplexLeavesThemUnset) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "dual-simplex");
  const Solution s = solve(netlib("afiro"), options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_TRUE(std::isnan(s.kkt_1e4_seconds));
  EXPECT_EQ(s.kkt_1e8_iterations, -1);
}

}  // namespace
}  // namespace sankhya
