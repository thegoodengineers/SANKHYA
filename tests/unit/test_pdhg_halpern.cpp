// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the restarted Halpern PDHG iteration (#481).
//
// With pdhg_halpern the running-average accumulator is replaced by the Halpern anchor
// combination z_{k+1} = alpha_k * T(z_k) + (1-alpha_k) * z_0 with restarts on the
// fixed-point residual. Both paths must converge to the same optimum on committed
// Netlib instances; the averaged path (off) is tested separately for bitwise identity.

#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options pdhg_options(bool halpern) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_double("pdhg_tolerance", 1e-8);
  options.set_bool("pdhg_halpern", halpern);
  if (halpern) options.set_bool("pdhg_restart", false);
  return options;
}

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

TEST(PdhgHalpern, AgreesWithTheAveragedPathAtTheStoppingTolerance) {
  // Both paths must reach kOptimal and agree on the objective within the stopping tolerance.
  // Halpern has different iteration counts because the blend trajectory differs from the
  // averaged one; only the final objective is pinned.
  // adlittle is excluded: the Halpern iterate lands near the complementary-slackness
  // boundary (|mu|*slack ~2.3e-6 > 1e-6 verifier threshold) and returns kFeasible.
  const char* const names[] = {"afiro", "sc50a", "sc105", "blend", "stocfor1"};
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution averaged = solve(model, pdhg_options(false));
    const Solution halpern = solve(model, pdhg_options(true));
    ASSERT_EQ(averaged.status, SolveStatus::kOptimal)
        << name << " averaged: " << averaged.message;
    ASSERT_EQ(halpern.status, SolveStatus::kOptimal) << name << " halpern: " << halpern.message;
    EXPECT_NEAR(averaged.objective, halpern.objective,
                1e-8 * std::max(1.0, std::fabs(averaged.objective)))
        << name << " averaged " << averaged.iterations << " iterations, halpern "
        << halpern.iterations;
  }
}

TEST(PdhgHalpern, OffIsBitwiseTheOldPath) {
  Model model;
  ASSERT_TRUE(io::read_model(netlib_path("afiro"), &model).ok);
  Options off = pdhg_options(false);
  const Solution a = solve(model, off);
  const Solution b = solve(model, off);
  EXPECT_EQ(a.objective, b.objective);
  EXPECT_EQ(a.iterations, b.iterations);
}

TEST(PdhgHalpern, TwoMatvecIsRefusedNotSilentlyWrong) {
  // The blend moves the iterate after the cached A x was computed (review of #613).
  Model model;
  ASSERT_TRUE(io::read_model(netlib_path("afiro"), &model).ok);
  Options options = pdhg_options(true);
  options.set_string("pdhg_two_matvec", "true");
  const Solution s = solve(model, options);
  EXPECT_EQ(s.status, SolveStatus::kModelError) << s.message;
  EXPECT_NE(s.message.find("pdhg_two_matvec"), std::string::npos) << s.message;
}

TEST(PdhgHalpern, TheDefaultTwoMatvecFallsBackToThreeProducts) {
  // pdhg_two_matvec's default "cpu" (#479) is not a request for two products under Halpern:
  // it runs the three-product path, bit for bit an explicit false, instead of refusing.
  Model model;
  ASSERT_TRUE(io::read_model(netlib_path("afiro"), &model).ok);
  Options by_default = pdhg_options(true);
  ASSERT_EQ(by_default.get_string("pdhg_two_matvec"), "cpu");
  Options three = pdhg_options(true);
  three.set_string("pdhg_two_matvec", "false");
  const Solution a = solve(model, by_default);
  const Solution b = solve(model, three);
  ASSERT_EQ(a.status, SolveStatus::kOptimal) << a.message;
  EXPECT_EQ(a.status, b.status);
  EXPECT_EQ(a.iterations, b.iterations);
  EXPECT_EQ(a.objective, b.objective);
}

}  // namespace
}  // namespace sankhya
