// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP PDHG's relative-KKT termination (option pdhg_relative_stop; Applegate et
// al., PDLP, NeurIPS 2021, section 2).
//
// Pinned here: the option is on by default at the 1e-8 hand-off; with it on, PDHG alone never
// claims `optimal` for a point short of the project's absolute standard; and through the
// default pipeline (the interior-point polish on) the answer is a verified optimum agreeing
// with the default path on committed Netlib instances.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options pdhg_options(bool relative_stop, bool polish) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", polish);
  options.set_double("pdhg_tolerance", 1e-8);
  options.set_bool("pdhg_relative_stop", relative_stop);
  return options;
}

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

TEST(PdhgRelativeStop, OnByDefaultAtTheTightHandOff) {
  const Options defaults;
  EXPECT_TRUE(defaults.get_bool("pdhg_relative_stop"));
  EXPECT_DOUBLE_EQ(defaults.get_double("pdhg_tolerance"), tol::kPdhgTight);
}

TEST(PdhgRelativeStop, AloneItNeverClaimsOptimalShortOfTheStandard) {
  const char* const names[] = {"afiro", "sc50a", "sc105", "blend", "stocfor1", "adlittle"};
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution s = solve(model, pdhg_options(true, false));
    ASSERT_TRUE(s.status == SolveStatus::kOptimal || s.status == SolveStatus::kFeasible ||
                s.status == SolveStatus::kIterationLimit)
        << name << ": " << to_string(s.status) << " - " << s.message;
    if (s.status == SolveStatus::kOptimal) {
      EXPECT_LE(s.primal_infeasibility, tol::kPrimalFeasibility) << name;
      EXPECT_LE(s.dual_infeasibility, tol::kDualFeasibility) << name;
    }
    if (s.status == SolveStatus::kFeasible) {
      EXPECT_LE(s.primal_infeasibility, tol::kPrimalFeasibility) << name;
    }
  }
}

TEST(PdhgRelativeStop, WithThePolishItReachesTheDefaultPathsOptimum) {
  const char* const names[] = {"afiro", "sc50a", "sc105", "blend", "stocfor1"};
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution base = solve(model, pdhg_options(false, true));
    ASSERT_EQ(base.status, SolveStatus::kOptimal) << name << ": " << base.message;
    const Solution s = solve(model, pdhg_options(true, true));
    EXPECT_EQ(s.status, SolveStatus::kOptimal) << name << ": " << s.message;
    EXPECT_NEAR(s.objective, base.objective, 1e-6 * std::max(1.0, std::fabs(base.objective)))
        << name << ": " << s.iterations << " iterations against " << base.iterations;
  }
}

}  // namespace
}  // namespace sankhya
