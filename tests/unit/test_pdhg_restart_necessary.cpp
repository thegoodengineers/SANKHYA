// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP PDHG's "necessary" restart criterion (option pdhg_restart_necessary;
// Applegate et al., PDLP, NeurIPS 2021, section 3.3).
//
// Pinned here: the option is off by default, and with it on the engine still reaches a
// verified optimum that agrees with the default path on committed Netlib instances.
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

Options pdhg_options(bool necessary) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_double("pdhg_tolerance", 1e-8);
  options.set_bool("pdhg_restart_necessary", necessary);
  return options;
}

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

TEST(PdhgRestartNecessary, OffByDefault) {
  const Options defaults;
  EXPECT_FALSE(defaults.get_bool("pdhg_restart_necessary"));
}

TEST(PdhgRestartNecessary, ReachesTheDefaultPathsOptimum) {
  const char* const names[] = {"afiro", "sc50a", "sc105", "blend", "stocfor1"};
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution base = solve(model, pdhg_options(false));
    ASSERT_EQ(base.status, SolveStatus::kOptimal) << name << ": " << base.message;
    const Solution s = solve(model, pdhg_options(true));
    EXPECT_EQ(s.status, SolveStatus::kOptimal) << name << ": " << s.message;
    EXPECT_NEAR(s.objective, base.objective, 1e-6 * std::max(1.0, std::fabs(base.objective)))
        << name << ": " << s.iterations << " iterations against " << base.iterations;
  }
}

}  // namespace
}  // namespace sankhya
