// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the unscaled retry starts from the scaled attempt's basis when that attempt
// ended optimal (src/simplex/primal_simplex.cpp, solve_with_scaling).
//
// pilot4 is the Netlib medium-tier instance that takes this route on main: the scaled
// attempt is optimal after about a thousand iterations, but its point is not feasible to
// the tolerance in original units, and the retry used to begin again from the slack basis.
// Started from the scaled basis it needs a handful of pivots. What is under test is that
// the shortcut changes the route and not the claim: the answer is optimal, at the
// published objective, and meets primal and dual feasibility measured afresh against the
// model - the checks tools/verify_solution.py makes.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

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

TEST(UnscaledRetryWarm, Pilot4IsCleanedUpFromTheScaledBasisAndStillVerifies) {
  const Model model = netlib("pilot4");
  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NE(solution.message.find("the unscaled retry from its basis produced this answer"),
            std::string::npos)
      << solution.message;

  // Netlib's readme: PILOT4 -2.5811392641E+03.
  const double published = -2.5811392641e+03;
  EXPECT_LE(std::fabs(solution.objective - published) / std::max(1.0, std::fabs(published)),
            1e-6)
      << solution.objective;

  Solution measured = solution;
  measured.recompute_quality(model);
  EXPECT_LE(measured.primal_infeasibility_scaled, tol::kPrimalFeasibility);
  EXPECT_LE(measured.dual_infeasibility_scaled, tol::kDualFeasibility);
}

}  // namespace
}  // namespace sankhya
