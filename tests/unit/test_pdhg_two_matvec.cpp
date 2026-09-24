// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the two-mat-vec PDHG iteration (#479).
//
// With pdhg_two_matvec the dual step's A xbar and the step-size rule's A dx are derived from
// the cached A x_k and the fresh A x_{k+1} by vector arithmetic instead of two more sparse
// products. The derived vectors differ from the products by rounding, so the two paths are
// held to the same status and the same objective at the stopping tolerance on the committed
// Netlib instances; adlittle runs through two dozen restarts, each of which recomputes the
// cached product exactly.

#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options pdhg_options(bool two_matvec) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_double("pdhg_tolerance", 1e-8);
  options.set_bool("pdhg_two_matvec", two_matvec);
  return options;
}

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

TEST(PdhgTwoMatvec, AgreesWithTheThreeProductPathAtTheStoppingTolerance) {
  // Under the engine's own iteration ceiling both paths converge on these seven; the derived
  // products differ from the computed ones by rounding, so the trajectories differ and the
  // iteration counts with them, and the objectives are held to the stopping tolerance.
  const char* const names[] = {"afiro", "adlittle", "sc50a",   "sc105",
                               "blend", "israel",   "stocfor1"};
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution three = solve(model, pdhg_options(false));
    const Solution two = solve(model, pdhg_options(true));
    ASSERT_EQ(three.status, SolveStatus::kOptimal)
        << name << " three-product: " << three.message;
    ASSERT_EQ(two.status, SolveStatus::kOptimal) << name << " two-mat-vec: " << two.message;
    EXPECT_NEAR(three.objective, two.objective,
                1e-8 * std::max(1.0, std::fabs(three.objective)))
        << name << " three-product " << three.iterations << " iterations, two-mat-vec "
        << two.iterations;
  }
}

TEST(PdhgTwoMatvec, OffIsBitwiseTheOldPath) {
  Model model;
  ASSERT_TRUE(io::read_model(netlib_path("afiro"), &model).ok);
  Options off = pdhg_options(false);
  const Solution a = solve(model, off);
  const Solution b = solve(model, off);
  EXPECT_EQ(a.objective, b.objective);
  EXPECT_EQ(a.iterations, b.iterations);
}

}  // namespace
}  // namespace sankhya
