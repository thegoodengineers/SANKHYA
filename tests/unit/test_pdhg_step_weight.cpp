// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP PDHG's constant step on a certified norm bound and its PID primal weight
// (#482, options pdhg_constant_step and pdhg_primal_weight_pid).
//
// What is pinned here: both options are off by default; each one, and both together, still
// reach a verified optimum that agrees with the default path on committed Netlib instances,
// on the PDLP-averaged path and under the Halpern iteration; and the PID controller at
// kp = 0.5, ki = kd = 0 is PDLP's theta = 0.5 smoothing, the update the default runs.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "qp/qp_first_order_accel.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options pdhg_options(bool constant_step, bool pid, bool halpern) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_double("pdhg_tolerance", 1e-8);
  options.set_bool("pdhg_constant_step", constant_step);
  options.set_bool("pdhg_primal_weight_pid", pid);
  options.set_bool("pdhg_halpern", halpern);
  if (halpern) options.set_bool("pdhg_restart", false);
  return options;
}

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

TEST(PdhgStepWeight, BothOptionsAreOffByDefault) {
  const Options defaults;
  EXPECT_FALSE(defaults.get_bool("pdhg_constant_step"));
  EXPECT_FALSE(defaults.get_bool("pdhg_primal_weight_pid"));
}

TEST(PdhgStepWeight, PidAtProportionalOneHalfIsPdlpsSmoothing) {
  // log omega - 0.5 log(omega dx / dy) = 0.5 log(dy / dx) + 0.5 log omega, the update the
  // default path applies at a restart ([PDLP] section 3.2, theta = 0.5).
  const qp::PidGains gains{0.5, 0.0, 0.0};
  qp::PidState state;
  double omega = 1.0;
  double reference = 1.0;
  const double movements[][2] = {{3.0, 0.2}, {0.5, 4.0}, {1e-3, 7.0}, {2.0, 2.0}, {9.0, 1e-4}};
  for (const auto& move : movements) {
    omega = qp::pid_primal_weight(omega, move[0], move[1], gains, &state);
    reference = std::clamp(
        std::exp(0.5 * std::log(move[1] / move[0]) + 0.5 * std::log(reference)), 1e-6, 1e6);
    EXPECT_NEAR(omega, reference, 1e-13 * reference);
  }
}

TEST(PdhgStepWeight, EveryArmReachesTheDefaultPathsOptimum) {
  // adlittle is left out as in test_pdhg_halpern.cpp: under Halpern its point misses the
  // verifier's complementarity threshold and is reported feasible, on the default step too.
  const char* const names[] = {"afiro", "sc50a", "sc105", "blend", "stocfor1"};
  for (const bool halpern : {false, true}) {
    for (const char* name : names) {
      Model model;
      ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
      const Solution base = solve(model, pdhg_options(false, false, halpern));
      ASSERT_EQ(base.status, SolveStatus::kOptimal) << name << ": " << base.message;
      for (const auto& arm :
           {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
        const Solution s = solve(model, pdhg_options(arm.first, arm.second, halpern));
        const std::string label = std::string(name) + (halpern ? " halpern" : " averaged") +
                                  (arm.first ? " constant-step" : "") +
                                  (arm.second ? " pid" : "");
        EXPECT_EQ(s.status, SolveStatus::kOptimal) << label << ": " << s.message;
        EXPECT_NEAR(s.objective, base.objective,
                    1e-6 * std::max(1.0, std::fabs(base.objective)))
            << label << ": " << s.iterations << " iterations against " << base.iterations;
      }
    }
  }
}

TEST(PdhgStepWeight, PidAtDefaultGainsStaysInsideTheDefaultPathsSeedSpread) {
  // kp = 0.5, ki = kd = 0 is the default update written another way, equal to rounding, but
  // not bit for bit: the two expressions round differently. That is enough to move the run.
  // The trajectory is sensitive to the last bits of anything that feeds a restart decision,
  // and the default path itself shows how much: random_seed changes nothing but the power
  // iteration's start, so the norm estimate moves in its last digits, and measured on this
  // change sc50a takes 12,560 to 14,880 iterations over seeds 0 to 3 and blend 62,640 to
  // 83,160. So the test is: the same verdict, the same optimum, and an iteration count inside
  // the default path's own range over those four seeds, widened by a fifth either side. An
  // iteration A/B between the legs therefore has to be read over several seeds
  // (pdhg_step_weight_ab.py --seeds), not one.
  const char* const names[] = {"afiro", "sc50a", "blend"};
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution base = solve(model, pdhg_options(false, false, false));
    Count low = base.iterations;
    Count high = base.iterations;
    for (int seed = 1; seed <= 3; ++seed) {
      Options seeded = pdhg_options(false, false, false);
      seeded.set_int("random_seed", seed);
      const Solution other = solve(model, seeded);
      low = std::min(low, other.iterations);
      high = std::max(high, other.iterations);
    }
    Options pid = pdhg_options(false, true, false);
    pid.set_double("pdhg_pid_kp", 0.5);
    pid.set_double("pdhg_pid_ki", 0.0);
    pid.set_double("pdhg_pid_kd", 0.0);
    const Solution s = solve(model, pid);
    EXPECT_EQ(s.status, base.status) << name;
    EXPECT_NEAR(s.objective, base.objective, 1e-8 * std::max(1.0, std::fabs(base.objective)))
        << name;
    EXPECT_GE(static_cast<double>(s.iterations), 0.8 * static_cast<double>(low))
        << name << ": " << s.iterations << " against the seeds' " << low << " to " << high;
    EXPECT_LE(static_cast<double>(s.iterations), 1.2 * static_cast<double>(high))
        << name << ": " << s.iterations << " against the seeds' " << low << " to " << high;
  }
}

}  // namespace
}  // namespace sankhya
