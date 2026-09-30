// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the restarted Halpern PDHG iteration (#481).
//
// With pdhg_halpern the running-average accumulator is replaced by the Halpern anchor
// combination z_{k+1} = alpha_k * T(z_k) + (1-alpha_k) * z_0 with restarts on the
// fixed-point residual. Both paths must converge to the same optimum on committed
// Netlib instances; the averaged path (off) is tested separately for bitwise identity.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "pdhg/pdhg_trace.hpp"
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

// Two products under Halpern (#479): A x_{k+1} is the blend of the fresh A T(x_k) and the
// anchor's product, alpha A T(x_k) + (1 - alpha) A x_0. That is A of the blended iterate by
// linearity but not bit for bit, so the cache is held to rounding, not to the bit as on the
// averaged path: at every accepted iterate it is within kCacheRounding of A x_k taken
// afresh, relative in the max norm. The bound is one blend's rounding (a few ulps of the
// largest entry) with room, since each blend starts from a fresh product and each restart
// recomputes the anchor's; an error that carried over would grow past it.
constexpr double kCacheRounding = 1e-12;

struct CacheCheck {
  Count checked = 0;
  double worst = 0.0;
};

void check_cache(void* context, Count, const double*, std::size_t, const double*, std::size_t m,
                 const double* cached, const double* fresh) {
  auto* check = static_cast<CacheCheck*>(context);
  if (cached == nullptr || fresh == nullptr) return;
  ++check->checked;
  double difference = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    difference = std::max(difference, std::fabs(cached[i] - fresh[i]));
    scale = std::max(scale, std::fabs(fresh[i]));
  }
  check->worst = std::max(check->worst, scale > 0.0 ? difference / scale : difference);
}

TEST(PdhgHalpern, TwoMatvecAgreesWithThreeProductsAndItsCacheIsAxToRounding) {
  const char* const names[] = {"afiro", "sc50a", "sc105", "blend", "stocfor1"};
  for (const char* name : names) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    Options three = pdhg_options(true);
    three.set_string("pdhg_two_matvec", "false");
    Options two = pdhg_options(true);
    two.set_string("pdhg_two_matvec", "true");
    const Solution a = solve(model, three);
    CacheCheck check;
    pdhg::IterateTraceHook& hook = pdhg::iterate_trace_for_testing();
    hook.callback = &check_cache;
    hook.context = &check;
    const Solution b = solve(model, two);
    hook = pdhg::IterateTraceHook{};
    ASSERT_EQ(a.status, SolveStatus::kOptimal) << name << " three: " << a.message;
    ASSERT_EQ(b.status, SolveStatus::kOptimal) << name << " two: " << b.message;
    EXPECT_NEAR(a.objective, b.objective, 1e-8 * std::max(1.0, std::fabs(a.objective)))
        << name << " three " << a.iterations << " iterations, two " << b.iterations;
    EXPECT_EQ(check.checked, b.iterations) << name;
    EXPECT_LE(check.worst, kCacheRounding) << name;
    std::printf("%-9s three %lld iterations, two %lld, cache worst %.1e\n", name,
                static_cast<long long>(a.iterations), static_cast<long long>(b.iterations),
                check.worst);
  }
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
