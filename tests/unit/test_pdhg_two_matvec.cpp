// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the two-mat-vec PDHG iteration (#479).
//
// With pdhg_two_matvec the dual step's A xbar and the step-size rule's A dx are derived from
// the cached A x_k and the fresh A x_{k+1} by vector arithmetic instead of two more sparse
// products. The derived vectors differ from the products by rounding, so the two paths are
// held to the same status and the same objective at the stopping tolerance on the committed
// Netlib instances; adlittle runs through two dozen restarts, each of which recomputes the
// cached product exactly.
//
// The iterate tests below compare the two paths ITERATE BY ITERATE over the issue's 10,000
// iterations on the nine committed Netlib instances, through the trace seam in
// src/pdhg/pdhg_trace.hpp, CPU and CUDA each against its own three-product path; what they
// hold, and why it is not the issue's literal 1e-12 over 10,000, is set out above them.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pdhg/pdhg_trace.hpp"
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
  options.set_string("pdhg_two_matvec", two_matvec ? "true" : "false");
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

TEST(PdhgTwoMatvec, TheDefaultIsTwoProductsOnTheCpuEngine) {
  // The default "cpu" (#479, the A/B in bench/results/pdhg-two-matvec-58a8374.csv) is the
  // explicit "true" path on the CPU engine, bit for bit.
  EXPECT_EQ(Options().get_string("pdhg_two_matvec"), "cpu");
  Model model;
  ASSERT_TRUE(io::read_model(netlib_path("adlittle"), &model).ok);
  Options by_default = pdhg_options(true);
  by_default.set_string("pdhg_two_matvec", "cpu");
  const Solution a = solve(model, by_default);
  const Solution b = solve(model, pdhg_options(true));
  EXPECT_EQ(a.status, b.status);
  EXPECT_EQ(a.iterations, b.iterations);
  EXPECT_EQ(a.objective, b.objective);
}

// ---- Iterate by iterate (#479 acceptance) ---------------------------------------------------
//
// THE ISSUE ASKS FOR 1e-12 RELATIVE OVER 10,000 ITERATIONS, AND NO IMPLEMENTATION CAN MEET
// IT. Measured on an L4 box (the PR for this test has the table): the two paths agree to a
// few ulps at first (0 at iteration 1, 1e-17..1e-15 at 10), and the difference then grows
// by about a decade every 10-20 iterations, crossing 1e-12 between iterations 33 and 123 on
// the nine and reaching 1e-2..4e-1 relative by 10,000. The growth is the iteration's own:
// the adaptive step of [PDLP] section 3.1 sets eta_{k+1} from the movement / interaction
// ratio of the step just taken, so a rounding-level change in one iterate changes the next
// step size, and the restarted method amplifies it. The control is on the device: the CUDA
// three-product path run twice, whose only difference is the order its atomic block
// reductions land in (#451), drifts from ITSELF by the same order, and by a different
// amount on each run (israel crosses 1e-12 at iteration 62 on one run and 67 on the next,
// and reaches 1.9e-1 on one and 2.0e-2 on the other; adlittle, blend and, on one run,
// share2b likewise). Holding the derivation to 1e-12 over 10,000 iterations would hold
// the engine to a reproducibility its own reductions do not have.
//
// What is held instead, and why it is the whole statement about the derivation:
//  1. LOCKSTEP. Over the first kLockstepIterations accepted iterates the two paths agree
//     to the issue's 1e-12, relative in the Euclidean norm of z_k = (x_k, y_k). The window
//     is measured, not derived: the largest difference inside it on the L4 was 1.2e-14
//     (CPU afiro; CUDA 6.6e-15, afiro), and the earliest crossing of 1e-12 was at 33 (CUDA
//     israel, which was 1.6e-14 at 30). Twenty iterates leave two decades of margin on
//     every instance, and a wrong derivation shows at its first step, not after twenty.
//  2. THE CACHE IS EXACT. At every accepted iterate of all 10,000, restarts and rejected
//     steps included, the cached A x_k the next step derives from is BITWISE the product
//     A x_k computed afresh. So the derived A xbar and A dx differ from the three-product
//     path's by one step's rounding and nothing carries over: the drift after the window
//     is the amplification above, not accumulated error in the cache.
// The answer at the stopping tolerance is held above (AgreesWithTheThreeProductPath...) and
// on the device in test_pdhg_cuda_regression.cpp.

// The nine committed Netlib instances (data/netlib/reference.json), as in
// test_pdhg_cuda_regression.cpp.
const char* const kNetlibNine[] = {"afiro", "sc50a",   "sc105",    "sc50b", "adlittle",
                                   "blend", "share2b", "stocfor1", "israel"};

// The issue's figure.
constexpr Count kTraceIterations = 10000;
// The issue's agreement, held over the lockstep window (see above).
constexpr double kIterateAgreement = 1e-12;
constexpr Count kLockstepIterations = 20;

// Every accepted iterate, (x, y) concatenated, in the engine's scaled space, and the
// comparison of the cached A x_k with A x_k at each of them.
struct Trajectory {
  std::size_t width = 0;
  std::vector<double> points;
  bool in_order = true;
  Count cache_checked = 0;
  Count cache_mismatches = 0;
  Count first_mismatch = -1;
  double worst_cache = 0.0;  // max over k of ||cached - fresh||_inf / ||fresh||_inf
  Count count() const { return width == 0 ? 0 : static_cast<Count>(points.size() / width); }
};

void record_iterate(void* context, Count iteration, const double* x, std::size_t n,
                    const double* y, std::size_t m, const double* ax_cached,
                    const double* ax_fresh) {
  auto* trajectory = static_cast<Trajectory*>(context);
  trajectory->width = n + m;
  if (iteration != trajectory->count() + 1) trajectory->in_order = false;
  trajectory->points.insert(trajectory->points.end(), x, x + n);
  trajectory->points.insert(trajectory->points.end(), y, y + m);
  if (ax_cached == nullptr || ax_fresh == nullptr) return;
  ++trajectory->cache_checked;
  double difference = 0.0;
  double scale = 0.0;
  bool exact = true;
  for (std::size_t i = 0; i < m; ++i) {
    exact = exact && ax_cached[i] == ax_fresh[i];
    difference = std::max(difference, std::fabs(ax_cached[i] - ax_fresh[i]));
    scale = std::max(scale, std::fabs(ax_fresh[i]));
  }
  if (!exact) {
    ++trajectory->cache_mismatches;
    if (trajectory->first_mismatch < 0) trajectory->first_mismatch = iteration;
  }
  const double relative = scale > 0.0 ? difference / scale : difference;
  trajectory->worst_cache = std::max(trajectory->worst_cache, relative);
}

Options trace_options(bool two_matvec, bool gpu) {
  Options options = pdhg_options(two_matvec);
  options.set_bool("gpu", gpu);
  // The fixed iteration count is the comparison: the tightest tolerance the option accepts
  // keeps both runs going to it on every instance that does not converge to 1e-14 first.
  options.set_double("pdhg_tolerance", 1e-14);
  options.set_int("iteration_limit", static_cast<int>(kTraceIterations));
  return options;
}

Trajectory traced_solve(const Model& model, const Options& options, Solution* solution) {
  Trajectory trajectory;
  pdhg::IterateTraceHook& hook = pdhg::iterate_trace_for_testing();
  hook.callback = &record_iterate;
  hook.context = &trajectory;
  *solution = solve(model, options);
  hook = pdhg::IterateTraceHook{};
  return trajectory;
}

struct Drift {
  Count compared = 0;
  double lockstep = 0.0;  // max over the first kLockstepIterations
  double worst = 0.0;     // max over all compared, ||z_k - z'_k||_2 / ||z_k||_2
  Count worst_at = 0;
  Count first_over = -1;  // the first iterate over kIterateAgreement, -1 for none
  double at_one_thousand = std::numeric_limits<double>::quiet_NaN();
};

// Relative in the Euclidean norm of the whole iterate z_k = (x_k, y_k), against the
// reference path's own iterate. Per component would divide by entries that are exactly zero
// on a bound; per iterate is the statement "the two paths are at the same point".
Drift drift(const Trajectory& reference, const Trajectory& other) {
  Drift d;
  d.compared = std::min(reference.count(), other.count());
  const std::size_t w = reference.width;
  for (Count k = 0; k < d.compared; ++k) {
    double difference = 0.0;
    double norm = 0.0;
    for (std::size_t i = 0; i < w; ++i) {
      const double a = reference.points[static_cast<std::size_t>(k) * w + i];
      const double b = other.points[static_cast<std::size_t>(k) * w + i];
      difference += (a - b) * (a - b);
      norm += a * a;
    }
    const double relative =
        norm > 0.0 ? std::sqrt(difference / norm) : (difference > 0.0 ? 1.0 : 0.0);
    if (k < kLockstepIterations) d.lockstep = std::max(d.lockstep, relative);
    if (relative > d.worst) {
      d.worst = relative;
      d.worst_at = k + 1;
    }
    if (d.first_over < 0 && relative > kIterateAgreement) d.first_over = k + 1;
    if (k + 1 == 1000) d.at_one_thousand = relative;
  }
  return d;
}

std::string restarts_of(const Solution& s) {
  const std::size_t at = s.message.find(" restarts");
  if (at == std::string::npos) return "?";
  const std::size_t from = s.message.rfind(' ', at - 1);
  return s.message.substr(from + 1, at - from - 1);
}

void print_drift(const char* label, const char* name, const Solution& a, const Solution& b,
                 const Drift& d) {
  std::printf(
      "%-8s %-9s iterations %5lld/%5lld restarts %s/%s  first %lld: %.1e  first over %.0e at "
      "%lld  at 1000 %.1e  worst %.1e at %lld\n",
      label, name, static_cast<long long>(a.iterations), static_cast<long long>(b.iterations),
      restarts_of(a).c_str(), restarts_of(b).c_str(),
      static_cast<long long>(kLockstepIterations), d.lockstep, kIterateAgreement,
      static_cast<long long>(d.first_over), d.at_one_thousand, d.worst,
      static_cast<long long>(d.worst_at));
  std::fflush(stdout);
}

void print_cache(const char* label, const char* name, const Trajectory& t) {
  std::printf(
      "%-8s %-9s cache checked at %lld iterates, %lld not bitwise A x_k (first %lld), "
      "worst %.1e\n",
      label, name, static_cast<long long>(t.cache_checked),
      static_cast<long long>(t.cache_mismatches), static_cast<long long>(t.first_mismatch),
      t.worst_cache);
  std::fflush(stdout);
}

bool cuda_was_used(const Solution& s) {
  return s.algorithm.find("cuda") != std::string::npos ||
         s.algorithm.find("gpu") != std::string::npos;
}

void expect_lockstep_and_exact_cache(const char* name, const Trajectory& reference,
                                     const Trajectory& derived, const Solution& two,
                                     const Drift& d) {
  ASSERT_TRUE(reference.in_order && derived.in_order) << name;
  ASSERT_GE(reference.count(), kLockstepIterations) << name;
  ASSERT_GE(derived.count(), kLockstepIterations) << name;
  EXPECT_LE(d.lockstep, kIterateAgreement)
      << name << ": the first " << kLockstepIterations << " iterates differ by " << d.lockstep
      << " relative; first over " << kIterateAgreement << " at " << d.first_over;
  // Every iterate the two-product run accepted had its cache checked, and none differed.
  EXPECT_EQ(derived.cache_checked, two.iterations) << name;
  EXPECT_EQ(derived.cache_mismatches, 0)
      << name << ": the cached A x_k is not A x_k at " << derived.cache_mismatches
      << " iterates, first at " << derived.first_mismatch << ", worst " << derived.worst_cache;
  EXPECT_EQ(reference.cache_checked, 0) << name << ": the three-product path has no cache";
}

TEST(PdhgTwoMatvec, TheIteratesAgreeInLockstepAndTheCacheIsExactOverTenThousandIterations) {
  for (const char* name : kNetlibNine) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    Solution three;
    Solution two;
    const Trajectory reference = traced_solve(model, trace_options(false, false), &three);
    const Trajectory derived = traced_solve(model, trace_options(true, false), &two);
    EXPECT_EQ(reference.count(), three.iterations) << name;
    const Drift d = drift(reference, derived);
    print_drift("cpu", name, three, two, d);
    print_cache("cpu", name, derived);
    expect_lockstep_and_exact_cache(name, reference, derived, two, d);
  }
}

TEST(PdhgTwoMatvec, OnCudaTheIteratesAgreeInLockstepAndTheCacheIsExact) {
  Model probe_model;
  ASSERT_TRUE(io::read_model(netlib_path("afiro"), &probe_model).ok);
  Options probe = trace_options(false, true);
  probe.set_int("iteration_limit", 1);
  if (!cuda_was_used(solve(probe_model, probe))) {
    GTEST_SKIP() << "CUDA backend not in this build: skipped, not passed.";
  }
  for (const char* name : kNetlibNine) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    Solution three;
    Solution again;
    Solution two;
    const Trajectory reference = traced_solve(model, trace_options(false, true), &three);
    const Trajectory repeat = traced_solve(model, trace_options(false, true), &again);
    const Trajectory derived = traced_solve(model, trace_options(true, true), &two);
    ASSERT_TRUE(cuda_was_used(three) && cuda_was_used(again) && cuda_was_used(two)) << name;
    EXPECT_EQ(reference.count(), three.iterations) << name;
    // The control: the device's three-product path against itself (see above).
    print_drift("cuda3v3", name, three, again, drift(reference, repeat));
    const Drift d = drift(reference, derived);
    print_drift("cuda3v2", name, three, two, d);
    print_cache("cuda", name, derived);
    expect_lockstep_and_exact_cache(name, reference, derived, two, d);
  }
}

}  // namespace
}  // namespace sankhya
