// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a single-threaded LP solve is a function of its input, not of the run order (#909).
//
// dfl001 took 85,012 dual simplex iterations on some runs and 57,076 on others, from one
// binary and one input, under the 120 s limit the Netlib runners pass. The two counts were
// two routes through the scaled/unscaled portfolio in solve_with_scaling: the scaled attempt
// was given half the time limit (scaled_share=0.5), it needed 58 to 60 s of its 60 s share on
// that machine, and whichever run was a little slower (the first of each A/B pair) ran out of
// its share and was answered by the unscaled retry from the slack basis. Nothing in memory or
// in container order chose the path; the clock did, on a limit the whole solve fits inside.
//
// The tests below pin both halves of the claim. The same model solved twice in one process,
// with other solves in between, takes the same path to the bit. And a time limit the solve
// finishes well inside must not change the path either: only a limit that actually stops the
// solve may decide what it returns.

#include <algorithm>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Model netlib(const char* name) {
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" /
       "netlib" / (std::string(name) + ".mps"))
          .string();
  Model model;
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

/// The options the Netlib runners use, minus the limit: defaults, one thread, quiet.
Options defaults() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("threads", 1);
  return options;
}

void expect_same_path(const Solution& a, const Solution& b, const std::string& what) {
  EXPECT_EQ(a.status, b.status) << what;
  EXPECT_EQ(a.iterations, b.iterations) << what << ": a different path through the simplex";
  EXPECT_EQ(a.objective, b.objective) << what;
  EXPECT_EQ(a.col_value, b.col_value) << what;
  EXPECT_EQ(a.row_dual, b.row_dual) << what;
  EXPECT_EQ(a.algorithm, b.algorithm) << what;
}

TEST(RunOrderDeterminism, TheSameModelTwiceInOneProcessTakesTheSamePath) {
  // 25fv47 is a few thousand dual simplex iterations: long enough that any state carried
  // from one solve to the next (a cache, a static, uninitialised memory reused by the
  // allocator) would have room to steer it. The solves in between are other models of
  // other shapes, so the allocator hands the second 25fv47 solve different memory.
  const Model model = netlib("25fv47");
  ASSERT_GT(model.num_rows(), 0);
  const Solution first = solve(model, defaults());
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;

  for (const char* other : {"afiro", "sc205", "adlittle"}) {
    const Model between = netlib(other);
    ASSERT_GT(between.num_rows(), 0) << other;
    const Solution done = solve(between, defaults());
    EXPECT_EQ(done.status, SolveStatus::kOptimal) << other << ": " << done.message;
  }

  expect_same_path(first, solve(model, defaults()), "25fv47 after three other solves");
  expect_same_path(first, solve(model, defaults()), "25fv47 a third time");
}

TEST(RunOrderDeterminism, ATimeLimitTheSolveFinishesInsideDoesNotChangeItsPath) {
  // The #909 mechanism on a model small enough for every build. With no limit the scaled
  // attempt solves 25fv47 on its own and no route note is attached. Each limit below is 1.6
  // times what the unlimited solve just before it took; the solve fits inside it, so the
  // answer must be the unlimited one, iteration for iteration. Under the old split the
  // scaled attempt got half of it, 0.8 of what it needs, ran out, and the unscaled retry
  // either answered with another iteration count and a "route:" note or ran out too: the
  // dfl001 flip, made to happen on every run.
  const Model model = netlib("25fv47");
  ASSERT_GT(model.num_rows(), 0);
  const Solution reference = solve(model, defaults());
  ASSERT_EQ(reference.status, SolveStatus::kOptimal) << reference.message;
  ASSERT_EQ(reference.message.find("route:"), std::string::npos)
      << "the scaled attempt no longer solves this model on its own, so the test no longer "
         "exercises the split: "
      << reference.message;

  // The limit is set from a solve made just before it, and a run that is still 60% slower
  // than that one (a busy host) is stopped by the limit for real. A stopped solve may
  // differ, so it is tried again; only three stops in a row fail. The old split fails every
  // time: its scaled attempt never had enough.
  constexpr double kHeadroom = 1.6;
  Solution timed;
  double limit = 0.0;
  for (int attempt = 0; attempt < 3; ++attempt) {
    const Solution unlimited = solve(model, defaults());
    expect_same_path(reference, unlimited, "two unlimited solves");
    limit = kHeadroom * std::max(unlimited.solve_seconds, 1e-3);
    Options limited = defaults();
    limited.set_double("time_limit", limit);
    timed = solve(model, limited);
    if (timed.status != SolveStatus::kTimeLimit) break;
  }
  ASSERT_NE(timed.status, SolveStatus::kTimeLimit)
      << "three limits of " << kHeadroom << "x what the solve had just taken each stopped "
      << "it; the last, " << limit << " s: " << timed.message;
  EXPECT_EQ(timed.message.find("route:"), std::string::npos)
      << "a limit the solve fits inside sent it down another route: " << timed.message;
  expect_same_path(reference, timed, "a time limit the solve fits inside");
}

}  // namespace
}  // namespace sankhya
