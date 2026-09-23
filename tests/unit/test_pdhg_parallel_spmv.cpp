// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the row-parallel A x in PDHG (#487).
//
// The claim is determinism, not speed: with pdhg_parallel_spmv on, every output entry of
// A x is written by exactly one thread in a fixed static partition, so the whole iteration
// is bitwise the same at 1, 2 and 4 threads. The serial product is a column scatter in a
// different summation order, so against it the parallel path is held to rounding, not to
// the bit: the same status and iteration count, the objective to 1e-9 relative. Both
// checks run on the committed Netlib instances, the same set the CUDA regression uses.

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options pdhg_options(bool parallel, int threads) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", "pdhg");
  options.set_bool("pdhg_polish", false);
  options.set_double("pdhg_tolerance", 1e-6);
  options.set_int("iteration_limit", 20000);
  options.set_bool("pdhg_parallel_spmv", parallel);
  options.set_int("threads", threads);
  return options;
}

std::string netlib_path(const char* name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
          "data/netlib" / (std::string(name) + ".mps"))
      .string();
}

const char* const kInstances[] = {"afiro", "adlittle", "sc50a", "sc105", "blend", "israel"};

TEST(PdhgParallelSpmv, TheSameBitsAtOneTwoAndFourThreads) {
  for (const char* name : kInstances) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution one = solve(model, pdhg_options(true, 1));
    const Solution two = solve(model, pdhg_options(true, 2));
    const Solution four = solve(model, pdhg_options(true, 4));
    EXPECT_EQ(one.status, two.status) << name;
    EXPECT_EQ(one.status, four.status) << name;
    EXPECT_EQ(one.iterations, two.iterations) << name;
    EXPECT_EQ(one.iterations, four.iterations) << name;
    // Bitwise: the double compares equal, not nearly equal.
    EXPECT_EQ(one.objective, two.objective) << name;
    EXPECT_EQ(one.objective, four.objective) << name;
    ASSERT_EQ(one.col_value.size(), four.col_value.size()) << name;
    for (std::size_t j = 0; j < one.col_value.size(); ++j) {
      ASSERT_EQ(one.col_value[j], four.col_value[j]) << name << " column " << j;
    }
  }
}

TEST(PdhgParallelSpmv, AgreesWithTheSerialProductToRounding) {
  for (const char* name : kInstances) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution serial = solve(model, pdhg_options(false, 1));
    const Solution parallel = solve(model, pdhg_options(true, 4));
    EXPECT_EQ(serial.status, parallel.status) << name << ": " << parallel.message;
    // A different summation order can move a restart decision by an ulp on a knife edge;
    // it has not on these six, and if it ever does the iteration count says so here.
    EXPECT_EQ(serial.iterations, parallel.iterations) << name;
    EXPECT_NEAR(serial.objective, parallel.objective,
                1e-9 * std::max(1.0, std::fabs(serial.objective)))
        << name;
  }
}

}  // namespace
}  // namespace sankhya
