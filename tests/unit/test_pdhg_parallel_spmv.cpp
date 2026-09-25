// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the row-parallel A x in PDHG (#487).
//
// The claim is determinism, not speed: with pdhg_parallel_spmv on, every output
// entry of A x is written by exactly one thread in a fixed static partition, so
// the whole iteration is bitwise the same at 1, 2 and 4 threads. The serial
// product is a column scatter in a different summation order, so against it the
// parallel path is held to rounding, not to the bit: the same status and
// iteration count, the objective to 1e-9 relative. Both checks run on the
// committed Netlib instances, the same set the CUDA regression uses.
//
// pdhg_parallel_updates (#487, src/pdhg/pdhg_parallel.hpp) moves the rest of the
// iteration onto the workers: the projected updates and the step rule's sums, the
// sums in fixed chunks of tol::kPdhgParallelChunk entries. The last two tests hold
// it to the same two standards, on a generated model large enough that every sum
// spans several chunks (a Netlib instance fits in one, which would test nothing).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

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

TEST(PdhgParallelSpmv, SerialIsReproducible) {
  // Two back-to-back serial solves must produce the exact same bits: same
  // objective, same iteration count, same column values.  This is the baseline
  // against which the parallel path is compared in
  // AgreesWithTheSerialProductToRounding below.
  for (const char* name : kInstances) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution a = solve(model, pdhg_options(false, 1));
    const Solution b = solve(model, pdhg_options(false, 1));
    EXPECT_EQ(a.status, b.status) << name;
    EXPECT_EQ(a.iterations, b.iterations) << name;
    EXPECT_EQ(a.objective, b.objective) << name;
    ASSERT_EQ(a.col_value.size(), b.col_value.size()) << name;
    for (std::size_t j = 0; j < a.col_value.size(); ++j) {
      ASSERT_EQ(a.col_value[j], b.col_value[j]) << name << " column " << j;
    }
  }
}

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
    // A different summation order can move a restart decision by an ulp on a
    // knife edge; it has not on these six, and if it ever does the iteration
    // count says so here.
    EXPECT_EQ(serial.iterations, parallel.iterations) << name;
    EXPECT_NEAR(serial.objective, parallel.objective,
                1e-9 * std::max(1.0, std::fabs(serial.objective)))
        << name;
  }
}

/// A feasible, bounded LP with three entries per column in rows j, j + m/3 and j + 2m/3
/// (mod m), so no two coincide: maximise sum x subject to row sums <= b, 0 <= x <= 10.
/// Deterministic in its size alone.
Model spread_lp(Index rows, Index cols) {
  Model model;
  const auto n = static_cast<std::size_t>(cols);
  const auto m = static_cast<std::size_t>(rows);
  model.col_cost.assign(n, -1.0);
  model.col_lower.assign(n, 0.0);
  model.col_upper.assign(n, 10.0);
  model.col_type.assign(n, VarType::kContinuous);
  model.row_lower.assign(m, -kInfinity);
  model.row_upper.assign(m, 0.0);
  model.matrix.reset(rows, cols);
  for (Index i = 0; i < rows; ++i) {
    model.row_upper[static_cast<std::size_t>(i)] = 20.0 + static_cast<double>(i % 11);
  }
  for (Index j = 0; j < cols; ++j) {
    for (Index k = 0; k < 3; ++k) {
      const Index i = (j + k * (rows / 3)) % rows;
      model.matrix.add_entry(i, j, 1.0 + static_cast<double>((j * 7 + k * 3) % 5));
    }
  }
  model.matrix.finalize();
  model.hessian.reset(cols, cols);
  model.hessian.finalize();
  return model;
}

Options updates_options(int threads, bool two_matvec) {
  Options options = pdhg_options(true, threads);
  options.set_bool("pdhg_parallel_updates", true);
  options.set_bool("pdhg_two_matvec", two_matvec);
  // A fixed iteration count: the claim is about the arithmetic, every step of which the
  // bitwise comparison covers whether or not the solve has converged by then. The tolerance
  // is far below the other tests' 1e-6 so the comparison covers more of the run: on an L4
  // box this model stopped at 640 iterations at 1e-6 and at 880 at this one.
  options.set_double("pdhg_tolerance", 1e-13);
  options.set_int("iteration_limit", 1500);
  return options;
}

TEST(PdhgParallelSpmv, TheVectorUpdatesGiveTheSameBitsAtOneTwoFourAndEightThreads) {
  // 6,000 rows and 9,000 columns: six and nine chunks, so the chunk order and not only the
  // chunk contents is exercised.
  const Model model = spread_lp(6000, 9000);
  ASSERT_GT(model.num_rows(), 4 * tol::kPdhgParallelChunk);
  for (const bool two_matvec : {false, true}) {
    const Solution one = solve(model, updates_options(1, two_matvec));
    ASSERT_NE(one.status, SolveStatus::kNotSolved) << one.message;
    ASSERT_GT(one.iterations, 100) << one.message;
    std::printf("spread_lp 6000 x 9000%s: %s after %lld iterations at every thread count\n",
                two_matvec ? ", two-mat-vec" : "", to_string(one.status),
                static_cast<long long>(one.iterations));
    for (const int threads : {2, 4, 8}) {
      const Solution other = solve(model, updates_options(threads, two_matvec));
      const std::string what = std::to_string(threads) + " threads" +
                               (two_matvec ? ", two-mat-vec" : ", three products");
      EXPECT_EQ(one.status, other.status) << what;
      EXPECT_EQ(one.iterations, other.iterations) << what;
      EXPECT_EQ(one.objective, other.objective) << what;
      EXPECT_EQ(one.col_value, other.col_value) << what << ": the primal points differ";
      EXPECT_EQ(one.row_dual, other.row_dual) << what << ": the duals differ";
    }
  }
}

TEST(PdhgParallelSpmv, TheVectorUpdatesAgreeWithTheSerialLoopsToRounding) {
  // Against the serial loops the three sums are taken in a different order, so the
  // standard is rounding, and rounding in a step-rule sum is enough to move a restart by an
  // evaluation: from there the two runs take different paths (sc50a on an L4 box: 12,880
  // iterations serial, 14,440 parallel). What can be asked of two different paths is what
  // is asked of any two runs of this method: the same status, and where both stopped on the
  // tolerance, the objective to that tolerance (1e-6 here, the one both were asked for).
  // Where both ran out of the 20,000-iteration budget the two points are wherever their
  // paths had got to and no closeness is implied, so only the status is compared. The
  // counts are printed so a drift is visible even where it is allowed.
  int compared = 0;
  for (const char* name : kInstances) {
    Model model;
    ASSERT_TRUE(io::read_model(netlib_path(name), &model).ok) << name;
    const Solution serial = solve(model, pdhg_options(true, 4));
    Options parallel_options = pdhg_options(true, 4);
    parallel_options.set_bool("pdhg_parallel_updates", true);
    const Solution parallel = solve(model, parallel_options);
    EXPECT_EQ(serial.status, parallel.status) << name << ": " << parallel.message;
    const bool both_optimal =
        serial.status == SolveStatus::kOptimal && parallel.status == SolveStatus::kOptimal;
    if (both_optimal) {
      ++compared;
      EXPECT_NEAR(serial.objective, parallel.objective,
                  1e-6 * std::max(1.0, std::fabs(serial.objective)))
          << name;
    }
    std::printf("%s: %s, serial updates %lld iterations, parallel updates %lld\n", name,
                to_string(parallel.status), static_cast<long long>(serial.iterations),
                static_cast<long long>(parallel.iterations));
  }
  EXPECT_GE(compared, 2) << "afiro and sc50a converge inside the budget on both paths";
}

}  // namespace
}  // namespace sankhya
