// SPDX-License-Identifier: Apache-2.0
// SANKHYA - what a branch and bound stopped by a limit reports (#505).
//
// MIPLIB `ej` is one equality row over three general integer columns,
//
//     min x0   s.t.   31013 x0 - 41014 x1 - 51015 x2 = 0,   x0 >= 1,  x1, x2 >= 0,  integer,
//
// optimum 25508, and a search that runs for minutes without an integer point. In
// bench/results/miplib-078cb24.csv it came back time_limit with objective 46624.61 - the LP
// relaxation of whichever node the clock stopped - and the independent verifier rejected the
// .sol on integrality. A limited search must carry a verified incumbent or no point at all;
// these tests hold the solver to that on `ej` itself (built here rather than read, since
// data/miplib is fetched, not tracked), through the sequential and the parallel search, and
// through the .sol writer.

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

constexpr double kEjOptimum = 25508.0;  // MIPLIB 2017's published optimum for ej

Model ej() {
  Model model;
  model.name = "ej";
  model.col_names = {"x0", "x1", "x2"};
  model.row_names = {"c1"};
  model.col_cost = {1.0, 0.0, 0.0};
  model.col_lower = {1.0, 0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity, kInfinity};
  model.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
  model.matrix.reset(1, 3);
  model.matrix.add_entry(0, 0, 31013.0);
  model.matrix.add_entry(0, 1, -41014.0);
  model.matrix.add_entry(0, 2, -51015.0);
  model.matrix.finalize();
  model.row_lower = {0.0};
  model.row_upper = {0.0};
  model.hessian.reset(3, 3);
  model.hessian.finalize();
  return model;
}

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

/// The contract, in whichever form the solve took: an incumbent that is integral, satisfies
/// the row exactly as integers do and has objective x0 - or nothing. Returns true when it was
/// the "nothing" form, which is the one these tests exist for.
bool holds_an_incumbent_or_nothing(const Solution& s) {
  if (claims_a_point(s)) {
    EXPECT_EQ(s.col_value.size(), 3U) << s.message;
    if (s.col_value.size() != 3U) return false;
    for (const double v : s.col_value) {
      EXPECT_LE(std::fabs(v - std::round(v)), tol::kIntegrality)
          << "a limited search claimed a fractional point: " << s.message;
    }
    // On the rounded values the row is integer arithmetic, exact in double far past ej's
    // optimum, so it must hold exactly.
    const double row = 31013.0 * std::round(s.col_value[0]) -
                       41014.0 * std::round(s.col_value[1]) -
                       51015.0 * std::round(s.col_value[2]);
    EXPECT_EQ(row, 0.0) << "a limited search claimed a point off the row: " << s.message;
    EXPECT_DOUBLE_EQ(s.objective, s.col_value[0]);
    EXPECT_GE(s.objective, kEjOptimum - 1e-6) << "better than the published optimum";
    return false;
  }
  EXPECT_TRUE(s.col_value.empty()) << "no point is claimed, so none may be carried";
  EXPECT_TRUE(s.row_activity.empty());
  EXPECT_TRUE(std::isinf(s.objective) && s.objective > 0.0)
      << "objective " << s.objective << " on a search with no incumbent: " << s.message;
  EXPECT_EQ(s.integrality_violation, 0.0);
  EXPECT_EQ(s.primal_infeasibility, 0.0) << "measured on a point nobody claimed: " << s.message;
  EXPECT_TRUE(std::isinf(s.relative_gap));
  EXPECT_TRUE(std::isfinite(s.dual_bound)) << "the open nodes still prove a bound";
  EXPECT_LE(s.dual_bound, kEjOptimum) << "a bound above the optimum proves something false";
  return true;
}

TEST(MipLimitReporting, EjStoppedByTheClockCarriesNoRelaxation) {
  // The benchmark's own path: default options, presolve on, a time limit.
  const Model model = ej();
  Options options = quiet();
  options.set_double("time_limit", 0.5);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kTimeLimit) << s.message;
  EXPECT_EQ(s.stopped_by, LimitReason::kTime);
  EXPECT_GT(s.nodes, 0);
  EXPECT_TRUE(holds_an_incumbent_or_nothing(s))
      << "ej found an integer point in half a second, so this test no longer reaches the "
         "case it was written for; lower the limit or switch off the heuristics";
}

TEST(MipLimitReporting, EjStoppedByANodeLimitWithoutPresolveCarriesNoRelaxation) {
  // Deterministic: no clock, no heuristics, no presolve, so the node LP point reaches the
  // report directly rather than through postsolve.
  const Model model = ej();
  Options options = quiet();
  options.set_bool("presolve", false);
  options.set_bool("mip_heuristics", false);
  options.set_int("node_limit", 200);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kNodeLimit) << s.message;
  EXPECT_TRUE(holds_an_incumbent_or_nothing(s));
}

TEST(MipLimitReporting, EjStoppedInTheParallelSearchCarriesNoRelaxation) {
  const Model model = ej();
  Options options = quiet();
  options.set_int("mip_threads", 2);
  options.set_double("time_limit", 0.5);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kTimeLimit) << s.message;
  EXPECT_TRUE(holds_an_incumbent_or_nothing(s));
}

TEST(MipLimitReporting, TheSolutionFileOfALimitThatFoundNothingHasNoPoint) {
  // What tools/verify_solution.py reads. It used to get the relaxation (rejected on
  // integrality), and before that a block of zeros; a limit that found nothing now writes
  // its status, its bound and an infinite objective, and no columns or rows section.
  const Model model = ej();
  Options options = quiet();
  options.set_bool("mip_heuristics", false);
  options.set_int("node_limit", 50);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kNodeLimit) << s.message;
  ASSERT_TRUE(holds_an_incumbent_or_nothing(s));

  testing::TempFile out("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(out.path(), model, s, options, &error)) << error;
  const std::string text = read_file(out.path());
  EXPECT_NE(text.find("status node_limit"), std::string::npos) << text;
  EXPECT_NE(text.find("objective inf"), std::string::npos) << text;
  EXPECT_EQ(text.find("begin columns"), std::string::npos) << text;
  EXPECT_EQ(text.find("begin rows"), std::string::npos) << text;
}

TEST(MipLimitReporting, ALimitThatHoldsAPointStillWritesIt) {
  // The other side of claims_a_point(const Solution&): an LP stopped by a limit hands back
  // its iterate, finite objective and all, and the file keeps its columns block.
  Solution s;
  s.status = SolveStatus::kTimeLimit;
  s.objective = 3.0;
  s.col_value = {3.0, 0.0, 0.0};
  s.row_activity = {93039.0};
  EXPECT_TRUE(claims_a_point(s));

  Solution nothing;
  nothing.status = SolveStatus::kTimeLimit;
  nothing.objective = kInfinity;
  EXPECT_FALSE(claims_a_point(nothing));
  EXPECT_TRUE(claims_a_point(nothing.status)) << "the status alone still says a limit";

  // An empty point with a FINITE objective is not "nothing found": that is a claim with its
  // point missing, and it stays a claim so the checker can say so.
  Solution broken;
  broken.status = SolveStatus::kTimeLimit;
  broken.objective = 3.0;
  EXPECT_TRUE(claims_a_point(broken));

  const Model model = ej();
  testing::TempFile out("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(out.path(), model, s, &error)) << error;
  EXPECT_NE(read_file(out.path()).find("begin columns 3"), std::string::npos);
}

}  // namespace
}  // namespace sankhya
