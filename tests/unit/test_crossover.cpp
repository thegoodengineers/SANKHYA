// SPDX-License-Identifier: Apache-2.0
// SANKHYA - crossover from the interior point to a vertex (#219).
//
// THE CLAIM UNDER TEST IS THE VERTEX, not the guess. crossover_guess() is a heuristic and is
// checked only for its contract - m basic entries, every nonbasic one on a bound it has; the
// answer is checked the way any simplex answer is: optimal status, the published objective,
// exactly m basic entries in the reported basis, and every nonbasic entry sitting on its
// bound within the feasibility tolerance. A degenerate model - one where the interior point
// stops at the analytic centre of a whole optimal face - is where a crossover earns its
// keep, so that is the first model.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/status_guard.hpp"
#include "sankhya/io.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/crossover.hpp"
#include "simplex/simplex_core.hpp"

namespace sankhya {
namespace {

Options ipm_options(bool crossover) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("crossover", crossover);
  // Presolve stays ON: the vertex checks below count basic entries against m through
  // postsolve as well, which #341 made exact (every restored row adds one basic entry).
  return options;
}

Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

/// A vertex: exactly m basic entries, and every nonbasic entry on the bound its status names.
void expect_a_vertex(const Model& model, const Solution& s) {
  ASSERT_EQ(static_cast<Index>(s.col_status.size()), model.num_cols()) << s.message;
  ASSERT_EQ(static_cast<Index>(s.row_status.size()), model.num_rows()) << s.message;
  Index basic = 0;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double x = s.col_value[u];
    switch (s.col_status[u]) {
      case BasisStatus::kBasic: ++basic; break;
      case BasisStatus::kAtLower:
        EXPECT_NEAR(x, model.col_lower[u], tol::kPrimalFeasibility) << "column " << j;
        break;
      case BasisStatus::kAtUpper:
        EXPECT_NEAR(x, model.col_upper[u], tol::kPrimalFeasibility) << "column " << j;
        break;
      case BasisStatus::kFixed:
        EXPECT_NEAR(x, model.col_lower[u], tol::kPrimalFeasibility) << "column " << j;
        break;
      case BasisStatus::kNonbasicFree: EXPECT_NEAR(x, 0.0, tol::kPrimalFeasibility); break;
      case BasisStatus::kUnknown: ADD_FAILURE() << "column " << j << " has no status"; break;
    }
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    if (s.row_status[static_cast<std::size_t>(i)] == BasisStatus::kBasic) ++basic;
  }
  EXPECT_EQ(basic, model.num_rows()) << "a basis has exactly m basic entries";
}

TEST(Crossover, ADegenerateModelEndsAtAVertexNotAtTheAnalyticCentre) {
  // min x1 + x2 + x3 subject to x1 + x2 + x3 >= 3, each x in [0, 3]: every point of the
  // face x1 + x2 + x3 = 3 is optimal. The interior point stops at the centre (1, 1, 1);
  // a vertex has two coordinates at a bound.
  const Model model = make_lp({{1.0, 1.0, 1.0}}, {3.0}, {kInfinity}, {1.0, 1.0, 1.0},
                              {0.0, 0.0, 0.0}, {3.0, 3.0, 3.0});
  const Solution interior = solve(model, ipm_options(false));
  ASSERT_EQ(interior.status, SolveStatus::kOptimal) << interior.message;
  for (const BasisStatus status : interior.col_status) {
    EXPECT_EQ(status, BasisStatus::kUnknown) << "the interior point reports no basis";
  }

  const CrossoverGuess guess = crossover_guess(model, interior);
  EXPECT_EQ(guess.basic, model.num_rows());

  const Solution vertex = solve(model, ipm_options(true));
  ASSERT_EQ(vertex.status, SolveStatus::kOptimal) << vertex.message;
  EXPECT_NEAR(vertex.objective, 3.0, 1e-7);
  EXPECT_NE(vertex.algorithm.find("crossover"), std::string::npos) << vertex.algorithm;
  EXPECT_NE(vertex.message.find("crossover:"), std::string::npos) << vertex.message;
  expect_a_vertex(model, vertex);
  Index at_bound = 0;
  for (const double x : vertex.col_value) {
    if (std::fabs(x) <= tol::kPrimalFeasibility ||
        std::fabs(x - 3.0) <= tol::kPrimalFeasibility) {
      ++at_bound;
    }
  }
  EXPECT_GE(at_bound, 2) << "a vertex of this face has at least two coordinates at a bound";
}

/// Holds the interior point for `seconds` after it returns, for one test (#576).
class InteriorPointTakesLonger {
 public:
  explicit InteriorPointTakesLonger(double seconds) {
    interior_point_extra_seconds_for_testing() = seconds;
  }
  ~InteriorPointTakesLonger() { interior_point_extra_seconds_for_testing() = 0.0; }
  InteriorPointTakesLonger(const InteriorPointTakesLonger&) = delete;
  InteriorPointTakesLonger& operator=(const InteriorPointTakesLonger&) = delete;
};

TEST(Crossover, GetsWhatTheTimeLimitHasLeftAfterTheInteriorPointNotLessTwice) {
  // #576: solve() handed the crossover a time_limit already net of the elapsed seconds, and
  // crossover_to_vertex() subtracts them again, so an interior point that used more than half
  // the limit left the crossover nothing: irish-electricity finished at 170 s of 300 and was
  // told "no time left for crossover" with 130 s left. Here the interior point is held for
  // 1.5 s of a 2.5 s limit; about 1.0 s is left, and counted twice it was -0.5 s.
  const Model model = make_lp({{1.0, 1.0, 1.0}}, {3.0}, {kInfinity}, {1.0, 1.0, 1.0},
                              {0.0, 0.0, 0.0}, {3.0, 3.0, 3.0});
  Options options = ipm_options(true);
  options.set_double("time_limit", 2.5);
  const InteriorPointTakesLonger held(1.5);
  const Solution vertex = solve(model, options);
  ASSERT_EQ(vertex.status, SolveStatus::kOptimal) << vertex.message;
  EXPECT_EQ(vertex.message.find("no time left for crossover"), std::string::npos)
      << vertex.message;
  EXPECT_NE(vertex.algorithm.find("crossover"), std::string::npos) << vertex.message;
  expect_a_vertex(model, vertex);
}

TEST(Crossover, ThePushStopsAtTheTimeLimit) {
  // #417: run_push() armed its deadline before the time limit was read, so the push had no
  // deadline and ran every superbasic to a bound whatever was left: 244 s against 143 s on
  // rmine15, and only the primal loop after it noticed the limit. Here the push itself is
  // given a limit that is gone at once, from the interior point's answer on a degenerate
  // Netlib model; it must stop with no pivot made. Without the fix it made them all and
  // reported the limit after them.
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib/scsd8.mps")
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  ASSERT_TRUE(read.ok) << path << ": " << read.error;
  Logger quiet(nullptr);
  const Solution interior = ipm::solve_ipm(model, ipm_options(false), quiet);
  ASSERT_EQ(interior.status, SolveStatus::kOptimal) << interior.message;
  const CrossoverGuess guess = crossover_guess(model, interior);
  ASSERT_EQ(guess.basic, model.num_rows());
  WarmStart warm;
  warm.col_status = guess.col_status;
  warm.row_status = guess.row_status;

  // The same push with no limit makes pivots, so a stop at zero is the deadline's doing.
  Options unlimited = ipm_options(true);
  unlimited.set_string("algorithm", "dual-simplex");
  detail::Simplex free_run(model, unlimited, quiet, nullptr);
  const Solution pushed = free_run.run_push(warm, interior.col_value, interior.row_activity);
  ASSERT_GT(pushed.iterations, 0) << pushed.message;

  Options limited = unlimited;
  limited.set_double("time_limit", 1e-9);
  detail::Simplex timed(model, limited, quiet, nullptr);
  const Solution stopped = timed.run_push(warm, interior.col_value, interior.row_activity);
  EXPECT_EQ(stopped.status, SolveStatus::kTimeLimit) << stopped.message;
  EXPECT_EQ(stopped.iterations, 0) << stopped.message;
}

/// A dense m x m equality system with every column in [0, 1], and the interior point's
/// answer at x = 1/2: every column strictly inside its bounds and every row an equality, so
/// the guess is all m columns and its rank repair is one factorization of a dense m x m
/// matrix, with partial pivoting. The entries are a fixed congruential sequence in [-1, 1],
/// so the matrix and the work are the same on every run (#951).
struct DenseCentre {
  Model model;
  Solution interior;
};

DenseCentre dense_centre(Index m) {
  DenseCentre made;
  Model& model = made.model;
  const auto size = static_cast<std::size_t>(m);
  model.col_cost.assign(size, 1.0);
  model.col_lower.assign(size, 0.0);
  model.col_upper.assign(size, 1.0);
  model.col_type.assign(size, VarType::kContinuous);
  model.matrix.reset(m, m);
  std::vector<double> activity(size, 0.0);
  std::uint64_t state = 951;
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < m; ++j) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      const double v = static_cast<double>(state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
      model.matrix.add_entry(i, j, v);
      activity[static_cast<std::size_t>(i)] += 0.5 * v;
    }
  }
  model.matrix.finalize();
  model.row_lower = activity;
  model.row_upper = activity;
  model.hessian.reset(m, m);
  model.hessian.finalize();

  Solution& interior = made.interior;
  interior.allocate_for(model);
  interior.status = SolveStatus::kOptimal;
  interior.algorithm = "ipm";
  interior.col_value.assign(size, 0.5);
  interior.row_activity = activity;
  return made;
}

TEST(Crossover, TheBasisGuessStopsAtTheTimeLimit) {
  // #951: crossover_to_vertex() read the time limit only after crossover_guess() had built
  // its guess, and the guess's rank repair factorizes it with partial pivoting, with no
  // deadline. On the refinery year at T = 365 under a 20 s limit the interior point
  // converged at 18.6 s and the result reported 26.3 s, "no time left for crossover". Here
  // the guess is one dense 1,500 x 1,500 factorization, seconds of work, under a limit of
  // 0.2 s: the crossover must give up near the limit and hand back the interior point's
  // answer as it came. Without the fix the whole factorization ran first.
  const DenseCentre centre = dense_centre(1500);
  Options options = ipm_options(true);
  constexpr double kLimit = 0.2;
  options.set_double("time_limit", kLimit);
  Logger quiet(nullptr);
  const Timer timer;
  const Solution answer =
      crossover_to_vertex(centre.model, centre.interior, options, quiet, nullptr, timer);
  const double spent = timer.elapsed_seconds();
  // 3 s, not the limit plus a little: building the dense guess before the first deadline
  // check is itself a second under the sanitizer build (1.08 s there), and the old code
  // took 10.2 s in Release, so the bound still separates the two.
  EXPECT_LT(spent, 3.0) << "the crossover ran past its time limit: " << answer.message;
  EXPECT_EQ(answer.status, SolveStatus::kOptimal) << answer.message;
  EXPECT_EQ(answer.algorithm, "ipm") << "no vertex was claimed";
  EXPECT_NE(answer.message.find("the interior point's answer stands"), std::string::npos)
      << answer.message;
  for (const double x : answer.col_value) ASSERT_EQ(x, 0.5) << "the point is not touched";
}

TEST(Crossover, TheBasisGuessReadsItsDeadline) {
  // The contract under the test above: a deadline that has already fired stops the rank
  // repair before its first factorization, and the guess says it was stopped.
  const DenseCentre centre = dense_centre(200);
  const CrossoverGuess stopped =
      crossover_guess(centre.model, centre.interior, [] { return true; });
  EXPECT_TRUE(stopped.stopped);
  const CrossoverGuess whole = crossover_guess(centre.model, centre.interior);
  EXPECT_FALSE(whole.stopped);
  EXPECT_EQ(whole.basic, centre.model.num_rows());
}

TEST(Crossover, OffLeavesTheInteriorPointsAnswerAlone) {
  const Model model = make_lp({{1.0, 1.0}, {1.0, -1.0}}, {1.0, -kInfinity}, {kInfinity, 1.0},
                              {1.0, 2.0}, {0.0, 0.0}, {kInfinity, kInfinity});
  const Solution s = solve(model, ipm_options(false));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.algorithm.find("crossover"), std::string::npos) << s.algorithm;
  for (const BasisStatus status : s.col_status) EXPECT_EQ(status, BasisStatus::kUnknown);
}

TEST(Crossover, NetlibInstancesReachAVertexAtThePublishedObjective) {
  // The committed small set, through ipm + crossover: the objective the published table
  // gives, with a basis, from a start that is already optimal to 1e-8. The pivot count is
  // in the message; the claim that it is small is what the Netlib CSV records (#219).
  struct Instance {
    const char* name;
    double published;
  };
  const Instance instances[] = {{"afiro", -464.75314286},   {"sc50b", -70.0},
                                {"share2b", -415.73224074}, {"stocfor1", -41131.976219},
                                {"adlittle", 225494.96316}, {"israel", -896644.82186}};
  for (const Instance& instance : instances) {
    Model model;
    const std::string path =
        (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
         "data/netlib" / (std::string(instance.name) + ".mps"))
            .string();
    const io::ReadResult read = io::read_model(path, &model);
    ASSERT_TRUE(read.ok) << path << ": " << read.error;
    const Solution s = solve(model, ipm_options(true));
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << instance.name << ": " << s.message;
    EXPECT_NEAR(s.objective, instance.published,
                1e-6 * std::max(1.0, std::fabs(instance.published)))
        << instance.name;
    EXPECT_NE(s.algorithm.find("crossover"), std::string::npos)
        << instance.name << ": " << s.message;
    expect_a_vertex(model, s);
  }
}

}  // namespace
}  // namespace sankhya
