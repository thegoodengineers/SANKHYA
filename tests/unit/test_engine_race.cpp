// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the verifier-gated engine race (#476) and the in-process KKT check it rests on.
//
// THE PROPERTY THAT MATTERS: the first engine to finish is not trusted for finishing first. A
// wrong answer injected into the engine that finishes first must be rejected by the check,
// must not stop the others, and the race must still return a right answer - one that passes
// the same check afresh here. Every other test is about the machinery around that: the check
// itself, the fixed order of deterministic mode, the fallback when nothing passes, an engine
// that runs out of memory, and the option being off by default.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <new>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "core/engine_race.hpp"
#include "core/kkt_check.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

constexpr double kAfiroOptimum = -464.75314286;

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

Options quiet(const char* algorithm) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", algorithm);
  return options;
}

Options racing() {
  Options options = quiet("auto");
  options.set_bool("engine_race", true);
  // The PDHG lane runs to the standard here: under the relative stop (#1163, on by default)
  // it stops at a feasible point the check cannot accept, and the simplex lanes win.
  options.set_bool("pdhg_relative_stop", false);
  return options;
}

/// The hooks are process-wide; every test that sets them clears them on the way out, however
/// it leaves.
struct HooksGuard {
  HooksGuard() = default;
  HooksGuard(const HooksGuard&) = delete;
  HooksGuard& operator=(const HooksGuard&) = delete;
  ~HooksGuard() { engine_race_hooks_for_testing() = EngineRaceTestHooks{}; }
};

/// A wrong answer that still CLAIMS optimal: every value moved by one unit and the objective
/// recomputed to match, so the claim is internally tidy and wrong about the model.
void corrupt(const Model& model, Solution* answer) {
  for (double& x : answer->col_value) x += 1.0;
  answer->objective = model.evaluate_objective(answer->col_value.data());
}

TEST(KktCheck, AcceptsTheDualSimplexOptimumOfAfiro) {
  const Model model = netlib("afiro");
  const Solution s = solve(model, quiet("dual-simplex"));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  const KktVerdict verdict = check_lp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
}

TEST(KktCheck, AcceptsTheInteriorPointWithCrossoverOnAMaximizeModel) {
  // Sign conventions: a maximize model's duals are reported in its own sense, and the check
  // works in minimize space as the verifier does.
  Model model;
  model.sense = ObjSense::kMaximize;
  model.col_cost = {3.0, 5.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {-kInfinity, -kInfinity, -kInfinity};
  model.row_upper = {4.0, 12.0, 18.0};
  model.matrix.reset(3, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(1, 1, 2.0);
  model.matrix.add_entry(2, 0, 3.0);
  model.matrix.add_entry(2, 1, 2.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  for (const char* algorithm : {"dual-simplex", "ipm", "simplex"}) {
    const Solution s = solve(model, quiet(algorithm));
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << algorithm << ": " << s.message;
    EXPECT_NEAR(s.objective, 36.0, 1e-9);
    const KktVerdict verdict = check_lp_optimality(model, s);
    EXPECT_TRUE(verdict.passed) << algorithm << ": " << verdict.check << ": " << verdict.detail;
  }
}

TEST(KktCheck, NamesTheCheckAWrongAnswerFails) {
  const Model model = netlib("afiro");
  const Solution good = solve(model, quiet("dual-simplex"));
  ASSERT_EQ(good.status, SolveStatus::kOptimal) << good.message;

  Solution moved = good;
  corrupt(model, &moved);
  const KktVerdict primal = check_lp_optimality(model, moved);
  EXPECT_FALSE(primal.passed);
  EXPECT_TRUE(primal.check == "row activity" || primal.check == "column bounds")
      << primal.check << ": " << primal.detail;

  Solution objective = good;
  objective.objective += 1.0;
  EXPECT_EQ(check_lp_optimality(model, objective).check, "objective");

  Solution duals = good;
  for (double& y : duals.row_dual) y += 1.0;
  EXPECT_EQ(check_lp_optimality(model, duals).check, "reduced costs");

  // Duals moved consistently - y and d = c - A^T y together - leave the reduced costs tidy
  // and break optimality itself: a sign condition, complementarity or strong duality.
  Solution consistent = good;
  consistent.row_dual[0] += 10.0;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index p = 0; p < column.size; ++p) {
      if (column.rows[p] == 0)
        consistent.col_dual[static_cast<std::size_t>(j)] -= 10.0 * column.values[p];
    }
  }
  const KktVerdict dual = check_lp_optimality(model, consistent);
  EXPECT_FALSE(dual.passed) << "a dual moved off the optimum was accepted";
  EXPECT_NE(dual.check, "reduced costs") << dual.detail;

  Solution feasible = good;
  feasible.status = SolveStatus::kFeasible;
  EXPECT_EQ(check_lp_optimality(model, feasible).check, "verdict");
}

TEST(EngineRace, AnInjectedWrongAnswerThatFinishesFirstDoesNotWin) {
  // The acceptance test of #476. The dual simplex finishes first - the other two are held
  // back 300 ms - and its answer is replaced by a wrong one that still claims optimal. The
  // race must reject it, keep the others running, and return an answer that passes the check.
  const Model model = netlib("afiro");
  HooksGuard guard;
  EngineRaceTestHooks& hooks = engine_race_hooks_for_testing();
  hooks.before_engine = [](const std::string& engine) {
    if (engine != "dual-simplex") std::this_thread::sleep_for(std::chrono::milliseconds(300));
  };
  hooks.after_engine = [&model](const std::string& engine, Solution* answer) {
    if (engine == "dual-simplex" && answer->status == SolveStatus::kOptimal) {
      corrupt(model, answer);
    }
  };
  const Solution s = solve(model, racing());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, kAfiroOptimum, 1e-6 * std::fabs(kAfiroOptimum)) << s.message;
  EXPECT_EQ(s.engine_rule, "race");
  EXPECT_EQ(s.engine_reason.find("engine_race: dual-simplex"), std::string::npos)
      << s.engine_reason;
  EXPECT_NE(s.message.find("dual-simplex: optimal"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("REJECTED"), std::string::npos) << s.message;
  const KktVerdict verdict = check_lp_optimality(model, s);
  EXPECT_TRUE(verdict.passed) << verdict.check << ": " << verdict.detail;
}

TEST(EngineRace, ACorrectAnswerWinsAndEveryEnginesOutcomeIsLogged) {
  // Nothing injected: whichever engine is accepted first wins, the answer passes the check,
  // and every engine's outcome is in the message.
  const Model model = netlib("afiro");
  const Solution s = solve(model, racing());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, kAfiroOptimum, 1e-6 * std::fabs(kAfiroOptimum));
  EXPECT_EQ(s.engine_rule, "race");
  for (const char* engine : {"dual-simplex:", "ipm:", "pdhg"}) {
    EXPECT_NE(s.message.find(engine), std::string::npos) << engine << " in " << s.message;
  }
  EXPECT_NE(s.message.find("the winner"), std::string::npos) << s.message;
  EXPECT_TRUE(check_lp_optimality(model, s).passed);
}

TEST(EngineRace, MatchesTheRuleTablesAnswerOnTheNetlibSmallSet) {
  // The race may not report a worse answer than the single engine the rule table would have
  // run: on every instance of the committed small set both reach optimal, agree on the
  // objective to the pass tolerance of bench/runners/netlib.py, and the race's answer passes
  // the check afresh here. Whichever engine won carries its own name; a basis comes only from
  // the dual simplex or from crossover, so a PDHG win carries none.
  for (const char* name : {"afiro", "sc50a", "sc50b", "sc105", "adlittle", "share2b", "blend",
                           "stocfor1", "israel"}) {
    const Model model = netlib(name);
    const Solution rule = solve(model, quiet("auto"));
    const Solution race = solve(model, racing());
    ASSERT_EQ(rule.status, SolveStatus::kOptimal) << name << ": " << rule.message;
    ASSERT_EQ(race.status, SolveStatus::kOptimal) << name << ": " << race.message;
    EXPECT_NEAR(race.objective, rule.objective, 1e-6 * std::max(1.0, std::fabs(rule.objective)))
        << name;
    EXPECT_EQ(race.engine_rule, "race") << name;
    EXPECT_FALSE(race.algorithm.empty()) << name;
    EXPECT_TRUE(check_lp_optimality(model, race).passed) << name << ": " << race.message;
    if (!race.col_status.empty()) {
      EXPECT_TRUE(race.algorithm.find("simplex") != std::string::npos ||
                  race.algorithm.find("crossover") != std::string::npos)
          << name << ": a basis from " << race.algorithm;
    }
  }
}

TEST(EngineRace, APdhgWinCarriesNoBasis) {
  // #1027: the small-set test above once saw, under load, a PDHG win on afiro whose answer
  // came with col_status filled. There was no race in the recording: PDHG's Solution is
  // allocate_for()'d, so its statuses were a full vector of kUnknown, and postsolve carried
  // them through and labelled a few restored entries on top. A PDHG win carries no basis
  // (Solution's contract), and that is forced here: the simplex and the interior point are
  // held until the PDHG arm has finished, so PDHG's answer is the first one judged.
  const Model model = netlib("afiro");
  HooksGuard guard;
  std::atomic<bool> pdhg_finished{false};
  EngineRaceTestHooks& hooks = engine_race_hooks_for_testing();
  hooks.before_engine = [&pdhg_finished](const std::string& engine) {
    if (engine.rfind("pdhg", 0) == 0) return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!pdhg_finished.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // PDHG's answer is judged and recorded right after its after_engine hook; this arm still
    // has a presolve and a solve ahead of it, and a little slack on top.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  };
  hooks.after_engine = [&pdhg_finished](const std::string& engine, Solution*) {
    if (engine.rfind("pdhg", 0) == 0) pdhg_finished = true;
  };
  const Solution s = solve(model, racing());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  ASSERT_EQ(s.algorithm.rfind("pdhg", 0), 0u) << s.algorithm << ": " << s.message;
  EXPECT_NE(s.message.find("engine race (#476), pdhg"), std::string::npos) << s.message;
  EXPECT_TRUE(s.col_status.empty()) << "a basis from " << s.algorithm;
  EXPECT_TRUE(s.row_status.empty()) << "a basis from " << s.algorithm;
  EXPECT_TRUE(check_lp_optimality(model, s).passed) << s.message;
}

TEST(EngineRace, DeterministicModeRunsAFixedOrderAndReproduces) {
  const Model model = netlib("afiro");
  Options options = racing();
  options.set_bool("deterministic", true);
  const Solution first = solve(model, options);
  const Solution second = solve(model, options);
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  // The dual simplex is first in the order and its answer passes, so nothing else runs.
  EXPECT_NE(first.engine_reason.find("engine_race: dual-simplex's answer passed"),
            std::string::npos)
      << first.engine_reason;
  EXPECT_NE(first.message.find("ipm: not run"), std::string::npos) << first.message;
  EXPECT_EQ(first.objective, second.objective);
  EXPECT_EQ(first.iterations, second.iterations);
  EXPECT_EQ(first.col_value, second.col_value);
  EXPECT_EQ(first.row_dual, second.row_dual);
}

TEST(EngineRace, DeterministicModeMovesOnWhenTheFirstAnswerIsRejected) {
  const Model model = netlib("afiro");
  HooksGuard guard;
  engine_race_hooks_for_testing().after_engine = [&model](const std::string& engine,
                                                          Solution* answer) {
    if (engine == "dual-simplex" && answer->status == SolveStatus::kOptimal) {
      corrupt(model, answer);
    }
  };
  Options options = racing();
  options.set_bool("deterministic", true);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NE(s.engine_reason.find("engine_race: ipm's answer passed"), std::string::npos)
      << s.engine_reason;
  EXPECT_TRUE(check_lp_optimality(model, s).passed);
}

TEST(EngineRace, WithNoAnswerAcceptedTheRuleEnginesAnswerIsNotReportedOptimal) {
  const Model model = netlib("afiro");
  HooksGuard guard;
  engine_race_hooks_for_testing().after_engine = [&model](const std::string&,
                                                          Solution* answer) {
    if (answer->status == SolveStatus::kOptimal) corrupt(model, answer);
  };
  const Solution s = solve(model, racing());
  EXPECT_NE(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NE(s.message.find("no answer passed its check"), std::string::npos) << s.message;
  EXPECT_NE(s.engine_reason.find("no answer passed its check"), std::string::npos)
      << s.engine_reason;
}

TEST(EngineRace, AnEngineThatRunsOutOfMemoryLosesAndTheRaceGoesOn) {
  // #437: an exhausted allocation in one engine is that engine's declined status, not the
  // process's end and not the race's.
  const Model model = netlib("afiro");
  HooksGuard guard;
  engine_race_hooks_for_testing().before_engine = [](const std::string& engine) {
    if (engine == "dual-simplex") throw std::bad_alloc();
  };
  const Solution s = solve(model, racing());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NE(s.message.find("dual-simplex: numerical_error"), std::string::npos) << s.message;
  EXPECT_TRUE(check_lp_optimality(model, s).passed);
}

TEST(EngineRace, TheLosersAreCancelledOnceTheWinnerIsAccepted) {
  // The cancellation path (#535): PDHG is held back until the dual simplex has been
  // accepted, so it starts with the race's control already interrupted and must come back
  // interrupted, logged as cancelled, with the winner's answer untouched. The race is in
  // order on one core and then has no losers to cancel.
  if (std::thread::hardware_concurrency() <= 1)
    GTEST_SKIP() << "one core: the race is in order";
  const Model model = netlib("afiro");
  HooksGuard guard;
  engine_race_hooks_for_testing().before_engine = [](const std::string& engine) {
    if (engine.rfind("pdhg", 0) == 0) std::this_thread::sleep_for(std::chrono::seconds(1));
  };
  const Solution s = solve(model, racing());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, kAfiroOptimum, 1e-6 * std::fabs(kAfiroOptimum));
  EXPECT_NE(s.message.find("cancelled at"), std::string::npos) << s.message;
  EXPECT_NE(s.message.find("the winner"), std::string::npos) << s.message;
  EXPECT_TRUE(check_lp_optimality(model, s).passed);
}

TEST(EngineRace, IsOffByDefault) {
  const Model model = netlib("afiro");
  const Solution s = solve(model, quiet("auto"));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NE(s.engine_rule, "race");
  EXPECT_EQ(s.message.find("engine race"), std::string::npos) << s.message;
}

}  // namespace
}  // namespace sankhya
