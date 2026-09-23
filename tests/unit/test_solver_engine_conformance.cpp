// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the contract every registered SolverEngine is held to (#297).
//
// test_solver_engine.cpp tests the registry, the selector and each wrapper by name. This file
// is the other half: tests written once against the INTERFACE and run over whatever
// SolverRegistry::builtin() holds, so an engine registered tomorrow is held to them without
// anyone adding a line here. For every engine and every class its capabilities() accepts:
//
//   interface     a solve through SolverEngine::solve() returns a point whose objective,
//                 row activities, feasibility and integrality this file re-measures itself
//   capability    a flag that is set is backed by what the answer carries (duals, a basis,
//                 a certificate, a checkpoint), and a checkpoint asked of an engine that does
//                 not declare one writes nothing
//   resource      a zero budget (iterations or nodes, time) and an interrupt each end the
//                 solve with the limit status and the reason, never a numerical error
//   determinism   deterministic=true twice gives the same status, numbers and effort
//   checkpoint    an engine that declares it stops, resumes and reaches the uninterrupted
//                 optimum through its own solve()
//   concurrency   two engines solving at once on their own models, options and loggers give
//                 the answers they give alone (docs/ARCHITECTURE.md section 13)
//
// The engines' own algorithms are tested in their own files; what is asserted here is that
// the common contract holds for each of them.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/certificate.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/tolerances.hpp"
#include "support/temp_file.hpp"

#include "solver_engine/solver_engine.hpp"
#include "solver_engine/solver_engine_dispatch.hpp"
#include "solver_engine/solver_registry.hpp"

namespace sankhya::engine {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

Model afiro() {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib/afiro.mps")
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

/// A correlated 0-1 knapsack: a real tree of a few hundred nodes
/// (tests/unit/test_checkpoint.cpp uses the same family for the same reason).
Model knapsack(int columns, int seed) {
  Model m;
  const auto n = static_cast<Index>(columns);
  const auto un = static_cast<std::size_t>(n);
  m.sense = ObjSense::kMaximize;
  m.col_cost.resize(un);
  m.col_lower.assign(un, 0.0);
  m.col_upper.assign(un, 1.0);
  m.col_type.assign(un, VarType::kInteger);
  m.matrix.reset(1, n);
  double total = 0.0;
  for (Index j = 0; j < n; ++j) {
    const double w = 20.0 + static_cast<double>((j * 37 + seed * 11) % 51);
    m.col_cost[static_cast<std::size_t>(j)] = w + 10.0;
    m.matrix.add_entry(0, j, w);
    total += w;
  }
  m.matrix.finalize();
  m.row_lower = {-kInfinity};
  m.row_upper = {std::floor(total / 2.0)};
  m.hessian.reset(n, n);
  m.hessian.finalize();
  return m;
}

/// min x^2 + y^2 - 4x - 2y  s.t.  x + y <= 2,  0 <= x, y <= 10.  The unconstrained minimum
/// (2, 1) violates the row; the optimum is its projection (1.5, 0.5), objective -4.5. With x
/// integer it is -4, reached at (1, 1) and at (2, 0), so an MIQP answer of -4.5 is the
/// relaxation.
Model small_qp(bool x_integer) {
  Model m;
  m.resize_columns(2);
  m.resize_rows(1);
  m.col_cost = {-4.0, -2.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {10.0, 10.0};
  if (x_integer) m.col_type = {VarType::kInteger, VarType::kContinuous};
  m.row_lower = {-kInfinity};
  m.row_upper = {2.0};
  m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.finalize();
  m.hessian.reset(2, 2);
  m.hessian.add_entry(0, 0, 2.0);
  m.hessian.add_entry(1, 1, 2.0);
  m.hessian.finalize();
  return m;
}

Model model_of(ProblemClass problem_class) {
  switch (problem_class) {
    case ProblemClass::kLp: return afiro();
    case ProblemClass::kMilp: return knapsack(16, 3);
    case ProblemClass::kQp: return small_qp(false);
    case ProblemClass::kMiqp: return small_qp(true);
  }
  return {};
}

/// Every (engine, class) pair the registry declares, in registration order.
std::vector<std::pair<const SolverEngine*, ProblemClass>> declared_pairs() {
  std::vector<std::pair<const SolverEngine*, ProblemClass>> pairs;
  const SolverRegistry& registry = SolverRegistry::builtin();
  for (const std::string& name : registry.names()) {
    const SolverEngine* engine = registry.find(name);
    for (const ProblemClass problem_class :
         {ProblemClass::kLp, ProblemClass::kMilp, ProblemClass::kQp, ProblemClass::kMiqp}) {
      if (engine->capabilities().accepts(problem_class))
        pairs.emplace_back(engine, problem_class);
    }
  }
  return pairs;
}

std::string label(const SolverEngine* engine, ProblemClass problem_class) {
  return engine->name() + " on " + to_string(problem_class);
}

/// What this file measures about a point, from the model and col_value alone: no field the
/// engine filled in other than the point is read.
struct Measured {
  double objective = 0.0;
  double primal_violation = 0.0;  ///< max absolute violation over rows and column bounds
  double integrality_violation = 0.0;
  double worst_activity_error = 0.0;  ///< |row_activity reported - A x measured|
};

Measured measure(const Model& model, const Solution& solution) {
  Measured out;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  const std::vector<double>& x = solution.col_value;
  std::vector<double> activity(m, 0.0);
  const std::vector<Index>& starts = model.matrix.column_starts();
  const std::vector<Index>& rows = model.matrix.row_indices();
  const std::vector<double>& values = model.matrix.values();
  double objective = model.objective_offset;
  for (std::size_t j = 0; j < n; ++j) {
    objective += model.col_cost[j] * x[j];
    for (auto k = static_cast<std::size_t>(starts[j]);
         k < static_cast<std::size_t>(starts[j + 1]); ++k) {
      activity[static_cast<std::size_t>(rows[k])] += values[k] * x[j];
    }
    out.primal_violation =
        std::max({out.primal_violation, model.col_lower[j] - x[j], x[j] - model.col_upper[j]});
    if (!model.col_type.empty() && model.col_type[j] != VarType::kContinuous) {
      out.integrality_violation =
          std::max(out.integrality_violation, std::fabs(x[j] - std::round(x[j])));
    }
  }
  // 0.5 x'Qx over the stored lower triangle: a diagonal entry once, an off-diagonal one twice.
  if (model.hessian.num_nonzeros() > 0) {
    const std::vector<Index>& hs = model.hessian.column_starts();
    const std::vector<Index>& hr = model.hessian.row_indices();
    const std::vector<double>& hv = model.hessian.values();
    for (std::size_t j = 0; j < n; ++j) {
      for (auto k = static_cast<std::size_t>(hs[j]); k < static_cast<std::size_t>(hs[j + 1]);
           ++k) {
        const auto i = static_cast<std::size_t>(hr[k]);
        objective += (i == j ? 0.5 : 1.0) * hv[k] * x[i] * x[j];
      }
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    out.primal_violation = std::max({out.primal_violation, model.row_lower[i] - activity[i],
                                     activity[i] - model.row_upper[i]});
    if (solution.row_activity.size() == m) {
      out.worst_activity_error =
          std::max(out.worst_activity_error, std::fabs(solution.row_activity[i] - activity[i]));
    }
  }
  out.objective = objective;
  return out;
}

bool is_limit(SolveStatus status) {
  return status == SolveStatus::kIterationLimit || status == SolveStatus::kTimeLimit ||
         status == SolveStatus::kNodeLimit || status == SolveStatus::kInterrupted ||
         status == SolveStatus::kFeasible;
}

std::string read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

// =========================================================================================
// Interface and verification
// =========================================================================================

TEST(EngineConformance, EveryDeclaredClassHasAtLeastOneEngine) {
  for (const ProblemClass problem_class :
       {ProblemClass::kLp, ProblemClass::kMilp, ProblemClass::kQp, ProblemClass::kMiqp}) {
    EXPECT_FALSE(SolverRegistry::builtin().candidates(model_of(problem_class)).empty())
        << to_string(problem_class);
  }
}

TEST(EngineConformance, EveryEngineAnswersEveryClassItDeclaresWithAPointThatMeasuresUp) {
  // Every check below is EXPECT, or a plain `if` that reports with ADD_FAILURE and CONTINUES
  // to the next (engine, class) pair, rather than ASSERT. An ASSERT_* failure here would
  // `return` out of the whole test function - not just this iteration - so one engine failing
  // one class would silently skip verifying every pair after it. The three checks that stay
  // as an `if` (rather than a bare EXPECT) are the ones a later line would be unsafe to run
  // past: measure() below indexes solution.col_value by column and reads model.num_rows()
  // worth of row_activity, so a point that was never claimed, or came back the wrong size,
  // must skip the rest of THIS iteration - it must not skip the rest of the test.
  for (const auto& [engine, problem_class] : declared_pairs()) {
    SCOPED_TRACE(label(engine, problem_class));
    const Model model = model_of(problem_class);
    if (classify(model) != problem_class) {
      ADD_FAILURE() << "model_of() built a model classify() reads as "
                    << to_string(classify(model)) << ", not " << to_string(problem_class);
      continue;
    }
    if (!engine->supports(model)) {
      ADD_FAILURE() << "the engine declares " << to_string(problem_class)
                    << " but supports() refuses the model built for it";
      continue;
    }
    Logger logger(nullptr);
    const Solution solution = engine->solve(model, quiet(), logger);

    if (!claims_a_point(solution.status)) {
      ADD_FAILURE() << to_string(solution.status) << ": " << solution.message;
      continue;
    }
    EXPECT_FALSE(solution.algorithm.empty());
    if (solution.col_value.size() != static_cast<std::size_t>(model.num_cols()) ||
        solution.row_activity.size() != static_cast<std::size_t>(model.num_rows())) {
      ADD_FAILURE() << "wrong-sized answer: col_value " << solution.col_value.size()
                    << " (want " << model.num_cols() << "), row_activity "
                    << solution.row_activity.size() << " (want " << model.num_rows() << ")";
      continue;
    }
    EXPECT_TRUE(std::isfinite(solution.objective));

    // The answer's own numbers against this file's measurement of the point it returned.
    const Measured measured = measure(model, solution);
    const double scale = std::max(1.0, std::fabs(measured.objective));
    EXPECT_NEAR(solution.objective, measured.objective, 1e-9 * scale)
        << "the reported objective is not the objective of the reported point";
    EXPECT_LE(measured.worst_activity_error, 1e-9 * scale)
        << "row_activity is not A x of the reported point";
    EXPECT_LE(measured.primal_violation, solution.primal_infeasibility + 1e-12)
        << "the point is less feasible than the answer says";
    if (solution.status == SolveStatus::kOptimal) {
      // A first-order engine may stop short of the project's standard; the status then says
      // kFeasible rather than kOptimal, and only kOptimal is held to it here.
      EXPECT_LE(measured.primal_violation, tol::kPrimalFeasibility * scale);
    }
    if (problem_class == ProblemClass::kMilp || problem_class == ProblemClass::kMiqp) {
      EXPECT_LE(measured.integrality_violation, tol::kIntegrality)
          << "a mixed-integer answer that is not integral is the relaxation";
    }
  }
}

TEST(EngineConformance, EveryEngineReachesTheKnownOptimumOfTheSmallModels) {
  // afiro's optimum is Netlib's published -464.753142857; the knapsack's is whatever the
  // plain dispatcher proves, and the QP pair's is worked out on small_qp() above.
  Logger logger(nullptr);
  const double knapsack_optimum = sankhya::solve(knapsack(16, 3), quiet()).objective;
  for (const auto& [engine, problem_class] : declared_pairs()) {
    SCOPED_TRACE(label(engine, problem_class));
    const Solution solution = engine->solve(model_of(problem_class), quiet(), logger);
    ASSERT_TRUE(claims_a_point(solution.status)) << solution.message;
    double expected = 0.0;
    double tolerance = 1e-6;
    switch (problem_class) {
      case ProblemClass::kLp:
        expected = -464.753142857;
        tolerance = engine->capabilities().supports_basis ? 1e-6 : 1e-2;  // a vertex or not
        break;
      case ProblemClass::kMilp: expected = knapsack_optimum; break;
      case ProblemClass::kQp:
        expected = -4.5;
        tolerance = 1e-4;
        break;
      case ProblemClass::kMiqp:
        expected = -4.0;
        tolerance = 1e-4;
        break;
    }
    EXPECT_NEAR(solution.objective, expected, tolerance * std::max(1.0, std::fabs(expected)));
  }
}

// =========================================================================================
// Capabilities: a flag that is set is a flag the answer backs
// =========================================================================================

TEST(EngineCapabilities, DeclaredDualsAndBasisAreOnTheAnswer) {
  Logger logger(nullptr);
  for (const auto& [engine, problem_class] : declared_pairs()) {
    SCOPED_TRACE(label(engine, problem_class));
    const EngineCapabilities caps = engine->capabilities();
    const Model model = model_of(problem_class);
    const Solution solution = engine->solve(model, quiet(), logger);
    ASSERT_TRUE(claims_a_point(solution.status)) << solution.message;
    if (caps.supports_duals) {
      ASSERT_EQ(solution.row_dual.size(), static_cast<std::size_t>(model.num_rows()));
      ASSERT_EQ(solution.col_dual.size(), static_cast<std::size_t>(model.num_cols()));
      for (const double y : solution.row_dual) EXPECT_TRUE(std::isfinite(y));
      EXPECT_TRUE(std::any_of(solution.row_dual.begin(), solution.row_dual.end(), [](double y) {
        return y != 0.0;
      })) << "afiro has binding rows; a declared dual vector of zeros is no dual";
    }
    if (caps.supports_basis) {
      ASSERT_EQ(solution.col_status.size(), static_cast<std::size_t>(model.num_cols()));
      ASSERT_EQ(solution.row_status.size(), static_cast<std::size_t>(model.num_rows()));
      const auto basic = std::count(solution.col_status.begin(), solution.col_status.end(),
                                    BasisStatus::kBasic) +
                         std::count(solution.row_status.begin(), solution.row_status.end(),
                                    BasisStatus::kBasic);
      EXPECT_EQ(basic, model.num_rows()) << "a basis has one basic variable per row";
    }
  }
}

TEST(EngineCapabilities, DeclaredCertificatesProveAnInfeasibleLp) {
  // x + y >= 5 with both columns in [0, 1]: infeasible, and a single Farkas multiplier on the
  // one row proves it.
  Model model;
  model.resize_columns(2);
  model.resize_rows(1);
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {1.0, 1.0};
  model.row_lower = {5.0};
  model.row_upper = {kInfinity};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();

  Logger logger(nullptr);
  int declared = 0;
  for (const auto& [engine, problem_class] : declared_pairs()) {
    if (problem_class != ProblemClass::kLp || !engine->capabilities().supports_certificates) {
      continue;
    }
    ++declared;
    SCOPED_TRACE(engine->name());
    const Solution solution = engine->solve(model, quiet(), logger);
    ASSERT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
    ASSERT_FALSE(solution.farkas_dual.empty())
        << "a declared certificate that was not produced";
    EXPECT_TRUE(farkas_proves_infeasible(model, solution.farkas_dual));
  }
  EXPECT_GT(declared, 0) << "no engine declares certificates; this test has nothing to hold";
}

TEST(EngineCapabilities, AnEngineThatDoesNotDeclareCheckpointsWritesNone) {
  for (const auto& [engine, problem_class] : declared_pairs()) {
    if (engine->capabilities().supports_checkpoint) continue;
    SCOPED_TRACE(label(engine, problem_class));
    testing::TempFile file("", ".chk");
    Options options = quiet();
    options.set_string("checkpoint", file.path());
    options.set_int("iteration_limit", 1);  // stopped by a limit, when a checkpoint is due
    Logger logger(nullptr);
    (void)engine->solve(model_of(problem_class), options, logger);
    EXPECT_TRUE(read_all(file.path()).empty());
  }
}

// =========================================================================================
// Resources: a limit is a limit, through every engine's own solve()
// =========================================================================================

TEST(EngineResources, AZeroCountBudgetStopsEveryEngineWithTheLimitStatus) {
  for (const auto& [engine, problem_class] : declared_pairs()) {
    SCOPED_TRACE(label(engine, problem_class));
    const bool tree =
        problem_class == ProblemClass::kMilp || problem_class == ProblemClass::kMiqp;
    Options options = quiet();
    options.set_int(tree ? "node_limit" : "iteration_limit", 0);
    Logger logger(nullptr);
    const Solution stopped = engine->solve(model_of(problem_class), options, logger);
    EXPECT_EQ(stopped.stopped_by, tree ? LimitReason::kNodes : LimitReason::kIterations)
        << to_string(stopped.status) << ": " << stopped.message;
    EXPECT_TRUE(is_limit(stopped.status)) << to_string(stopped.status);
    EXPECT_NE(stopped.status, SolveStatus::kNumericalError);
  }
}

TEST(EngineResources, AZeroTimeLimitStopsEveryEngineOnTheClock) {
  for (const auto& [engine, problem_class] : declared_pairs()) {
    SCOPED_TRACE(label(engine, problem_class));
    Options options = quiet();
    options.set_double("time_limit", 0.0);
    Logger logger(nullptr);
    const Solution stopped = engine->solve(model_of(problem_class), options, logger);
    EXPECT_EQ(stopped.stopped_by, LimitReason::kTime)
        << to_string(stopped.status) << ": " << stopped.message;
    EXPECT_TRUE(is_limit(stopped.status)) << to_string(stopped.status);
  }
}

TEST(EngineResources, AnInterruptBeforeTheFirstSafePointStopsEveryEngineThatDeclaresIt) {
  for (const auto& [engine, problem_class] : declared_pairs()) {
    if (!engine->capabilities().supports_interrupt) continue;
    SCOPED_TRACE(label(engine, problem_class));
    SolveControl control;
    control.interrupt();
    Logger logger(nullptr);
    const Solution stopped = engine->solve(model_of(problem_class), quiet(), logger, &control);
    EXPECT_EQ(stopped.status, SolveStatus::kInterrupted) << stopped.message;
    EXPECT_EQ(stopped.stopped_by, LimitReason::kInterrupt);
  }
}

// =========================================================================================
// Determinism: the same engine twice under deterministic=true
// =========================================================================================

TEST(EngineDeterminism, EveryEngineRepeatsItselfUnderDeterministicMode) {
  for (const auto& [engine, problem_class] : declared_pairs()) {
    if (!engine->capabilities().supports_deterministic_mode) continue;
    SCOPED_TRACE(label(engine, problem_class));
    Options options = quiet();
    options.set_bool("deterministic", true);
    const Model model = model_of(problem_class);
    Logger logger(nullptr);
    const Solution first = engine->solve(model, options, logger);
    const Solution second = engine->solve(model, options, logger);
    ASSERT_TRUE(claims_a_point(first.status)) << first.message;
    EXPECT_EQ(first.status, second.status);
    EXPECT_EQ(first.algorithm, second.algorithm);
    EXPECT_EQ(first.objective, second.objective);
    EXPECT_EQ(first.col_value, second.col_value);
    EXPECT_EQ(first.row_dual, second.row_dual);
    EXPECT_EQ(first.iterations, second.iterations);
    EXPECT_EQ(first.nodes, second.nodes);
  }
}

// =========================================================================================
// Checkpoint: stop, save, resume, finish, through the engine's own solve()
// =========================================================================================

TEST(EngineCheckpoint, AnEngineThatDeclaresCheckpointsResumesToTheUninterruptedOptimum) {
  int declared = 0;
  for (const auto& [engine, problem_class] : declared_pairs()) {
    if (!engine->capabilities().supports_checkpoint || problem_class != ProblemClass::kMilp) {
      continue;
    }
    ++declared;
    for (int seed = 0; seed < 4; ++seed) {
      SCOPED_TRACE(engine->name() + " seed " + std::to_string(seed));
      const Model model = knapsack(18, seed);
      Options plain = quiet();
      // A tree that is still open when the node limit falls: with the objective known to be
      // integral these knapsacks close at the root (tests/unit/test_checkpoint.cpp).
      plain.set_bool("mip_objective_integrality", false);
      Logger logger(nullptr);
      const Solution whole = engine->solve(model, plain, logger);
      ASSERT_EQ(whole.status, SolveStatus::kOptimal) << whole.message;
      ASSERT_GT(whole.nodes, 20) << "the model should need a real tree";

      testing::TempFile file("", ".chk");
      Options first = plain;
      first.set_string("checkpoint", file.path());
      first.set_int("node_limit", std::max<Count>(2, whole.nodes / 10));
      const Solution stopped = engine->solve(model, first, logger);
      ASSERT_EQ(stopped.stopped_by, LimitReason::kNodes) << stopped.message;
      ASSERT_FALSE(read_all(file.path()).empty()) << "a limit-stopped search writes its state";

      Options second = plain;
      second.set_string("resume", file.path());
      const Solution resumed = engine->solve(model, second, logger);
      ASSERT_EQ(resumed.status, SolveStatus::kOptimal) << resumed.message;
      EXPECT_NEAR(resumed.objective, whole.objective, 1e-9);
      EXPECT_GE(resumed.nodes, stopped.nodes) << "the count continues from the checkpoint";
    }
  }
  EXPECT_GT(declared, 0) << "no engine declares checkpoints; this test has nothing to hold";
}

// =========================================================================================
// Concurrency: what docs/ARCHITECTURE.md section 13 promises
// =========================================================================================

TEST(EngineConcurrency, EnginesSolvingAtOnceGiveTheAnswersTheyGiveAlone) {
  // Every (engine, class) pair at once, each on a thread with its own Model, Options, Logger
  // and Solution, all sharing the one registry and the engine instances in it.
  const auto pairs = declared_pairs();
  std::vector<Solution> alone(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    Logger logger(nullptr);
    alone[k] = pairs[k].first->solve(model_of(pairs[k].second), quiet(), logger);
  }
  std::vector<Solution> together(pairs.size());
  std::vector<std::thread> threads;
  threads.reserve(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    threads.emplace_back([&, k] {
      Logger logger(nullptr);
      together[k] = pairs[k].first->solve(model_of(pairs[k].second), quiet(), logger);
    });
  }
  for (std::thread& t : threads) t.join();
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    SCOPED_TRACE(label(pairs[k].first, pairs[k].second));
    EXPECT_EQ(together[k].status, alone[k].status);
    EXPECT_EQ(together[k].objective, alone[k].objective);
    EXPECT_EQ(together[k].col_value, alone[k].col_value);
    EXPECT_EQ(together[k].iterations, alone[k].iterations);
  }
}

// =========================================================================================
// The registry's own entry point: validate, select, presolve, run, postsolve, guard
// =========================================================================================

TEST(EngineConformance, TheRegistryEntryPointAgreesWithSolveOnEveryClass) {
  for (const ProblemClass problem_class :
       {ProblemClass::kLp, ProblemClass::kMilp, ProblemClass::kQp, ProblemClass::kMiqp}) {
    SCOPED_TRACE(to_string(problem_class));
    const Model model = model_of(problem_class);
    Logger logger(nullptr);
    const Solution via_registry = solve(SolverRegistry::builtin(), model, quiet(), logger);
    const Solution via_solve = sankhya::solve(model, quiet());
    ASSERT_TRUE(claims_a_point(via_registry.status)) << via_registry.message;
    EXPECT_EQ(via_registry.status, via_solve.status);
    EXPECT_NEAR(via_registry.objective, via_solve.objective,
                1e-6 * std::max(1.0, std::fabs(via_solve.objective)));
    const Measured measured = measure(model, via_registry);
    EXPECT_LE(measured.primal_violation, via_registry.primal_infeasibility + 1e-12);
  }
}

}  // namespace
}  // namespace sankhya::engine
