// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the SolverEngine / SolverRegistry / SolverSelector layer (#297).
//
// Every engine wrapper tested here calls the SAME free-function engine solve()'s dispatcher
// calls (src/core/solve.cpp); the assertions below are therefore about the adapter - name,
// capabilities, the supports() gate, and that solve() on the wrapper reaches the same answer
// the underlying algorithm is already known to reach - not a second copy of the algorithm's
// own correctness tests.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "solver_engine/builtin_engines.hpp"
#include "solver_engine/solver_engine.hpp"
#include "solver_engine/solver_registry.hpp"
#include "solver_engine/solver_selector.hpp"

namespace sankhya::engine {
namespace {

Options quiet_options() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// min -x - y  s.t.  x + y <= 4, 0 <= x,y <= 10. Optimum at any point on x+y=4 with the
/// simplex convention landing on a vertex; the LP tests below only need the objective, -4.
Model tiny_lp() {
  Model model;
  model.resize_columns(2);
  model.resize_rows(1);
  model.col_cost = {-1.0, -1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.row_lower = {-kInfinity};
  model.row_upper = {4.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  return model;
}

/// min -x  s.t.  2x <= 5, 0 <= x <= 10, x integer. Relaxation optimum is x=2.5 (objective
/// -2.5); the integer optimum is x=2 (objective -2).
Model tiny_milp() {
  Model model;
  model.resize_columns(1);
  model.resize_rows(1);
  model.col_cost = {-1.0};
  model.col_lower = {0.0};
  model.col_upper = {10.0};
  model.col_type = {VarType::kInteger};
  model.row_lower = {-kInfinity};
  model.row_upper = {5.0};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 2.0);
  model.matrix.finalize();
  return model;
}

/// min x^2 - 4x  s.t.  0 <= x <= 10 (no rows). Unconstrained minimum is at x=2, objective -4.
Model tiny_qp() {
  Model model;
  model.resize_columns(1);
  model.col_cost = {-4.0};
  model.col_lower = {0.0};
  model.col_upper = {10.0};
  model.matrix.reset(0, 1);
  model.matrix.finalize();
  model.hessian.reset(1, 1);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.finalize();
  return model;
}

// =========================================================================================
// SolverRegistry

TEST(SolverRegistry, BuiltinRegistersEveryEngineSolveDispatchesTo) {
  const SolverRegistry& registry = SolverRegistry::builtin();
  const std::vector<std::string> names = registry.names();
  for (const char* expected :
       {"simplex", "dual-simplex", "pdhg", "ipm", "convex-qp", "branch-and-bound"}) {
    EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end())
        << "missing engine: " << expected;
  }
}

TEST(SolverRegistry, FindReturnsNullForAnUnregisteredName) {
  EXPECT_EQ(SolverRegistry::builtin().find("not-an-engine"), nullptr);
}

/// A stand-in engine, only for RegisterEngineReplacesAnExistingNameRatherThanDuplicatingIt:
/// it never solves anything, it just occupies the name "simplex" so the test can tell
/// whether registering it replaced the built-in simplex or sat beside it.
class FakeSimplexEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "simplex"; }
  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    return caps;
  }
  [[nodiscard]] Solution solve(const Model& model, const Options&, Logger&,
                               SolveControl*) const override {
    Solution solution;
    solution.allocate_for(model);
    solution.status = SolveStatus::kNotSolved;
    solution.algorithm = "fake-simplex";
    return solution;
  }
};

TEST(SolverRegistry, RegisterEngineReplacesAnExistingNameRatherThanDuplicatingIt) {
  SolverRegistry registry;
  register_builtin_engines(registry);
  const std::size_t before = registry.names().size();
  registry.register_engine(std::make_shared<FakeSimplexEngine>());
  EXPECT_EQ(registry.names().size(), before);
  Logger logger(nullptr);
  const Solution solution = registry.find("simplex")->solve(tiny_lp(), quiet_options(), logger);
  EXPECT_EQ(solution.algorithm, "fake-simplex");
}

TEST(SolverRegistry, CandidatesOnlyListsEnginesWhoseCapabilitiesAcceptTheClass) {
  const SolverRegistry& registry = SolverRegistry::builtin();
  const std::vector<const SolverEngine*> lp_candidates = registry.candidates(tiny_lp());
  for (const SolverEngine* candidate : lp_candidates) {
    EXPECT_TRUE(candidate->capabilities().lp) << candidate->name() << " should not be an LP "
                                              << "candidate";
  }
  const std::vector<const SolverEngine*> milp_candidates = registry.candidates(tiny_milp());
  ASSERT_FALSE(milp_candidates.empty());
  for (const SolverEngine* candidate : milp_candidates) {
    EXPECT_TRUE(candidate->capabilities().milp);
  }
}

// =========================================================================================
// Individual engine wrappers, each exercised against the algorithm it wraps.

TEST(SolverEngine, SimplexSolvesTheLpItWraps) {
  const SolverEngine* simplex = SolverRegistry::builtin().find("simplex");
  ASSERT_NE(simplex, nullptr);
  EXPECT_TRUE(simplex->supports(tiny_lp()));
  Logger logger(nullptr);
  const Solution solution = simplex->solve(tiny_lp(), quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -4.0, 1e-6);
}

TEST(SolverEngine, DualSimplexSolvesTheLpItWraps) {
  const SolverEngine* dual = SolverRegistry::builtin().find("dual-simplex");
  ASSERT_NE(dual, nullptr);
  Logger logger(nullptr);
  const Solution solution = dual->solve(tiny_lp(), quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -4.0, 1e-6);
}

TEST(SolverEngine, BranchAndBoundSolvesTheMilpAndDoesNotReportTheRelaxation) {
  const SolverEngine* bnb = SolverRegistry::builtin().find("branch-and-bound");
  ASSERT_NE(bnb, nullptr);
  EXPECT_TRUE(bnb->supports(tiny_milp()));
  EXPECT_FALSE(bnb->supports(tiny_lp())) << "an LP with no integrality is not this engine's "
                                         << "class - the plain simplex is";
  Logger logger(nullptr);
  const Solution solution = bnb->solve(tiny_milp(), quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -2.0, 1e-6)
      << "the relaxation is -2.5; a value near that means integrality was dropped";
}

TEST(SolverEngine, ConvexQpSolvesTheQpItWraps) {
  const SolverEngine* qp_engine = SolverRegistry::builtin().find("convex-qp");
  ASSERT_NE(qp_engine, nullptr);
  EXPECT_TRUE(qp_engine->supports(tiny_qp()));
  Logger logger(nullptr);
  const Solution solution = qp_engine->solve(tiny_qp(), quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -4.0, 1e-5);
}

TEST(SolverEngine, AnEngineHandedAClassOutsideItsCapabilitiesReportsSoRatherThanGuessing) {
  const SolverEngine* simplex = SolverRegistry::builtin().find("simplex");
  ASSERT_NE(simplex, nullptr);
  ASSERT_FALSE(simplex->supports(tiny_milp()));
  Logger logger(nullptr);
  const Solution solution = simplex->solve(tiny_milp(), quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kNotSolved);
  EXPECT_NE(solution.message.find("MILP"), std::string::npos) << solution.message;
}

// =========================================================================================
// SolverSelector

TEST(SolverSelector, RequestedNameIsHonouredWhenItSupportsTheClass) {
  Options options = quiet_options();
  options.set_string("algorithm", "ipm");
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_lp(), options);
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_EQ(choice.engine->name(), "ipm");
  EXPECT_EQ(choice.rule, "requested");
}

TEST(SolverSelector, UnregisteredRequestedNameComesBackWithNoEngine) {
  Options options = quiet_options();
  options.set_string("algorithm", "not-an-engine");
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_lp(), options);
  EXPECT_EQ(choice.engine, nullptr);
  EXPECT_EQ(choice.rule, "unregistered");
}

TEST(SolverSelector, RequestedNameThatDoesNotAcceptTheClassComesBackWithNoEngine) {
  Options options = quiet_options();
  options.set_string("algorithm", "ipm");  // LP-only
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_milp(), options);
  EXPECT_EQ(choice.engine, nullptr);
  EXPECT_EQ(choice.rule, "unsupported");
}

TEST(SolverSelector, AutoOnAnLpResolvesToARegisteredEngine) {
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_lp(), quiet_options());
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_TRUE(choice.engine->capabilities().lp);
}

TEST(SolverSelector, AutoOnAMilpPicksTheOnlyCandidate) {
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_milp(), quiet_options());
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_EQ(choice.engine->name(), "branch-and-bound");
  EXPECT_EQ(choice.rule, "only-candidate");
}

TEST(SolverSelector, AutoOnAQpPicksTheOnlyCandidate) {
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_qp(), quiet_options());
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_EQ(choice.engine->name(), "convex-qp");
  EXPECT_EQ(choice.rule, "only-candidate");
}

}  // namespace
}  // namespace sankhya::engine
