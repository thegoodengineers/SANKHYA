// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the SolverEngine / SolverRegistry / SolverSelector layer (#297).
//
// Every engine wrapper tested here calls the SAME free-function engine solve()'s dispatcher
// calls (src/core/solve.cpp); the assertions below are therefore about the adapter - name,
// capabilities, the supports() gate, and that solve() on the wrapper reaches the same answer
// the underlying algorithm is already known to reach - not a second copy of the algorithm's
// own correctness tests.

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

#include "solver_engine/builtin_engines.hpp"
#include "solver_engine/engine_listing.hpp"
#include "solver_engine/solver_engine.hpp"
#include "solver_engine/solver_engine_dispatch.hpp"
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
  // Model::fingerprint() (used by solve()'s deterministic-mode logging) reads the Hessian
  // unconditionally, even for a pure LP - an empty one still has to be finalized, matching
  // the convention tests/unit/test_miqp.cpp already follows.
  model.hessian.reset(2, 2);
  model.hessian.finalize();
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
  model.hessian.reset(1, 1);
  model.hessian.finalize();
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
  [[nodiscard]] Solution solve_verified(const Model& model, const Options&, Logger&,
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

// Every built-in engine now calls apply_deterministic_mode (src/core/deterministic_mode.hpp)
// itself, so the capability is honestly true everywhere (#297 review, A2 superseded by the
// full #297 implementation): the claim is backed by code in this file, not borrowed from
// solve()'s dispatcher.
TEST(SolverEngine, EveryBuiltinEngineClaimsAndTheClaimIsBackedByItsOwnSolve) {
  const SolverRegistry& registry = SolverRegistry::builtin();
  for (const std::string& name : registry.names()) {
    const SolverEngine* found = registry.find(name);
    ASSERT_NE(found, nullptr);
    EXPECT_TRUE(found->capabilities().supports_deterministic_mode)
        << name << " does not claim wrapper-level determinism; every built-in engine's "
        << "solve() calls apply_deterministic_mode itself and should";
  }
}

// The real proof the flag above is honest: with presolve off (so both paths run the SAME
// engine cold, on the SAME model, with nothing else in between), a wrapper called directly
// must reach the identical answer solve()'s own dispatcher reaches under
// deterministic=true - not merely an equally-reproducible one, but the SAME numbers, because
// both are now applying the identical Options rewrite before the identical algorithm runs.
TEST(SolverEngine,
     DeterministicModeGivesTheWrapperTheSameAnswerAsSolveNotJustAReproducibleOne) {
  Options options = quiet_options();
  options.set_bool("presolve", false);
  options.set_bool("deterministic", true);
  options.set_string("algorithm", "simplex");

  const Model lp = tiny_lp();
  const Solution via_dispatcher = solve(lp, options);

  const SolverEngine* simplex = SolverRegistry::builtin().find("simplex");
  ASSERT_NE(simplex, nullptr);
  Logger logger(nullptr);
  const Solution via_wrapper = simplex->solve(lp, options, logger);

  EXPECT_EQ(via_wrapper.status, via_dispatcher.status);
  EXPECT_EQ(via_dispatcher.status, SolveStatus::kOptimal) << via_dispatcher.message;
  EXPECT_EQ(via_wrapper.objective, via_dispatcher.objective);
  EXPECT_EQ(via_wrapper.col_value, via_dispatcher.col_value);
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

// The dual-simplex and simplex wrappers declare supports_warm_start=true; this proves the
// claim is honest by measuring what it is FOR - fewer pivots on a re-solve - directly
// through the wrapper, the way tests/unit/test_warm_start.cpp already proves it for solve()
// (#297 full integration: solve_primal_simplex/solve_dual_simplex's plain 4-argument
// overload never looks at SolveControl's starting basis on its own - see
// src/simplex/primal_simplex.cpp - so without the wrapper building a WarmStart itself, this
// capability would have been exactly the kind of unenforced claim A2 refused to allow).
TEST(SolverEngine, DualSimplexWrapperActuallyUsesTheStartingBasisNotJustClaimsTo) {
  Model model = afiro();
  Options options = quiet_options();
  const SolverEngine* dual = SolverRegistry::builtin().find("dual-simplex");
  ASSERT_NE(dual, nullptr);
  Logger logger(nullptr);

  const Solution first = dual->solve(model, options, logger);
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;

  Index pick = -1;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (first.col_status[u] != BasisStatus::kBasic) continue;
    if (pick < 0 || first.col_value[u] > first.col_value[static_cast<std::size_t>(pick)]) {
      pick = j;
    }
  }
  ASSERT_GE(pick, 0);
  const auto p = static_cast<std::size_t>(pick);
  ASSERT_GT(first.col_value[p], 0.0);
  model.col_upper[p] = 0.5 * first.col_value[p];

  const Solution cold = dual->solve(model, options, logger);
  ASSERT_EQ(cold.status, SolveStatus::kOptimal) << cold.message;

  SolveControl control;
  control.start_col_status = first.col_status;
  control.start_row_status = first.row_status;
  const Solution warm = dual->solve(model, options, logger, &control);
  ASSERT_EQ(warm.status, SolveStatus::kOptimal) << warm.message;
  EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * std::max(1.0, std::fabs(cold.objective)));
  EXPECT_LE(warm.iterations, cold.iterations)
      << "warm " << warm.iterations << " pivots against " << cold.iterations << " cold";
}

// =========================================================================================
// SolverSelector

TEST(SolverSelector, RequestedNameIsHonouredWhenItSupportsTheClass) {
  Options options = quiet_options();
  options.set_string("algorithm", "ipm");
  Logger logger(nullptr);
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_lp(), options, logger);
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_EQ(choice.engine->name(), "ipm");
  EXPECT_EQ(choice.rule, "requested");
}

TEST(SolverSelector, UnregisteredRequestedNameComesBackWithNoEngine) {
  Options options = quiet_options();
  options.set_string("algorithm", "not-an-engine");
  Logger logger(nullptr);
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_lp(), options, logger);
  EXPECT_EQ(choice.engine, nullptr);
  EXPECT_EQ(choice.rule, "unregistered");
}

TEST(SolverSelector, RequestedNameThatDoesNotAcceptTheClassComesBackWithNoEngine) {
  Options options = quiet_options();
  options.set_string("algorithm", "ipm");  // LP-only
  Logger logger(nullptr);
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_milp(), options, logger);
  EXPECT_EQ(choice.engine, nullptr);
  EXPECT_EQ(choice.rule, "unsupported");
}

TEST(SolverSelector, AutoOnAnLpResolvesToARegisteredEngine) {
  Logger logger(nullptr);
  const EngineChoice choice =
      select(SolverRegistry::builtin(), tiny_lp(), quiet_options(), logger);
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_TRUE(choice.engine->capabilities().lp);
}

TEST(SolverSelector, AutoOnAMilpPicksTheOnlyCandidate) {
  Logger logger(nullptr);
  const EngineChoice choice =
      select(SolverRegistry::builtin(), tiny_milp(), quiet_options(), logger);
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_EQ(choice.engine->name(), "branch-and-bound");
  EXPECT_EQ(choice.rule, "only-candidate");
}

TEST(SolverSelector, AutoOnAQpPicksTheOnlyCandidate) {
  Logger logger(nullptr);
  const EngineChoice choice =
      select(SolverRegistry::builtin(), tiny_qp(), quiet_options(), logger);
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  EXPECT_EQ(choice.engine->name(), "convex-qp");
  EXPECT_EQ(choice.rule, "only-candidate");
}

// solve.cpp's own MILP/QP/MIQP branches (src/core/solve.cpp) have never consulted the
// `algorithm` option - there is one engine per class, so nothing to select between - and
// still do not: SolverSelector is not wired into that dispatch, so its own, deliberately
// stricter contract below cannot and does not change what solve() does (#297 review, B2).
// This test pins solve()'s side of that fact so a future change cannot alter it silently.
TEST(SolverSelector, SolveIgnoresAlgorithmForMilpExactlyAsBeforeSolverSelectorExisted) {
  Options options = quiet_options();
  options.set_string("algorithm", "pdhg");  // meaningless for a MILP; solve() ignores it
  const Solution solution = solve(tiny_milp(), options);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_EQ(solution.algorithm, "branch-and-bound");
}

// SolverSelector's own, deliberately different contract (#297 review, B2): going through the
// registry directly, an `algorithm` that does not name a class-appropriate registered engine
// is refused rather than silently solved by whatever the class's one engine happens to be.
TEST(SolverSelector, RegistrySelectorRefusesAnAlgorithmThatDoesNotAcceptMilpUnlikeSolve) {
  Options options = quiet_options();
  options.set_string("algorithm", "pdhg");  // LP-only engine, requested for a MILP
  Logger logger(nullptr);
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_milp(), options, logger);
  EXPECT_EQ(choice.engine, nullptr);
  EXPECT_EQ(choice.rule, "unsupported");
}

// On a build with no CUDA backend compiled in (this machine, and solve()'s "GPU off" default
// everywhere without a device), the auto path must stay on CPU "pdhg" even when the caller
// asks for --gpu: there is nothing in the registry to upgrade to, and should_use_gpu_pdhg's
// #else branch says so. This is the one branch of the GPU-selection logic this machine can
// actually exercise; the CUDA-enabled branch is compiled only under SANKHYA_ENABLE_CUDA and
// is covered by the pre-existing GPU/CUDA test files (test_gpu_device.cpp etc.), which skip
// gracefully without a device, consistent with the rest of this suite.
TEST(SolverSelector, WithoutACudaBackendAutoStaysOnCpuPdhgEvenWithTheGpuFlagSet) {
  Options options = quiet_options();
  options.set_bool("gpu", true);
  Logger logger(nullptr);
  const EngineChoice choice = select(SolverRegistry::builtin(), tiny_lp(), options, logger);
  ASSERT_NE(choice.engine, nullptr) << choice.reason;
  if (choice.engine->capabilities().supports_gpu) {
    GTEST_SKIP() << "a pdhg-gpu engine is registered on this build; nothing to pin here";
  }
}

// =========================================================================================
// engine::solve() - the registry's own validated top-level entry point (#297 review, B3 /
// model validation contract).

TEST(RegistryDispatch, AnInvalidModelIsRefusedBeforeAnyEngineIsReached) {
  Model broken = tiny_lp();
  broken.col_lower.pop_back();  // now mismatched against num_cols(): Model::validate() fails
  ASSERT_NE(broken.validate(), "");

  Logger logger(nullptr);
  const Solution solution = solve(SolverRegistry::builtin(), broken, quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kModelError);
  EXPECT_EQ(solution.message, broken.validate());
}

TEST(RegistryDispatch, AValidLpIsSolvedThroughAutoSelection) {
  Logger logger(nullptr);
  const Solution solution =
      solve(SolverRegistry::builtin(), tiny_lp(), quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -4.0, 1e-6);
}

TEST(RegistryDispatch, AValidMilpIsSolvedThroughTheOnlyCandidate) {
  Logger logger(nullptr);
  const Solution solution =
      solve(SolverRegistry::builtin(), tiny_milp(), quiet_options(), logger);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_EQ(solution.algorithm, "branch-and-bound");
}

TEST(RegistryDispatch, AnUnservableRequestComesBackAsNotSolvedRatherThanAnEngineGuessing) {
  Options options = quiet_options();
  options.set_string("algorithm", "ipm");  // LP-only, requested for a MILP
  Logger logger(nullptr);
  const Solution solution = solve(SolverRegistry::builtin(), tiny_milp(), options, logger);
  EXPECT_EQ(solution.status, SolveStatus::kNotSolved);
  EXPECT_FALSE(solution.message.empty());
}

// Presolve, resource limits and the out-of-memory guard are shared with solve() through
// run_with_presolve/ResourceLimits/run_engine_guarded (#297 full integration); this proves
// presolve genuinely ran inside engine::solve() - not just that the answer happens to
// match - by checking the reduction it reports, on afiro (known to reduce: the CLI reports
// "27 -> 25 rows" with presolve on).
TEST(RegistryDispatch, PresolveGenuinelyRunsInsideEngineSolveAndMatchesSolve) {
  const Model model = afiro();
  Logger logger(nullptr);
  const Solution via_registry =
      solve(SolverRegistry::builtin(), model, quiet_options(), logger);
  ASSERT_EQ(via_registry.status, SolveStatus::kOptimal) << via_registry.message;
  EXPECT_TRUE(via_registry.presolve_report.ran);
  EXPECT_GT(via_registry.presolve_report.original_rows,
            via_registry.presolve_report.reduced_rows)
      << "presolve does not appear to have reduced afiro through engine::solve()";

  const Solution via_dispatcher = sankhya::solve(model, quiet_options());
  ASSERT_EQ(via_dispatcher.status, SolveStatus::kOptimal) << via_dispatcher.message;
  EXPECT_NEAR(via_registry.objective, via_dispatcher.objective, 1e-6);
  EXPECT_EQ(via_registry.presolve_report.original_rows,
            via_dispatcher.presolve_report.original_rows);
  EXPECT_EQ(via_registry.presolve_report.reduced_rows,
            via_dispatcher.presolve_report.reduced_rows);
}

// The bug this regression pins: presolve fixing tiny_milp()'s one integer column to its
// optimum leaves a reduced model with no integer columns at all, which classify() reports
// as LP - a class BranchAndBoundEngine does not declare in its capabilities(). Before
// SolverEngine::solve_verified() existed, run_with_presolve's inner call went through the
// GATED solve(), which re-derived supports() from that reduced model and wrongly refused
// "branch-and-bound does not support LP models", even though select() had already, and
// correctly, chosen branch-and-bound for the ORIGINAL, pre-presolve MILP.
TEST(RegistryDispatch, ADegenerateMilpAfterPresolveStillReachesBranchAndBound) {
  Logger logger(nullptr);
  const Solution solution =
      solve(SolverRegistry::builtin(), tiny_milp(), quiet_options(), logger);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_EQ(solution.algorithm, "branch-and-bound");
  EXPECT_NEAR(solution.objective, -2.0, 1e-9);
}

// solve_verified()'s own contract, directly: mip::solve_branch_and_bound genuinely accepts
// a model with no integrality at all (single LP solve, per its own header comment) - calling
// solve_verified() on one bypasses BranchAndBoundEngine's class gate exactly as
// engine::solve() now relies on it doing for a presolve-degenerated MILP above. The GATED
// solve() would refuse this same model; solve_verified() must not.
TEST(SolverEngine, SolveVerifiedRunsBranchAndBoundOnAPlainLpBypassingTheClassGate) {
  const SolverEngine* bnb = SolverRegistry::builtin().find("branch-and-bound");
  ASSERT_NE(bnb, nullptr);
  const Model lp = tiny_lp();
  ASSERT_FALSE(bnb->supports(lp));
  Logger logger(nullptr);

  const Solution gated = bnb->solve(lp, quiet_options(), logger);
  EXPECT_EQ(gated.status, SolveStatus::kNotSolved);

  const Solution trusted = bnb->solve_verified(lp, quiet_options(), logger, nullptr);
  EXPECT_EQ(trusted.status, SolveStatus::kOptimal) << trusted.message;
  EXPECT_NEAR(trusted.objective, -4.0, 1e-6);
}

// =========================================================================================
// One list of engines (#297): the option table, the dispatcher and the listing agree
// =========================================================================================

TEST(OptionsAndRegistry, AlgorithmChoicesAreAutoPlusTheRegistrysAlgorithmNames) {
  // src/util/options.cpp cannot ask the registry which names `algorithm` takes without the
  // option layer depending on the engines, so it carries the list and this test holds it to
  // SolverRegistry::algorithm_names(), the predicate solve() and `sankhya engines` use.
  const OptionSpec* spec = Options::find_spec("algorithm");
  ASSERT_NE(spec, nullptr);
  std::vector<std::string> expected = SolverRegistry::builtin().algorithm_names();
  expected.insert(expected.begin(), "auto");
  std::vector<std::string> choices = spec->choices;
  std::sort(expected.begin(), expected.end());
  std::sort(choices.begin(), choices.end());
  EXPECT_EQ(choices, expected);
  // The GPU engine is never a name the option takes; it is reached through pdhg or auto.
  EXPECT_EQ(std::find(choices.begin(), choices.end(), "pdhg-gpu"), choices.end());
}

TEST(EngineListing, JsonNamesEveryRegisteredEngineWithTheFlagsItDeclares) {
  const SolverRegistry& registry = SolverRegistry::builtin();
  const nlohmann::json doc = nlohmann::json::parse(format_engines_json(registry));
  ASSERT_TRUE(doc.contains("engines"));
  const std::vector<std::string> algorithm_names = registry.algorithm_names();
  std::vector<std::string> listed;
  for (const nlohmann::json& row : doc["engines"]) {
    const auto name = row["name"].get<std::string>();
    listed.push_back(name);
    const SolverEngine* engine = registry.find(name);
    ASSERT_NE(engine, nullptr) << name;
    const EngineCapabilities caps = engine->capabilities();
    EXPECT_EQ(row["classes"]["LP"].get<bool>(), caps.lp) << name;
    EXPECT_EQ(row["classes"]["MILP"].get<bool>(), caps.milp) << name;
    EXPECT_EQ(row["classes"]["QP"].get<bool>(), caps.qp) << name;
    EXPECT_EQ(row["classes"]["MIQP"].get<bool>(), caps.miqp) << name;
    EXPECT_EQ(row["capabilities"]["basis"].get<bool>(), caps.supports_basis) << name;
    EXPECT_EQ(row["capabilities"]["warm_start"].get<bool>(), caps.supports_warm_start) << name;
    EXPECT_EQ(row["capabilities"]["checkpoint"].get<bool>(), caps.supports_checkpoint) << name;
    EXPECT_EQ(row["selectable_by_algorithm"].get<bool>(),
              std::find(algorithm_names.begin(), algorithm_names.end(), name) !=
                  algorithm_names.end())
        << name;
    EXPECT_FALSE(row["summary"].get<std::string>().empty()) << name << " has no summary";
    EXPECT_FALSE(row["source"].get<std::string>().empty()) << name << " has no source";
  }
  std::vector<std::string> names = registry.names();
  std::sort(names.begin(), names.end());
  std::sort(listed.begin(), listed.end());
  EXPECT_EQ(listed, names);
  // The text listing carries the same names; a reader without a JSON parser sees them all.
  const std::string text = format_engines_text(registry);
  for (const std::string& name : names) {
    EXPECT_NE(text.find(name), std::string::npos) << name;
  }
#ifndef SANKHYA_ENABLE_CUDA
  // A CPU build says which engine the source tree has that this binary does not.
  EXPECT_EQ(doc["not_in_this_build"].size(), 1U);
  EXPECT_EQ(doc["not_in_this_build"][0].get<std::string>(), "pdhg-gpu");
  EXPECT_NE(text.find("not in this build"), std::string::npos);
#endif
}

TEST(SolveDispatch, AnLpEngineAskedForOnAMilpIsSaidOnTheAnswerNotDroppedSilently) {
  // The engine choice is unchanged (the MILP branch never read `algorithm`, see the test
  // above); what changes is that the answer now says the option chose nothing.
  Options options = quiet_options();
  options.set_string("algorithm", "ipm");
  const Solution said = solve(tiny_milp(), options);
  ASSERT_EQ(said.status, SolveStatus::kOptimal) << said.message;
  EXPECT_NEAR(said.objective, -2.0, 1e-9);
  EXPECT_EQ(said.algorithm, "branch-and-bound");
  EXPECT_NE(said.message.find("algorithm=ipm"), std::string::npos) << said.message;
  EXPECT_NE(said.message.find("branch-and-bound ran"), std::string::npos) << said.message;
  const Solution quiet = solve(tiny_milp(), quiet_options());
  ASSERT_EQ(quiet.status, SolveStatus::kOptimal) << quiet.message;
  EXPECT_EQ(quiet.message.find("algorithm="), std::string::npos) << quiet.message;
}

}  // namespace
}  // namespace sankhya::engine
