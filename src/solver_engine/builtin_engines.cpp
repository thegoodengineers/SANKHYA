// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the built-in SolverEngine wrappers (#297). Each class below is a thin adapter
// over one EXISTING solve_*() free function; the algorithm it calls is untouched, and every
// capability declared here is exactly what src/core/solve.cpp already relies on that engine
// for.
//
// Every solve() below applies apply_deterministic_mode (src/core/deterministic_mode.hpp)
// itself before calling its underlying algorithm - the SAME rewrite solve()'s dispatcher
// applies once, at the top, before any engine is chosen - so supports_deterministic_mode is
// true here because it is actually enforced by this wrapper, not because solve() happens to
// enforce it upstream of a caller who never reaches this file. The two simplex engines also
// apply verify_and_keep_certificate (src/core/certificate.hpp) themselves, for the same
// reason (#297 review, A2 and certificate support).
#include "solver_engine/builtin_engines.hpp"

#include <memory>
#include <utility>

#include "sankhya/certificate.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/timer.hpp"

#include "core/deterministic_mode.hpp"
#include "mip/components.hpp"
#include "simplex/crossover.hpp"
#include "simplex/primal_simplex.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/pdhg_gpu.hpp"
#include "gpu/pdhg_gpu_guard.hpp"
#endif

namespace sankhya::engine {
namespace {

class RevisedPrimalSimplexEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "simplex"; }
  [[nodiscard]] std::string summary() const override {
    return "Revised primal simplex, bounded variables, composite phase 1, no big-M; the "
           "hand-over engine when a basis needs repair";
  }
  [[nodiscard]] std::string source() const override { return "src/simplex/primal_simplex.cpp"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_warm_start = true;
    caps.supports_duals = true;
    caps.supports_basis = true;
    caps.supports_certificates = true;
    caps.supports_interrupt = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  // A starting basis in `control` is only usable if it names one status per column and row
  // of THIS model (#218); solve_primal_simplex's plain 4-argument overload never looks at
  // it (build_node_scaling / WarmStart have to be threaded through explicitly - see
  // src/simplex/primal_simplex.cpp), so without this the declared supports_warm_start
  // capability above would be exactly the kind of claim A2 refused to make: true in name,
  // never enforced by this wrapper's own solve() (#297 full integration).
  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Options effective = apply_deterministic_mode(options, logger);
    Solution solution;
    if (control != nullptr && control->has_starting_basis() &&
        static_cast<Index>(control->start_col_status.size()) == model.num_cols() &&
        static_cast<Index>(control->start_row_status.size()) == model.num_rows()) {
      WarmStart warm;
      warm.col_status = control->start_col_status;
      warm.row_status = control->start_row_status;
      solution = solve_primal_simplex(model, effective, logger,
                                      build_node_scaling(model, effective), control, &warm);
    } else {
      solution = solve_primal_simplex(model, effective, logger, control);
    }
    verify_and_keep_certificate(&solution, model, logger);
    return solution;
  }
};

class DualSimplexEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "dual-simplex"; }
  [[nodiscard]] std::string summary() const override {
    return "Revised dual simplex with the bound-flipping ratio test and devex pricing; the "
           "automatic choice below 20,000 rows and 100,000 nonzeros, and the branch and "
           "bound's node engine";
  }
  [[nodiscard]] std::string source() const override { return "src/simplex/dual_simplex.cpp"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_warm_start = true;
    caps.supports_duals = true;
    caps.supports_basis = true;
    caps.supports_certificates = true;
    caps.supports_interrupt = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  // See RevisedPrimalSimplexEngine::solve_verified() above for why this cannot just forward
  // `control`.
  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Options effective = apply_deterministic_mode(options, logger);
    Solution solution;
    if (control != nullptr && control->has_starting_basis() &&
        static_cast<Index>(control->start_col_status.size()) == model.num_cols() &&
        static_cast<Index>(control->start_row_status.size()) == model.num_rows()) {
      WarmStart warm;
      warm.col_status = control->start_col_status;
      warm.row_status = control->start_row_status;
      solution = solve_dual_simplex(model, effective, logger, control, &warm);
    } else {
      solution = solve_dual_simplex(model, effective, logger, control);
    }
    verify_and_keep_certificate(&solution, model, logger);
    return solution;
  }
};

class PdhgEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "pdhg"; }
  [[nodiscard]] std::string summary() const override {
    return "Restarted primal-dual hybrid gradient, first order, its answer finished by the "
           "interior point; the automatic choice from 100,000 rows";
  }
  [[nodiscard]] std::string source() const override { return "src/pdhg/pdhg.cpp"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_duals = true;  // row duals and reduced costs come with the point
    caps.supports_interrupt = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Options effective = apply_deterministic_mode(options, logger);
    return pdhg::solve_pdhg(model, effective, logger, control);
  }
};

#ifdef SANKHYA_ENABLE_CUDA
class PdhgCudaEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "pdhg-gpu"; }
  [[nodiscard]] std::string summary() const override {
    return "Restarted PDHG on a CUDA device with the CPU engine as the fallback; reached "
           "through algorithm=pdhg with gpu=true, or auto";
  }
  [[nodiscard]] std::string source() const override { return "src/gpu/pdhg_gpu.cu"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_gpu = true;
    caps.supports_interrupt = true;
    // Honest, not aspirational: gpu_pdhg_is_safe() below refuses the GPU outright when
    // deterministic=true (#383) and falls back to CPU PDHG on the SAME rewritten options
    // apply_deterministic_mode produces, so this engine's own solve() enforces the
    // guarantee end to end rather than relying on solve()'s dispatcher for it.
    caps.supports_deterministic_mode = true;
    return caps;
  }

  // Applies deterministic mode first (#288, #297 review), then the same compute-capability
  // and VRAM gates src/core/solve.cpp applies before it will run GPU PDHG (#281, #282, #383),
  // through the shared gpu::gpu_pdhg_is_safe (src/gpu/pdhg_gpu_guard.cu) - so calling this
  // engine directly cannot bypass any of them (#297 review, A1). A refusal falls back to CPU
  // PDHG, the same outcome solve.cpp's own dispatcher reaches in the same situation.
  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Options effective = apply_deterministic_mode(options, logger);
    if (!gpu::gpu_pdhg_is_safe(model, effective, logger)) {
      return pdhg::solve_pdhg(model, effective, logger, control);
    }
    return gpu::solve_pdhg_gpu(model, effective, logger, control);
  }
};
#endif

class IpmEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "ipm"; }
  [[nodiscard]] std::string summary() const override {
    return "Mehrotra predictor-corrector interior point on the normal equations over a sparse "
           "LDL^T, pushed to a vertex by crossover; the automatic choice from 20,000 rows or "
           "100,000 nonzeros";
  }
  [[nodiscard]] std::string source() const override { return "src/ipm/ipm.cpp"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_duals = true;
    caps.supports_basis = true;  // through crossover (#219), which is on by default
    caps.supports_interrupt = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  // The crossover is run HERE, as solve()'s LP branch runs it (src/core/solve.cpp), so the
  // basis declared above is on this wrapper's own answer: ipm::solve_ipm alone returns an
  // interior point with no basis, and without this the flag would be true only for a caller
  // who went through solve() (#297, found by test_solver_engine_conformance.cpp).
  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Timer timer;
    const Options effective = apply_deterministic_mode(options, logger);
    Solution interior = ipm::solve_ipm(model, effective, logger, control);
    if (effective.get_bool("crossover") && interior.status == SolveStatus::kOptimal) {
      return crossover_to_vertex(model, std::move(interior), effective, logger, control, timer);
    }
    return interior;
  }
};

class ConvexQpEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "convex-qp"; }
  [[nodiscard]] std::string summary() const override {
    return "Condat-Vu primal-dual splitting for a convex quadratic objective; a Hessian it "
           "cannot prove convex is refused, never solved to a local point";
  }
  [[nodiscard]] std::string source() const override { return "src/qp/qp_condat_vu.cpp"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.qp = true;
    caps.supports_interrupt = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Options effective = apply_deterministic_mode(options, logger);
    return qp::solve_convex_qp(model, effective, logger, control);
  }
};

class BranchAndBoundEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "branch-and-bound"; }
  [[nodiscard]] std::string summary() const override {
    return "Branch and bound with reliability branching, propagation, cut rounds and "
           "heuristics; node LPs on the warm-started dual simplex, QP relaxations for an MIQP, "
           "mip_threads workers";
  }
  [[nodiscard]] std::string source() const override { return "src/mip/branch_and_bound.cpp"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.milp = true;
    caps.miqp = true;
    caps.supports_interrupt = true;
    // checkpoint/resume are read inside solve_branch_and_bound itself (#287), so this wrapper
    // honours them by forwarding the options, as solve() does. The parallel tree search
    // declines them with a note in the log (mip_threads > 1, docs/ARCHITECTURE.md section 12).
    caps.supports_checkpoint = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  // Reports the composition #297 asks for (NodeSelector, BranchingStrategy, CutManager,
  // HeuristicManager, ConflictManager, RelaxationEngine - branch_and_bound.cpp already has
  // each as its own file/option, unrewritten) before running it, so the composition is
  // genuinely part of this engine's real execution path rather than a document only
  // describing it (src/mip/components.hpp).
  [[nodiscard]] Solution solve_verified(const Model& model, const Options& options,
                                        Logger& logger, SolveControl* control) const override {
    const Options effective = apply_deterministic_mode(options, logger);
    const mip::MilpComponents parts = mip::describe_components(effective);
    logger.info(
        "MILP composition: node selection {}, branching {}, relaxation engine {}, cuts {}, "
        "heuristics {}, conflict analysis {}",
        parts.node_selection, parts.branching, parts.relaxation_engine,
        parts.cuts_enabled ? "on" : "off", parts.heuristics_enabled ? "on" : "off",
        parts.conflict_analysis_enabled ? "on" : "off");
    return mip::solve_branch_and_bound(model, effective, logger, control);
  }
};

}  // namespace

void register_builtin_engines(SolverRegistry& registry) {
  registry.register_engine(std::make_shared<RevisedPrimalSimplexEngine>());
  registry.register_engine(std::make_shared<DualSimplexEngine>());
  registry.register_engine(std::make_shared<PdhgEngine>());
#ifdef SANKHYA_ENABLE_CUDA
  registry.register_engine(std::make_shared<PdhgCudaEngine>());
#endif
  registry.register_engine(std::make_shared<IpmEngine>());
  registry.register_engine(std::make_shared<ConvexQpEngine>());
  registry.register_engine(std::make_shared<BranchAndBoundEngine>());
}

}  // namespace sankhya::engine
