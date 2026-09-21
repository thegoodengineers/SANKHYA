// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the built-in SolverEngine wrappers (#297). Each class below is a thin adapter
// over one EXISTING solve_*() free function; the algorithm it calls is untouched, and every
// capability declared here is exactly what src/core/solve.cpp already relies on that engine
// for.
#include "solver_engine/builtin_engines.hpp"

#include <memory>

#include <fmt/format.h>

#include "sankhya/ipm.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/qp.hpp"

#include "simplex/primal_simplex.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/pdhg_gpu.hpp"
#endif

namespace sankhya::engine {
namespace {

/// The Solution an engine returns when handed a class outside its capabilities() - the
/// truth, per ENGINEERING_RULES.md, rather than a plausible answer to a different problem.
Solution not_supported(const std::string& engine_name, const Model& model) {
  Solution solution;
  solution.allocate_for(model);
  solution.status = SolveStatus::kNotSolved;
  solution.algorithm = "none";
  solution.message =
      fmt::format("{} does not support {} models", engine_name, to_string(classify(model)));
  return solution;
}

class RevisedPrimalSimplexEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "simplex"; }

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

  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control) const override {
    if (!supports(model)) return not_supported(name(), model);
    return solve_primal_simplex(model, options, logger, control);
  }
};

class DualSimplexEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "dual-simplex"; }

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

  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control) const override {
    if (!supports(model)) return not_supported(name(), model);
    return solve_dual_simplex(model, options, logger, control);
  }
};

class PdhgEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "pdhg"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_interrupt = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control) const override {
    if (!supports(model)) return not_supported(name(), model);
    return pdhg::solve_pdhg(model, options, logger, control);
  }
};

#ifdef SANKHYA_ENABLE_CUDA
class PdhgCudaEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "pdhg-gpu"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_gpu = true;
    caps.supports_interrupt = true;
    return caps;
  }

  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control) const override {
    if (!supports(model)) return not_supported(name(), model);
    return gpu::solve_pdhg_gpu(model, options, logger, control);
  }
};
#endif

class IpmEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "ipm"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.lp = true;
    caps.supports_duals = true;
    caps.supports_interrupt = true;
    caps.supports_deterministic_mode = true;
    return caps;
  }

  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control) const override {
    if (!supports(model)) return not_supported(name(), model);
    return ipm::solve_ipm(model, options, logger, control);
  }
};

class ConvexQpEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "convex-qp"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.qp = true;
    caps.supports_interrupt = true;
    return caps;
  }

  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control) const override {
    if (!supports(model)) return not_supported(name(), model);
    return qp::solve_convex_qp(model, options, logger, control);
  }
};

class BranchAndBoundEngine final : public SolverEngine {
 public:
  [[nodiscard]] std::string name() const override { return "branch-and-bound"; }

  [[nodiscard]] EngineCapabilities capabilities() const override {
    EngineCapabilities caps;
    caps.milp = true;
    caps.miqp = true;
    caps.supports_interrupt = true;
    return caps;
  }

  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control) const override {
    if (!supports(model)) return not_supported(name(), model);
    return mip::solve_branch_and_bound(model, options, logger, control);
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
