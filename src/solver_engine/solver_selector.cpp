// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/solver_selector.hpp"

#include <fmt/format.h>

#include "core/engine_selection.hpp"

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#include "gpu/pdhg_gpu_guard.hpp"
#endif

namespace sankhya::engine {
namespace {

/// Whether the LP engine named "pdhg" should be upgraded to the registry's "pdhg-gpu" engine
/// instead: `rule_says_gpu` is true when either the auto-selection rule table judged the
/// model GPU-eligible or the caller set --gpu (matching solve.cpp's historical
/// `chosen.use_gpu || options.get_bool("gpu")` - the same test applies whether "pdhg" was
/// reached through "auto" or requested explicitly, #297 full integration). A "pdhg-gpu"
/// engine must actually be registered, and gpu::gpu_pdhg_is_safe - the SAME shared guard
/// solve.cpp and the CUDA wrapper use (#297 review, A1) - must agree; either failing means
/// staying on CPU "pdhg" rather than refusing outright.
bool should_use_gpu_pdhg(const SolverRegistry& registry, const Model& model,
                         const Options& options, bool rule_says_gpu, Logger& logger) {
#ifdef SANKHYA_ENABLE_CUDA
  if (!rule_says_gpu) return false;
  if (registry.find("pdhg-gpu") == nullptr) return false;
  return gpu::gpu_pdhg_is_safe(model, options, logger);
#else
  (void)registry;
  (void)model;
  (void)options;
  (void)rule_says_gpu;
  (void)logger;
  return false;
#endif
}

}  // namespace

EngineChoice select(const SolverRegistry& registry, const Model& model, const Options& options,
                    Logger& logger, bool warm_start) {
  const ProblemClass problem_class = classify(model);
  const std::string requested = options.get_string("algorithm");

  if (requested != "auto") {
    const SolverEngine* named = registry.find(requested);
    if (named == nullptr) {
      return EngineChoice{nullptr, "unregistered",
                          fmt::format("'{}' is not a registered engine", requested)};
    }
    if (!named->capabilities().accepts(problem_class)) {
      return EngineChoice{
          nullptr, "unsupported",
          fmt::format("'{}' does not support {} models", requested, to_string(problem_class))};
    }
    // algorithm=pdhg combined with --gpu is upgraded here too, not only under "auto": #297's
    // own selector example asks for the CUDA backend "if eligible" underneath a plain
    // algorithm=pdhg request, not through a second user-facing name for it (the registered
    // "pdhg-gpu" engine itself stays reachable by explicit name below, already
    // capability-checked above - this just also lets an explicit "pdhg" ask upgrade).
    if (named->name() == "pdhg" &&
        should_use_gpu_pdhg(registry, model, options, options.get_bool("gpu"), logger)) {
      named = registry.find("pdhg-gpu");
    }
    // The rule and the reason come from select_engine(), as they did before the registry,
    // so an explicit request reads the same to a caller and in the stats JSON as it always
    // has: "algorithm=X was asked for (rows, columns, nonzeros)".
    const EngineSelection asked = select_engine(model, options, warm_start);
    return EngineChoice{named, asked.rule, asked.reason};
  }

  if (problem_class == ProblemClass::kLp) {
    // The GPU probe is conditional on auto-selection, exactly as solve.cpp's is: probing the
    // device has a real CUDA runtime init cost, and a caller who named an algorithm explicitly
    // already took the "requested" branch above.
    bool gpu_avail = false;
    std::string gpu_desc;
#ifdef SANKHYA_ENABLE_CUDA
    gpu_avail = gpu::device_available(&gpu_desc);
#endif
    const EngineSelection chosen =
        select_engine(model, options, warm_start, gpu_avail, gpu_desc);
    std::string engine_name = chosen.algorithm;
    if (chosen.algorithm == "pdhg" &&
        should_use_gpu_pdhg(registry, model, options, chosen.use_gpu || options.get_bool("gpu"),
                            logger)) {
      engine_name = "pdhg-gpu";
    }
    const SolverEngine* named = registry.find(engine_name);
    if (named == nullptr) {
      return EngineChoice{nullptr, chosen.rule,
                          fmt::format("{} (selected engine '{}' is not registered)",
                                      chosen.reason, engine_name)};
    }
    return EngineChoice{named, chosen.rule, chosen.reason};
  }

  const std::vector<const SolverEngine*> candidates = registry.candidates(model);
  if (candidates.empty()) {
    return EngineChoice{
        nullptr, "no-candidate",
        fmt::format("no registered engine supports {} models", to_string(problem_class))};
  }
  return EngineChoice{candidates.front(), "only-candidate",
                      fmt::format("'{}' is the registered engine for {} models",
                                  candidates.front()->name(), to_string(problem_class))};
}

}  // namespace sankhya::engine
