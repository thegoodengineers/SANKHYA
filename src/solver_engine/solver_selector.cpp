// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/solver_selector.hpp"

#include <fmt/format.h>

#include "core/engine_selection.hpp"

namespace sankhya::engine {

EngineChoice select(const SolverRegistry& registry, const Model& model, const Options& options,
                    bool warm_start) {
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
    return EngineChoice{named, "requested",
                        fmt::format("engine '{}' was requested", requested)};
  }

  if (problem_class == ProblemClass::kLp) {
    const EngineSelection chosen = select_engine(model, options, warm_start);
    const SolverEngine* named = registry.find(chosen.algorithm);
    if (named == nullptr) {
      return EngineChoice{nullptr, chosen.rule,
                          fmt::format("{} (selected engine '{}' is not registered)",
                                      chosen.reason, chosen.algorithm)};
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
