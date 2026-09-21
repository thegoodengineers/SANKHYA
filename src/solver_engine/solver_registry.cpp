// SPDX-License-Identifier: Apache-2.0
#include "solver_engine/solver_registry.hpp"

#include <utility>

#include "solver_engine/builtin_engines.hpp"

namespace sankhya::engine {

void SolverRegistry::register_engine(std::shared_ptr<const SolverEngine> engine) {
  const std::string name = engine->name();
  for (auto& existing : engines_) {
    if (existing->name() == name) {
      existing = std::move(engine);
      return;
    }
  }
  engines_.push_back(std::move(engine));
}

const SolverEngine* SolverRegistry::find(std::string_view name) const {
  for (const auto& engine : engines_) {
    if (engine->name() == name) return engine.get();
  }
  return nullptr;
}

std::vector<std::string> SolverRegistry::names() const {
  std::vector<std::string> result;
  result.reserve(engines_.size());
  for (const auto& engine : engines_) result.push_back(engine->name());
  return result;
}

std::vector<const SolverEngine*> SolverRegistry::candidates(const Model& model) const {
  std::vector<const SolverEngine*> result;
  for (const auto& engine : engines_) {
    if (engine->supports(model)) result.push_back(engine.get());
  }
  return result;
}

const SolverRegistry& SolverRegistry::builtin() {
  static const SolverRegistry registry = [] {
    SolverRegistry built;
    register_builtin_engines(built);
    return built;
  }();
  return registry;
}

}  // namespace sankhya::engine
