// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the SolverRegistry (#297): where a SolverEngine is discovered by name, so a
// caller does not have to hardcode which engines exist.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "solver_engine/solver_engine.hpp"

namespace sankhya::engine {

class SolverRegistry {
 public:
  /// Adds `engine` under its own name(). A second registration under a name already present
  /// replaces the first, so a caller can override a built-in engine for a test.
  void register_engine(std::shared_ptr<const SolverEngine> engine);

  /// The engine registered under `name`, or nullptr when none is.
  [[nodiscard]] const SolverEngine* find(std::string_view name) const;

  /// Every registered engine's name, in registration order.
  [[nodiscard]] std::vector<std::string> names() const;

  /// Every registered engine that supports() `model`, in registration order.
  [[nodiscard]] std::vector<const SolverEngine*> candidates(const Model& model) const;

  /// The registry with every engine src/solver_engine/builtin_engines.cpp knows how to
  /// build, registered once on first use.
  [[nodiscard]] static const SolverRegistry& builtin();

 private:
  std::vector<std::shared_ptr<const SolverEngine>> engines_;
};

}  // namespace sankhya::engine
