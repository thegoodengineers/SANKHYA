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

/// LIFECYCLE, OWNERSHIP AND THREAD SAFETY (#297 review, engine factory / lifecycle /
/// ownership / thread-safety DoD items).
///
/// OWNERSHIP: a registered SolverEngine is held by `shared_ptr<const SolverEngine>` - the
/// registry owns a reference, `find()`/`candidates()` hand out non-owning `const
/// SolverEngine*` valid for as long as the registry (or, for SolverRegistry::builtin(), the
/// process) is alive. No caller is ever handed an owning pointer, so there is nothing to
/// free and no ambiguity about who does.
///
/// LIFECYCLE: an engine is immutable and stateless once constructed - name() and
/// capabilities() are pure functions of the engine's own type, and solve() takes the model,
/// options, logger and control it needs as arguments rather than storing any of them. A
/// SolverEngine has no separate "configure" or "close" step; construction through
/// register_builtin_engines() (src/solver_engine/builtin_engines.cpp) - the factory - is the
/// whole lifecycle.
///
/// THREAD SAFETY: SolverRegistry::builtin() is safe to call concurrently - its one-time
/// construction is a C++11 function-local static ("magic static"), guaranteed
/// thread-safe-once by the language, and every read afterwards (find/names/candidates) is a
/// read of already-immutable data. A caller-owned, mutable SolverRegistry is NOT safe for
/// concurrent register_engine() calls (ordinary, undocumented-elsewhere mutable-container
/// rules apply: build it, then treat it as read-only, exactly as builtin() does internally).
/// SolverEngine::solve() itself is safe to call concurrently from multiple threads on
/// DIFFERENT Model/Options/Solution instances - every built-in wrapper is a pure function of
/// its arguments, matching the free-function engine it wraps. Two calls that SHARE one
/// SolveControl* inherit that type's own thread-safety contract
/// (include/sankhya/solve_control.hpp: interrupt() "is safe to call from another thread or a
/// signal handler"); nothing in this layer weakens or strengthens that.
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
