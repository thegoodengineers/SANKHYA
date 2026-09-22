// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the SolverEngine interface (#297): a common shape every optimization engine
// implements, so engines can be discovered, described and selected through one interface
// instead of the class-by-class branching src/core/solve.cpp does today.
//
// This header does not change how solve() dispatches - src/core/solve.cpp:1-12 names that
// file "THE SEAM" and it stays the seam. What lives here wraps the EXISTING free-function
// engines (solve_primal_simplex, solve_dual_simplex, pdhg::solve_pdhg, ipm::solve_ipm,
// qp::solve_convex_qp, mip::solve_branch_and_bound) behind one interface, in
// src/solver_engine/builtin_engines.cpp. None of those algorithms is reimplemented.
#pragma once

#include <string>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::engine {

/// The four problem classes solve() distinguishes (src/core/solve.cpp), named again here so
/// a SolverEngine can declare which ones it accepts without depending on the dispatcher's
/// own (anonymous-namespace, private) enum of the same shape.
enum class ProblemClass { kLp, kMilp, kQp, kMiqp };

/// `model`'s class, decided the same way src/core/solve.cpp's private classify() decides it:
/// integrality and a quadratic objective are properties of the model, not a caller's
/// assertion.
[[nodiscard]] ProblemClass classify(const Model& model);

[[nodiscard]] const char* to_string(ProblemClass problem_class);

/// What an engine can do, declared rather than probed. Only a capability the engine actually
/// implements is set true (#297); this is read by a selector to decide whether an engine is
/// even a candidate for a model, not a marketing claim about it.
struct EngineCapabilities {
  bool lp = false;
  bool milp = false;
  bool qp = false;
  bool miqp = false;

  bool supports_gpu = false;
  bool supports_warm_start = false;
  bool supports_duals = false;
  bool supports_basis = false;
  bool supports_certificates = false;
  bool supports_interrupt = false;
  // Whether THIS WRAPPER's solve() itself removes clock-dependent decisions and repeats
  // itself under options.deterministic=true, WITHOUT relying on solve()'s dispatcher to have
  // done it upstream. src/core/deterministic_mode.hpp (apply_deterministic_mode) is the one
  // rewrite that makes this true; a wrapper earns this flag by calling it itself, on the
  // options it hands to its underlying algorithm, before that call (#297 review, A2). Every
  // built-in engine (src/solver_engine/builtin_engines.cpp) does exactly that and sets this
  // true. A future engine that does NOT call apply_deterministic_mode itself - or one for
  // which no rewrite would suffice - must leave this false rather than borrow the guarantee
  // from a dispatcher it was not reached through.
  bool supports_deterministic_mode = false;

  /// Whether `problem_class` is one of the classes this engine accepts.
  [[nodiscard]] bool accepts(ProblemClass problem_class) const;
};

/// The Solution a SolverEngine returns when handed a model outside its capabilities() - the
/// truth, per ENGINEERING_RULES.md, rather than a plausible answer to a different problem.
/// Every built-in engine (src/solver_engine/builtin_engines.cpp) uses this; it is exported so
/// a THIRD-PARTY engine implementing this interface gets the same status/message convention
/// for free rather than having to invent one (#297: "adding a new solver should primarily
/// require: implement engine, register engine, declare capabilities" - not this).
[[nodiscard]] Solution unsupported_class_result(const std::string& engine_name,
                                                const Model& model);

/// A pluggable optimization engine (#297).
///
/// An implementation wraps an EXISTING solve_*() free function; `solve()` here is a thin
/// adapter and must not reimplement the algorithm. ENGINEERING_RULES.md's rule about an
/// unimplemented path applies here as everywhere: `supports()` is the honest gate, and an
/// engine handed a class it does not support returns kNotSolved and says so rather than
/// guessing at an answer.
class SolverEngine {
 public:
  virtual ~SolverEngine() = default;

  /// The name this engine is selected by - the same word the `algorithm` option takes,
  /// where one exists ("simplex", "dual-simplex", "pdhg", "ipm"), or a name of its own for
  /// an engine the option does not name directly ("branch-and-bound", "convex-qp").
  [[nodiscard]] virtual std::string name() const = 0;

  [[nodiscard]] virtual EngineCapabilities capabilities() const = 0;

  /// One sentence on what the engine is and when the dispatcher picks it, for
  /// `sankhya engines` (src/solver_engine/engine_listing.hpp); empty when it has nothing to
  /// say. Description only: nothing reads it to make a decision.
  [[nodiscard]] virtual std::string summary() const { return {}; }
  /// Where the algorithm lives, as a path under the repository root, for the same listing.
  [[nodiscard]] virtual std::string source() const { return {}; }

  /// True when this engine is willing to attempt `model`. The default implementation checks
  /// only the problem class against `capabilities()`; an engine with a narrower contract
  /// (say, one that refuses a non-convex Hessian) may override to say so before solve() is
  /// called.
  [[nodiscard]] virtual bool supports(const Model& model) const;

  /// Solve `model`: the safe, gated entry point. A TEMPLATE METHOD, not virtual - checks
  /// supports(model) once, here, in one place for every engine, and returns
  /// unsupported_class_result() rather than calling solve_verified() when it fails. An
  /// engine handed a class outside its capabilities gets a Solution with status kNotSolved
  /// and a message naming why, rather than guessing.
  [[nodiscard]] Solution solve(const Model& model, const Options& options, Logger& logger,
                               SolveControl* control = nullptr) const;

  /// Solve `model`, trusting that supports() was already checked and passed - possibly
  /// against a DIFFERENT, EARLIER model that presolve then reduced `model` from. Presolve
  /// can legitimately change what classify() reports (fixing every integer column removes
  /// the columns classify() would have read, degenerating a MILP into what looks like an
  /// LP) without that meaning this stopped being the right engine to run - solve()'s own
  /// dispatcher (src/core/solve.cpp) has never re-classified a presolved model before
  /// running the engine it already chose, and this is how a caller who did that choosing
  /// itself (src/solver_engine/solver_engine_dispatch.cpp's engine::solve(), through
  /// select() and run_with_presolve) gets the same freedom.
  ///
  /// NEVER call this directly unless you have already confirmed, on the model whose CLASS
  /// this engine was chosen for, that supports() held - solve() above is the safe default
  /// entry point, and every built-in engine implements ITS ALGORITHM here, not the gate.
  [[nodiscard]] virtual Solution solve_verified(const Model& model, const Options& options,
                                                Logger& logger,
                                                SolveControl* control) const = 0;
};

}  // namespace sankhya::engine
