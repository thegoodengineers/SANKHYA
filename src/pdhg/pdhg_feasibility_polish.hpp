// SPDX-License-Identifier: Apache-2.0
// SANKHYA - PDHG feasibility polishing (#483), option pdhg_feasibility_polish.
//
// NOT the same thing as pdhg_polish (#229), which hands PDHG's point to the interior point.
// This is a first-order finish: PDHG itself, run on two easier problems that share the
// original problem's constraints, so that the answer is primal feasible to kPdhgTight with
// its optimality gap measured rather than implied.
//
// Written from the paper; per ENGINEERING_RULES.md no solver source was consulted:
//   [AHLL25] Applegate, Hinder, Lu & Lubin, "PDLP: A Practical First-Order Method for
//            Large-Scale Linear Programming", arXiv:2501.07018 (2025), the feasibility
//            polishing section.
#pragma once

#include <functional>
#include <vector>

#include "pdhg_evaluate.hpp"
#include "sankhya/model.hpp"
#include "sankhya/types.hpp"

#include "../la/scaling.hpp"

namespace sankhya::pdhg {

/// The two feasibility problems of [AHLL25], built once per solve.
///
///   primal: the original constraints with the objective set to zero. Any optimal point is a
///           primal feasible point of the original LP; started from (x_k, 0) PDHG lands near
///           x_k, so the objective moves little.
///   dual:   the original objective with every FINITE bound, row and column, set to zero.
///           Its dual feasible set is exactly the original's (dual feasibility depends only
///           on which bounds are finite), x = 0 is optimal whenever that set is nonempty,
///           and started from (0, y_k) PDHG finds a dual feasible y near y_k.
class FeasibilityPolisher {
 public:
  FeasibilityPolisher(const Problem& problem, const Scaling& scaling, double spectral_norm);

  /// What one polish of a candidate pair produced.
  struct Outcome {
    bool primal_reached = false;  ///< x met kPdhgTight relative and kPrimalFeasibility
    bool dual_reached = false;    ///< y met kPdhgTight relative and kDualFeasibility
    std::vector<double> x;        ///< unscaled; the polished x, or the input if not reached
    std::vector<double> y;        ///< unscaled, PDHG's sign; likewise
    Residuals residuals;          ///< (x, y) measured on the ORIGINAL problem
    Count iterations = 0;         ///< PDHG steps spent, primal and dual phases together
  };

  /// Polish the unscaled pair (x, y), at most `budget` accepted steps per phase, starting
  /// from primal weight `omega` and step `eta` of the main run. `stop` is polled every
  /// iteration (time limit, interrupt). Keeps a phase's result only when it reached the
  /// target; otherwise that half of the input comes back unchanged.
  [[nodiscard]] Outcome polish(const std::vector<double>& x, const std::vector<double>& y,
                               Count budget, double omega, double eta,
                               const std::function<bool()>& stop);

 private:
  struct PhaseResult {
    bool reached = false;
    std::vector<double> x;  ///< scaled
    std::vector<double> y;  ///< scaled
    Count iterations = 0;
  };

  /// Restarted PDHG ([PDLP] sections 3.1, 3.2, 4.3, the same rules as the main loop) on the
  /// scaled problem with cost `cost` and bounds given, from scaled (x, y). `judge` measures
  /// a scaled pair in original units; `done` decides when the phase has what it came for.
  PhaseResult run(const std::vector<double>& cost, const std::vector<double>& col_lower,
                  const std::vector<double>& col_upper, const std::vector<double>& row_lower,
                  const std::vector<double>& row_upper, const Problem& judged_on,
                  std::vector<double> x, std::vector<double> y, Count budget, double omega,
                  double eta, const std::function<bool(const Residuals&)>& done,
                  const std::function<bool()>& stop);

  void unscale(const std::vector<double>& xs, const std::vector<double>& ys,
               std::vector<double>& x, std::vector<double>& y) const;

  const Problem& problem_;
  const Scaling& scaling_;
  double spectral_norm_;

  // Primal feasibility problem: zero cost, original bounds.
  std::vector<double> zero_cost_;
  Problem primal_problem_;

  // Dual feasibility problem: original cost, finite bounds moved to zero.
  Model homogenized_;
  Problem dual_problem_;
  std::vector<double> h_col_lower_, h_col_upper_, h_row_lower_, h_row_upper_;

  // Scratch for evaluate().
  std::vector<double> activity_, reduced_;
};

}  // namespace sankhya::pdhg
