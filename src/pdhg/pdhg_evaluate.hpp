// SPDX-License-Identifier: Apache-2.0
// SANKHYA - PDHG convergence evaluation, shared by all PDHG engines.
//
// Extracted from src/pdhg/pdhg.cpp to eliminate duplication across the CPU, single-GPU
// and multi-GPU PDHG engines.
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::pdhg {

/// Convergence measured in the ORIGINAL problem, never the scaled one. Reporting residuals
/// from the scaled problem would let a well-chosen preconditioner flatter the result.
struct Residuals {
  double primal = 0.0;  ///< relative primal infeasibility
  double dual = 0.0;    ///< relative dual infeasibility
  double gap = 0.0;     ///< relative primal-dual objective gap
  double primal_objective = 0.0;
  double dual_objective = 0.0;

  /// The same violations UNSCALED. A relative residual divides by (1 + ||bounds||), so on a
  /// model whose right-hand sides run to 1e4 a relative 1e-8 still permits an absolute
  /// violation around 1e-4. That is standard and fine as a stopping rule - it is what the
  /// PDLP literature uses - but it is NOT the standard the rest of this project reports
  /// against, and conflating the two is how the solver ends up stamping "optimal" on a point
  /// tools/verify_solution.py then rejects.
  double absolute_primal = 0.0;
  double absolute_dual = 0.0;

  /// The duality gap normalised the way tools/verify_solution.py normalises it, by
  /// max(1, |primal objective|), rather than the PDLP convention of 1 + |primal| + |dual|.
  /// The two differ by roughly a factor of two, which is more than enough for this engine to
  /// pass its own optimality test and fail the verifier's on the same point. The PDLP form
  /// stays as the stopping rule because that is the literature convention; the claim is
  /// judged by the verifier's form because that is what will be checked.
  double gap_as_verified = 0.0;

  /// max |multiplier| * slack over rows and columns - the same product form
  /// tools/verify_solution.py uses. The duality gap implies this only in the limit, so a
  /// point can show a tiny gap and still price a constraint it is not sitting on.
  double complementarity = 0.0;

  [[nodiscard]] double worst() const { return std::max({primal, dual, gap}); }

  /// Has the run met the tolerance the CALLER asked for, AND is the point actually feasible
  /// in absolute terms? Both are required to stop.
  ///
  /// The absolute half is not pedantry. kFeasible in sankhya::Solution asserts that a
  /// feasible point is being reported, so stopping on a relative residual alone would let
  /// this engine claim feasibility for a point that misses the project's own primal
  /// tolerance - a weaker claim than kOptimal, but still one the verifier rejects.
  [[nodiscard]] bool meets_request(double tolerance) const {
    return primal <= tolerance && dual <= tolerance && gap <= tolerance &&
           absolute_primal <= tol::kPrimalFeasibility;
  }

  /// Would this point survive independent verification? These are the project's own
  /// tolerances from include/sankhya/tolerances.hpp, the same ones the .sol file is judged
  /// against, and meeting them is the ONLY basis on which this engine claims kOptimal.
  [[nodiscard]] bool meets_project_standard() const {
    return absolute_primal <= tol::kPrimalFeasibility &&
           absolute_dual <= tol::kDualFeasibility && gap_as_verified <= tol::kDualityGap &&
           complementarity <= 1e-6;
  }
};

/// The unscaled problem, held once so the convergence test does not rebuild it.
struct Problem {
  const Model* model = nullptr;
  std::vector<double> cost;  ///< minimise-space objective
  double bound_norm = 0.0;   ///< ||finite row bounds||, for the relative primal residual
  double cost_norm = 0.0;    ///< ||c||, for the relative dual residual
};

[[nodiscard]] double euclidean_norm(const std::vector<double>& v);

/// Compute the unscaled residuals for a candidate (x, y).
[[nodiscard]] Residuals evaluate(const Problem& problem, const std::vector<double>& x,
                                 const std::vector<double>& y, std::vector<double>& activity,
                                 std::vector<double>& reduced);

/// FIRST CROSSINGS OF THE RELATIVE KKT ERROR (#486): the levels published PDLP figures and
/// Mittelmann's feasibility page quote, recorded by both the CPU and the CUDA engine so one
/// run yields all three times.
inline constexpr double kKktCrossingLevels[3] = {1e-4, 1e-6, 1e-8};

}  // namespace sankhya::pdhg
