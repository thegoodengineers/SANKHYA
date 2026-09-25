// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the KKT conditions of a nonlinear model at a point, in the model's own terms
// (NLP stage 2).
//
// WHAT DECIDES `optimal` / `locally_optimal`. The interior point's own error measure is
// scaled (Wachter and Biegler 2006, eq. 5) and lives in its slack-augmented space; the
// project's standard is the verifier's (tools/verify_solution.py), in the model's rows and
// columns. So the solve stops when THIS check passes, and the status is decided by it. The
// measures are the verifier's LP measures with the gradient and the Jacobian at x in place
// of c and A - the same substitution the verifier makes for a QP, carried one step further:
//
//   column bounds       max(l - x, x - u, 0) / max(1, |x_j|)              <= primal
//   rows                max(L - g(x), g(x) - U, 0) / max(1, |g_i(x)|)     <= primal
//   stationarity        |grad f - J^T y - d|_j / max(1, |grad f_j|, max_i |y_i J_ij|) <= dual
//   multiplier signs    a positive multiplier needs a finite lower bound, a negative one a
//                       finite upper bound (column: scaled as stationarity; row: by max|y|)
//   complementarity     |multiplier| * min(slack to the finite bounds)    <= complementarity
//                       (|multiplier| alone where no bound is finite)
//
// everything in MINIMISATION space: grad f is the gradient of the model objective times
// +1 / -1, y and d the row and column multipliers times the same sign (verify_solution.py's
// convention: a positive multiplier prices a lower bound). Nocedal and Wright, "Numerical
// Optimization", 2nd ed. (2006), thm. 12.1 for the conditions themselves.
//
// It certifies a KKT point, which is a LOCAL optimum of a smooth problem under a constraint
// qualification, and a global one only when the problem is convex - which is a separate
// finding (NonlinearModel::convexity()), not something this check can see.
#pragma once

#include <string>

#include "nlp/barrier_nlp.hpp"
#include "nlp/nlp_problem.hpp"

namespace sankhya::nlp {

struct NlpKktReport {
  bool evaluated = false;  ///< false when the functions are undefined at x
  bool primal_ok = false;
  bool passed = false;           ///< every condition, primal and dual
  std::string failed;            ///< the first condition that failed, in words
  double primal = 0.0;           ///< worst scaled bound or row violation
  double primal_absolute = 0.0;  ///< its unscaled value
  double stationarity = 0.0;     ///< worst scaled |grad f - J^T y - d|
  double sign = 0.0;             ///< worst scaled sign violation
  double complementarity = 0.0;  ///< worst |multiplier| * slack
};

/// The check at x with row multipliers y (m) and column multipliers d (n), both in
/// minimisation space. `primal`, `dual`, `complementarity` are the thresholds.
[[nodiscard]] NlpKktReport check_nlp_kkt(const NlpProblem& problem, const Vec& x, const Vec& y,
                                         const Vec& d, double primal, double dual,
                                         double complementarity);

}  // namespace sankhya::nlp
