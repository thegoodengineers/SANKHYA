// SPDX-License-Identifier: Apache-2.0
// SANKHYA - K linear programs that share A and differ in their column boxes, bounded in one
// batched PDHG run (#520): a branch-and-bound node's box, or a strong-branching child's.
//
// References, written from the papers (ENGINEERING_RULES.md red line: no solver source was
// consulted, PDLP / cuPDLP / cuPDLPx included):
//   [CP11]  Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//           applications to imaging", JMIV 40(1), 2011 - Algorithm 1, the iteration.
//   [PDLP]  Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue & Schudy, "Practical large-scale
//           linear programming using primal-dual hybrid gradient", NeurIPS 2021 - restarts
//           to the average (sec. 3.2), the primal weight and its update (sec. 3.3), Ruiz plus
//           Pock-Chambolle preconditioning (sec. 3.4).
//   [cuPDLP] Lu & Yang, "cuPDLP.jl: A GPU implementation of restarted primal-dual hybrid
//           gradient for linear programming in Julia", arXiv:2311.12180 (2023) - the restart
//           test on the weighted KKT error, and a constant step 1 / ||A||_2 in place of the
//           adaptive one on the device.
//   [NS04]  Neumaier & Shcherbina, "Safe bounds in linear and mixed-integer linear
//           programming", Math. Programming 99 (2004) 283-296 - the bound (core/safe_bound).
//   [SBB]   arXiv:2601.21990 (2026), batched first-order strong branching on the GPU: the use
//           the issue names; its protocol is not copied, only the idea of scoring children
//           by a batched first-order bound.
//
// WHAT IS RETURNED, AND WHY IT IS SAFE AT ANY ITERATION COUNT. The batch runs a fixed budget
// of iterations and is then asked for nothing but its duals. LP k's bound is the
// Neumaier-Shcherbina bound of those duals against LP k's OWN box, computed with outward
// rounding on the unscaled problem. That bound holds for every multiplier vector - an
// optimal y only makes it tight - so a bound from ten iterations is as valid as one from ten
// thousand, merely weaker, and a diverged run gives -inf, never a wrong number. That is what
// lets branch and bound prune on it; nothing here ever claims a PRIMAL point is optimal.
#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::pdhg {

/// K LPs:  min c'x  s.t.  model.row_lower <= model.matrix x <= model.row_upper,
///         col_lower_k <= x <= col_upper_k.
struct BatchProblem {
  /// The rows and the matrix. Its column bounds and cost are not read.
  const Model* model = nullptr;
  /// Minimise-space cost (sense * c), one entry per column.
  std::vector<double> cost;
  Index count = 0;  ///< K
  /// LP k's box is entries [k * n, (k + 1) * n).
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  /// A shared starting point, projected into each box; empty means zero.
  std::vector<double> primal_start;
  /// Shared starting multipliers in minimise space (y_i > 0 prices the lower side), or empty
  /// for zero. Shorter than the rows is padded with zeros (cut rows appended since).
  std::vector<double> dual_start;
};

enum class BatchBackendKind {
  kCpu,
  /// The CUDA backend when the build has one and a device answers, else the CPU (the reason
  /// is in BatchResult::note).
  kDevice,
};

struct BatchSettings {
  Count iterations = 0;  ///< per LP; 0 bounds from the starting multipliers alone
  /// Wall-clock seconds the run may take; it stops at the next restart check once over.
  /// Stopping early is safe for the same reason a small budget is.
  double time_limit = 1e300;
  BatchBackendKind backend = BatchBackendKind::kCpu;
};

struct BatchResult {
  /// Per LP: a guaranteed lower bound on its optimum, minimise space, without the model's
  /// objective offset. -inf when the multipliers prove nothing finite.
  std::vector<double> bound;
  /// The multipliers each bound came from, unscaled, minimise space, LP k at [k * m, ...).
  std::vector<double> dual;
  std::vector<Count> restarts;  ///< per LP
  Count iterations = 0;         ///< iterations run (the same for every LP)
  bool on_device = false;
  std::string note;  ///< why the device was not used, or why the run gave up
  double seconds = 0.0;
};

/// Run the batch and bound every LP. Never throws on a numerical failure: an LP whose
/// multipliers prove nothing gets -inf.
[[nodiscard]] BatchResult solve_batch(const BatchProblem& problem,
                                      const BatchSettings& settings);

}  // namespace sankhya::pdhg
