// SPDX-License-Identifier: Apache-2.0
// SANKHYA - which LP engine `algorithm=auto` means (#284): a rule-based selection from
// the model's shape, returned as a structured decision with the rule and the measurement
// behind it, so the choice is explainable and reproducible rather than a threshold buried
// in the dispatcher.
//
// The rules encode MEASUREMENTS, not intuition, and every threshold below names the CSV it
// comes from. Where the evidence does not reach, the rule says so in its reason. No machine
// learning; the first version is deliberately a short table anyone can read.
#pragma once

#include <string>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

/// The engines the selection can name. Strings, not an enum, because they are the same
/// words the `algorithm` option takes and the answer's `algorithm` field reports.
struct EngineSelection {
  std::string algorithm;  ///< "dual-simplex", "simplex", "ipm" or "pdhg"
  std::string rule;       ///< short tag of the rule that fired, e.g. "requested", "size:ipm"
  std::string reason;     ///< one sentence naming the statistics and the measurement
  Index rows = 0;
  Index columns = 0;
  Count nonzeros = 0;
  bool warm_start = false;
  bool use_gpu = false;  ///< true when the selection is GPU PDHG (size:pdhg-gpu)
};

/// Thresholds, each tied to a row of `docs/BENCHMARKS.md`:
///
/// - kDualSimplexRowLimit: below this many rows the dual simplex is the measured default:
///   80 of 89 on the Netlib full set (`netlib-full-b3f1660.csv`), whose largest instance
///   has 6,071 rows; at 20,000 rows on both generated shapes it hits the 120 s limit where
///   the interior point solves the staircase in 19 s (`scale-staircase-e134aeb.csv`).
///   The 5,000-row generated instances sit BELOW this limit and go to the dual simplex,
///   which times out on them while the interior point solves them; the limit is set where
///   the Netlib evidence ends, not where the generated evidence begins, because dfl001
///   (6,071 rows) solves in 47 s on the dual and not at all on the interior point at 120 s.
/// - kIpmNonzeroFloor: a dense model below the row limit goes to the interior point:
///   maros-r7 (3,136 rows, 144,848 nonzeros) times out on the dual simplex and solves in
///   14 s on the interior point with crossover; no Netlib instance under 100,000 nonzeros
///   is measured to prefer it (fit2p, 50,284 nonzeros, times out on it).
/// - kIpmDensityRowFloor: the density rule applies from this many rows. Its evidence is a
///   basis factorization the dual simplex cannot afford, and a model with a few hundred
///   rows has no such basis however many nonzeros its columns hold: fit2d (25 rows, 10,500
///   columns, 129,018 nonzeros after presolve) solves in 0.09 s on the dual simplex against
///   0.18 s on the interior point with crossover. maros-r7, the rule's reason, has 3,136.
/// - kPdhgRowFloor: from 100,000 rows the direct factorization is out of reach
///   (`scale-e134aeb.csv`: the interior point produces no output on the 100,000-row random
///   shape and hits the limit on the staircase; PDHG reaches the optimum to 1e-7 on both),
///   and the 779,640-row refinery year is reached only by PDHG
///   (`scale-refinery-e134aeb.csv`).
inline constexpr Index kDualSimplexRowLimit = 20000;
inline constexpr Count kIpmNonzeroFloor = 100000;
inline constexpr Index kIpmDensityRowFloor = 1000;
inline constexpr Index kPdhgRowFloor = 100000;

/// THE MITTELMANN RULES (#417), each read off the per-engine attribution of the eight
/// smallest Mittelmann LPs at 300 s on an A100 node (the table in the PR that added them):
///
/// - kNormalEquationsCeiling, on EngineFeatures::normal_equations_nnz_bound (the pattern
///   bound sum_j c_j^2 on A A^T, computed before anything is assembled). At or above
///   kDualSimplexRowLimit rows, a model whose bound is over this goes to PDHG instead of
///   the interior point: every such model in the set had a normal-equations factor the
///   interior point declined on BOTH factors after the set-up share of the limit
///   (chromaticindex1024-7, bound 6.0e8, factor 1.0e9 on cuDSS; Linf_520c 9.5e8, 1.0e9;
///   supportcase10 4.4e8, 7.8e9; bdry2 1.6e10), and chromaticindex1024-7 is then solved by
///   PDHG in seconds where auto reached it only after a 60 s decline. Every model under it
///   had a factor the device could hold (rmine15 1.9e7, irish-electricity 2.1e7, brazil3
///   8.7e5, qap15 5.6e5). 1e8 sits between the two groups (2.1e7 and 4.4e8) at their
///   geometric midpoint, rounded.
/// - With a device factor (a cuDSS build with a CUDA device, ipm_linear_solver not cpu), a
///   model at or above kDualSimplexRowLimit rows whose bound is under the ceiling goes to
///   the interior point whatever its row count: rmine15 (358,395 rows) is optimal on it in
///   137 s where PDHG, which the row floor chose, stops short of the project tolerance.
/// - kSimplexWorkCeiling, on rows x nonzeros, a proxy for the dual simplex's work (about
///   one pass over the nonzeros per iteration, iterations a small multiple of the rows):
///   below the row limit and the nonzero floor, a model over it goes to the interior point.
///   qap15 (6,330 x 94,950 = 6.0e8) times out on the dual simplex at 300 s and is optimal
///   on the interior point; the largest Netlib product the dual simplex solves is dfl001
///   (6,071 x 35,632 = 2.2e8, 47 s), and no other Netlib model is above 1.5e8 except
///   maros-r7 (4.5e8), which kIpmNonzeroFloor already sends to the interior point.
inline constexpr double kNormalEquationsCeiling = 1e8;
inline constexpr double kSimplexWorkCeiling = 3e8;

/// Decide the LP engine for `model` under `options`. An explicit `algorithm` other than
/// "auto" is honoured as given (rule "requested"); a starting basis forces a simplex
/// (rule "warm-start"), because only a simplex can use one.
///
/// `gpu_available` and `gpu_device` carry the result of a device probe at the call site.
/// When `gpu_available` is true and the model is at or above `kPdhgRowFloor`, the selection
/// sets `use_gpu` and uses rule "size:pdhg-gpu" with the device named in `reason`.
/// Passing the probe result in rather than probing here lets tests stub it without a GPU.
///
/// `device_factor` says the interior point would factor large normal equations on the
/// device (#489: a cuDSS build, a CUDA device, ipm_linear_solver auto or cudss); the call
/// site probes it, as it probes the GPU for PDHG.
[[nodiscard]] EngineSelection select_engine(const Model& model, const Options& options,
                                            bool warm_start, bool gpu_available = false,
                                            std::string gpu_device = {},
                                            bool device_factor = false);

}  // namespace sankhya
