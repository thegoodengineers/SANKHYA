// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP engine selection behind `algorithm=auto` (#284). See the header for
// the thresholds and the measurements each one rests on.

#include "engine_selection.hpp"

#include <fmt/format.h>

#include "engine_features.hpp"

namespace sankhya {

EngineSelection select_engine(const Model& model, const Options& options, bool warm_start,
                              bool gpu_available, std::string gpu_device, bool device_factor) {
  EngineSelection s;
  s.rows = model.num_rows();
  s.columns = model.num_cols();
  s.nonzeros = model.num_nonzeros();
  s.warm_start = warm_start;
  const std::string requested = options.get_string("algorithm");
  const std::string shape =
      fmt::format("{} rows, {} columns, {} nonzeros", s.rows, s.columns, s.nonzeros);

  if (requested != "auto") {
    s.algorithm = requested;
    s.rule = "requested";
    s.reason = fmt::format("algorithm={} was asked for ({})", requested, shape);
    return s;
  }
  if (warm_start) {
    // Only a simplex can start from a basis; the dual is the right restart after a bound
    // or right-hand-side edit (#218), which is what a re-solve usually is.
    s.algorithm = "dual-simplex";
    s.rule = "warm-start";
    s.reason =
        fmt::format("a starting basis was given, and only a simplex can use one ({})", shape);
    return s;
  }
  // One O(m + n + nnz) pass (engine_features.cpp) for the bound on the normal equations.
  const EngineFeatures features = compute_engine_features(model);
  const double normal_bound = features.normal_equations_nnz_bound;
  const bool normal_affordable = normal_bound < kNormalEquationsCeiling;
  if (device_factor && s.rows >= kDualSimplexRowLimit && normal_affordable) {
    s.algorithm = "ipm";
    s.rule = "size:ipm-device";
    s.reason = fmt::format(
        "{}: the normal equations are bounded by {:.3g} nonzeros, under {:.0e}, and this "
        "build factors them on the device (cuDSS, #489); rmine15 at 358,395 rows is optimal "
        "there in 137 s where PDHG stops short of the tolerance (#417)",
        shape, normal_bound, kNormalEquationsCeiling);
    return s;
  }
  if (s.rows >= kPdhgRowFloor) {
    s.algorithm = "pdhg";
    if (gpu_available && !gpu_device.empty()) {
      s.use_gpu = true;
      s.rule = "size:pdhg-gpu";
      s.reason = fmt::format(
          "{}: at {} rows and above the direct factorization is out of reach; GPU PDHG "
          "selected (device: {}; crossover measured in docs/BENCHMARKS.md section 1g)",
          shape, kPdhgRowFloor, gpu_device);
    } else {
      s.rule = "size:pdhg";
      s.reason = fmt::format(
          "{}: at {} rows and above the direct factorization is out of reach at the time "
          "limit and the first-order method reaches the optimum (scale-e134aeb.csv, "
          "scale-refinery-e134aeb.csv, docs/BENCHMARKS.md 1f)",
          shape, kPdhgRowFloor);
    }
    return s;
  }
  if (s.rows >= kDualSimplexRowLimit && !normal_affordable) {
    s.algorithm = "pdhg";
    s.rule = "normal:pdhg";
    s.reason = fmt::format(
        "{}: the normal equations are bounded by {:.3g} nonzeros, over {:.0e}; every model "
        "in the Mittelmann set that large had a factor the interior point declined after its "
        "set-up share (chromaticindex1024-7, Linf_520c, supportcase10, bdry2), and PDHG "
        "solves chromaticindex1024-7 in seconds (#417)",
        shape, normal_bound, kNormalEquationsCeiling);
    return s;
  }
  if (s.rows >= kDualSimplexRowLimit) {
    s.algorithm = "ipm";
    s.rule = "size:ipm";
    s.reason = fmt::format(
        "{}: from {} rows the dual simplex hits the time limit on every measured shape and "
        "the interior point solves the structured ones (scale-staircase-e134aeb.csv, "
        "scale-refinery-e134aeb.csv, docs/BENCHMARKS.md 1f.2-1f.3)",
        shape, kDualSimplexRowLimit);
    return s;
  }
  if (s.nonzeros >= kIpmNonzeroFloor && s.rows >= kIpmDensityRowFloor) {
    s.algorithm = "ipm";
    s.rule = "density:ipm";
    s.reason = fmt::format(
        "{}: a model this dense times out on the dual simplex and solves on the interior "
        "point with crossover (maros-r7, 144,848 nonzeros: 14 s against the 120 s limit; "
        "netlib-full-b3f1660.csv)",
        shape);
    return s;
  }
  const double work = static_cast<double>(s.rows) * static_cast<double>(s.nonzeros);
  if (work >= kSimplexWorkCeiling) {
    s.algorithm = "ipm";
    s.rule = "work:ipm";
    s.reason = fmt::format(
        "{}: rows x nonzeros is {:.3g}, over {:.0e}, beyond the dual simplex's measured "
        "reach (dfl001 at 2.2e8 solves on it in 47 s, qap15 at 6.0e8 times out at 300 s and "
        "is optimal on the interior point; #417)",
        shape, work, kSimplexWorkCeiling);
    return s;
  }
  s.algorithm = "dual-simplex";
  s.rule = "default:dual-simplex";
  s.reason = fmt::format(
      "{}: below {} rows, and below {} nonzeros or {} rows, the dual simplex is the "
      "measured default, 80 of 89 on the Netlib full set (netlib-full-b3f1660.csv); it "
      "produces the basis the rest of the pipeline uses",
      shape, kDualSimplexRowLimit, kIpmNonzeroFloor, kIpmDensityRowFloor);
  return s;
}

}  // namespace sankhya
