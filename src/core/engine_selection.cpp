// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP engine selection behind `algorithm=auto` (#284). See the header for
// the thresholds and the measurements each one rests on.

#include "engine_selection.hpp"

#include <fmt/format.h>

#include <utility>

#include "engine_features.hpp"
#include "engine_selection_tree.hpp"

namespace sankhya {
namespace {

/// The hand-written rule table (#284, #417), with the explicit request and the warm start
/// in front of it.
EngineSelection rule_table_selection(const Model& model, const Options& options,
                                     bool warm_start, bool gpu_available,
                                     std::string gpu_device, bool device_factor) {
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
  if (device_factor && work >= kSimplexWorkCeiling) {
    // qap15 (#417): optimal on the device interior point in 23 s where the dual simplex
    // times out at 300 s. ONLY WITH THE DEVICE FACTOR: without it the rule would send
    // Kennington's ken-11 (7.2e8) and pds-06 off the dual simplex, which solves them in
    // about 2.5 s, onto CPU PDHG, which took 84 s and 35 s and returns no basis. qap15 is
    // the one model in the measured sets that loses by staying, and it loses less than the
    // Kennington pair would.
    s.algorithm = "ipm";
    s.rule = "work:ipm-device";
    s.reason = fmt::format(
        "{}: rows x nonzeros is {:.3g}, over {:.0e}, beyond the dual simplex's measured reach "
        "(dfl001 at 2.2e8 solves on it in 47 s; qap15 at 6.0e8 times out at 300 s and is "
        "optimal on the interior point with the device factor in 23 s; #417)",
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

}  // namespace

EngineSelection select_engine(const Model& model, const Options& options, bool warm_start,
                              bool gpu_available, std::string gpu_device, bool device_factor) {
  EngineSelection rules = rule_table_selection(model, options, warm_start, gpu_available,
                                               std::move(gpu_device), device_factor);
  // An explicit engine and a starting basis are decided before any table, learned or not.
  if (options.get_string("algorithm_selection") != "learned" || rules.rule == "requested" ||
      rules.rule == "warm-start") {
    return rules;
  }
  // THE LEARNED TREE (#477), within the range it was trained on. Its training runs were on
  // the CPU, so with a device the rule table's measured GPU rules (#417) decide; past the
  // largest training model, the rule table's thresholds, measured up to the 779,640-row
  // refinery, decide. Either way the reason says the tree was not consulted and why.
  const LearnedTreeDomain domain = learned_tree_domain();
  std::string not_consulted;
  if (gpu_available || device_factor) {
    not_consulted = "it was trained on CPU timings and this run has a GPU";
  } else if (rules.rows > domain.max_rows) {
    not_consulted = fmt::format("{} rows, over the {} of the largest model it was trained on",
                                rules.rows, domain.max_rows);
  } else if (rules.nonzeros > domain.max_nonzeros) {
    not_consulted =
        fmt::format("{} nonzeros, over the {} of the largest model it was trained on",
                    rules.nonzeros, domain.max_nonzeros);
  }
  if (!not_consulted.empty()) {
    rules.reason = fmt::format("learned tree not consulted ({}); rule table: {}", not_consulted,
                               rules.reason);
    return rules;
  }
  const LearnedTreeChoice choice =
      learned_engine_tree(compute_engine_features(model, /*symbolic=*/true));
  EngineSelection s = rules;
  s.algorithm = choice.algorithm;
  s.use_gpu = false;
  s.rule = "learned:" + choice.algorithm;
  s.reason = fmt::format(
      "learned tree: {} -> {} ({} rows, {} columns, {} nonzeros; src/core/"
      "engine_selection_tree.cpp, #477; the rule table would choose {})",
      choice.path.empty() ? "a single leaf" : choice.path, choice.algorithm, s.rows, s.columns,
      s.nonzeros, rules.algorithm);
  return s;
}

}  // namespace sankhya
