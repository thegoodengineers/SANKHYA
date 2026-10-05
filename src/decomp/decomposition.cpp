// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the decomposition entry point (#525). See decomposition.hpp.

#include "decomp/decomposition.hpp"

#include <string>
#include <utility>

#include "decomp/benders.hpp"
#include "decomp/structure.hpp"

namespace sankhya::decomp {

std::optional<Solution> solve_by_decomposition(const Model& model, const Options& options,
                                               SolveControl* control, Logger& logger) {
  const std::string mode = options.get_string("decomposition");
  // What this module can decompose: a minimisation LP.
  std::string why_not;
  if (model.has_integrality()) {
    why_not = "the model has integer columns";
  } else if (model.has_quadratic_objective()) {
    why_not = "the objective is quadratic";
  } else if (model.has_semicontinuous_or_sos()) {
    why_not = "the model has semi-continuous or special ordered set columns";
  } else if (model.sense == ObjSense::kMaximize) {
    why_not = "the model maximises (only minimisation is decomposed)";
  } else if (model.num_rows() < 2 || model.num_cols() < 2) {
    why_not = "the model is too small";
  }
  if (!why_not.empty()) {
    logger.info("Decomposition: not applied, {}", why_not);
    return std::nullopt;
  }

  DetectionOptions detect;
  detect.blocks = static_cast<Index>(options.get_int("decomposition_blocks"));
  detect.max_linking_fraction = options.get_double("decomposition_max_linking");
  if (mode == "benders") {
    // Forced: any split into two blocks is tried, however weak.
    detect.max_linking_fraction = 1.0;
    detect.max_largest_share = 1.0;
  }
  const Detection columns = detect_block_structure(model, Linking::kColumns, detect);
  if (!columns.accepted) {
    logger.info("Decomposition: no linking-column structure strong enough ({})",
                columns.description);
    // Linking rows are what Dantzig-Wolfe would use. Looked for so the log can say so; not
    // built.
    const Detection rows = detect_block_structure(model, Linking::kRows, detect);
    if (rows.accepted) {
      logger.info(
          "Decomposition: the model has linking-row structure ({}); that is the form for "
          "Dantzig-Wolfe decomposition, which is not implemented, so it is solved "
          "monolithically",
          rows.description);
    }
    return std::nullopt;
  }
  logger.info("Decomposition: {} - {}", to_string(Linking::kColumns), columns.description);

  BendersOutcome outcome = solve_benders(model, columns.structure, options, control, logger);
  if (!outcome.solution.has_value()) {
    logger.info("Decomposition: Benders declined ({}); solving monolithically",
                outcome.declined);
    return std::nullopt;
  }
  return std::move(outcome.solution);
}

}  // namespace sankhya::decomp
