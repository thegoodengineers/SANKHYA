// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the decomposition entry point (#525). See decomposition.hpp.

#include "decomp/decomposition.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "decomp/benders.hpp"
#include "decomp/dantzig_wolfe.hpp"
#include "decomp/structure.hpp"

namespace sankhya::decomp {

BendersForecast forecast_benders(const Model& model, const BlockStructure& structure) {
  std::vector<double> block_nonzeros(static_cast<std::size_t>(structure.blocks), 0.0);
  double nonzeros = 0.0;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      if (column.values[k] == 0.0) continue;
      nonzeros += 1.0;
      const Index b = structure.row_block[static_cast<std::size_t>(column.rows[k])];
      if (b >= 0) block_nonzeros[static_cast<std::size_t>(b)] += 1.0;
    }
  }
  BendersForecast f;
  f.monolithic = static_cast<double>(model.num_rows()) * nonzeros;
  double largest = 0.0;
  for (Index b = 0; b < structure.blocks; ++b) {
    const auto u = static_cast<std::size_t>(b);
    largest =
        std::max(largest, static_cast<double>(structure.block_rows[u]) * block_nonzeros[u]);
  }
  f.rounds = static_cast<double>(structure.linking_columns) + 1.0;
  f.decomposed = f.rounds * largest;
  return f;
}

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
  if (mode == "dantzig-wolfe") {
    // Forced, like `benders`, over linking ROWS (#1048).
    detect.max_linking_fraction = 1.0;
    detect.max_largest_share = 1.0;
    const Detection rows = detect_block_structure(model, Linking::kRows, detect);
    if (!rows.accepted) {
      logger.info("Decomposition: no linking-row structure ({})", rows.description);
      return std::nullopt;
    }
    logger.info("Decomposition: {} - {}", to_string(Linking::kRows), rows.description);
    BendersOutcome dw = solve_dantzig_wolfe(model, rows.structure, options, control, logger);
    if (!dw.solution.has_value()) {
      logger.info("Decomposition: Dantzig-Wolfe declined ({}); solving monolithically",
                  dw.declined);
      return std::nullopt;
    }
    return std::move(dw.solution);
  }
  const Detection columns = detect_block_structure(model, Linking::kColumns, detect);
  if (!columns.accepted) {
    logger.info("Decomposition: no linking-column structure strong enough ({})",
                columns.description);
    // Linking rows are what Dantzig-Wolfe uses. `auto` does not run it (it is off until
    // measured on main, #1048); looked for so the log can say so.
    const Detection rows = detect_block_structure(model, Linking::kRows, detect);
    if (rows.accepted) {
      logger.info(
          "Decomposition: the model has linking-row structure ({}); that is the form for "
          "Dantzig-Wolfe decomposition, which `auto` does not run "
          "(decomposition=dantzig-wolfe does), so it is solved monolithically",
          rows.description);
    }
    return std::nullopt;
  }
  logger.info("Decomposition: {} - {}", to_string(Linking::kColumns), columns.description);
  const BendersForecast f = forecast_benders(model, columns.structure);
  logger.info(
      "Decomposition: forecast work {:.3g} for Benders ({:.0f} round(s) x the largest block's "
      "rows x nonzeros) against {:.3g} monolithic (rows x nonzeros)",
      f.decomposed, f.rounds, f.monolithic);
  if (options.get_bool("decomposition_predict") && !f.pays()) {
    logger.info(
        "Decomposition: not applied, the forecast says Benders cannot pay here; solving "
        "monolithically at once (decomposition_predict=false tries it anyway)");
    return std::nullopt;
  }

  BendersOutcome outcome = solve_benders(model, columns.structure, options, control, logger);
  if (!outcome.solution.has_value()) {
    logger.info("Decomposition: Benders declined ({}); solving monolithically",
                outcome.declined);
    return std::nullopt;
  }
  return std::move(outcome.solution);
}

}  // namespace sankhya::decomp
