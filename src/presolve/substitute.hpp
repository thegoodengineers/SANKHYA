// SPDX-License-Identifier: Apache-2.0
// SANKHYA - column substitution from equality rows, an LP presolve stage with a stack
// postsolve (presolve_substitution, #1006 follow-up).
//
// References, written from the literature:
//   Andersen & Andersen, "Presolving in linear programming", Math. Programming 71 (1995),
//     sec. 5 (doubleton equations, with the bound of the eliminated column carried to its
//     partner) and sec. 6 (implied free columns substituted out of an equality row) - the
//     reductions and the dual recovery below.
//   Gondzio, "Presolve analysis of linear programs prior to applying an interior point
//     method", INFORMS J. Computing 9 (1997) - the fill-in limit on a substitution.
//
// WHY A SEPARATE STAGE. The reduction set of presolve.cpp recovers duals record by record
// with one unknown each, so a column may take part in one doubleton fold at most, and an
// implied free column is substituted only from a singleton. On the Netlib instances where
// SANKHYA takes the most iterations against HiGHS that is most of the difference: stocfor3
// keeps 15,929 of 16,675 rows where HiGHS's presolve keeps 8,259 (724 doubleton equations
// against 2,054, and 6,277 substitutions more), greenbea 1,929 against 951, ganges 968
// against 471. This stage runs on the model presolve.cpp hands over and undoes its records in
// strict reverse order, each one from the solution of the model that existed right after it
// fired, so chains of substitutions are exact by construction:
//
//   eliminate x_e from the equality row r:  a_re x_e + sum_{k != e} a_rk x_k = rhs.
//   Every other row i holding x_e becomes row_i - (a_ie / a_re) row_r, with its bounds moved
//   by the same multiple of rhs; every cost c_k moves by -(c_e / a_re) a_rk and the offset by
//   c_e rhs / a_re. Then for the reduced duals y' the original ones are y_i = y'_i on every
//   other row, and with d = c - A^T y,
//       y_r = (c_e - sum_{i != r} a_ie y_i) / a_re       (d_e = 0, x_e basic),
//   and x_e = (rhs - sum_{k != e} a_rk x_k) / a_re.
//
// x_e's bounds have to hold at that value. They do when x_e is IMPLIED FREE by row r (the
// range the other columns' bounds give a_re x_e lies inside its box), or, for a doubleton
// row a x_e + b x_k = rhs, when the box of x_e is carried onto x_k. In the second case x_k
// can end on a bound that only x_e had; then x_e is the one at its bound, and the dual of
// row r is chosen to zero x_k's reduced cost instead:
//       y_r = (c_k - sum_{i != r} a_ik y_i) / b,  which leaves d_e = -(a / b) d'_k,
// of the sign x_e's bound needs because d'_k had the sign x_k's carried bound needed.
// The quantities on the right are those of the model before the record, kept in it.
//
// LP only: a model with integer columns, a quadratic objective, semi-continuous columns or
// special ordered sets is returned unchanged.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::presolve {

struct Substitution {
  Index row = -1;          ///< the equality row removed, in the stage's input numbering
  Index column = -1;       ///< the column eliminated
  double coefficient = 0;  ///< a_re
  double rhs = 0;
  double cost = 0;   ///< c_e at the time
  double lower = 0;  ///< x_e's bounds at the time
  double upper = 0;
  /// Row r's other entries at the time (column, a_rk).
  std::vector<std::pair<Index, double>> row_entries;
  /// Column e's entries in the other rows at the time (row, a_ie).
  std::vector<std::pair<Index, double>> column_entries;
  /// Doubleton with x_e's box carried onto the partner: the partner, its coefficient b in
  /// row r, its bounds before the record, its cost and its entries in the other rows then.
  bool carried = false;
  Index partner = -1;
  double partner_coefficient = 0;
  double partner_lower = 0;
  double partner_upper = 0;
  double partner_new_lower = 0;  ///< and after: the carried box intersected with its own
  double partner_new_upper = 0;
  double partner_cost = 0;
  std::vector<std::pair<Index, double>> partner_entries;
};

struct SubstitutionResult {
  Model model;  ///< the reduced model
  /// The stage's records in the order they fired; an empty row removed has column == -1.
  std::vector<Substitution> records;
  std::vector<Index> col_to_input;
  std::vector<Index> row_to_input;
  Index carried = 0;       ///< doubleton equations with a carried box
  Index implied_free = 0;  ///< implied free substitutions
  [[nodiscard]] bool reduced_anything() const { return !records.empty(); }
};

/// Substitute columns out of `model`'s equality rows to a fixed point. Leaves `model` alone.
[[nodiscard]] SubstitutionResult substitute(const Model& model, const Options& options,
                                            Logger& logger);

/// The solution of `input` (the model substitute() was given) from the solution of
/// result.model, replaying the records in reverse.
[[nodiscard]] Solution undo_substitution(const SubstitutionResult& result, const Model& input,
                                         const Solution& reduced);

/// The stage around one simplex solve: `simplex` runs on the substituted model when
/// presolve_substitution is on and the stage removed enough, else on `model` itself, and the
/// answer comes back for `model`. Called where a cold LP simplex solve starts (src/core/
/// solve.cpp, the dual simplex engine), so neither the interior point nor PDHG ever sees a
/// substituted model, whichever way they were chosen (#1125: under them the crossover on
/// cycle stopped short of a vertex and capri's dense-column directions lost accuracy).
[[nodiscard]] Solution solve_substituted(const Model& model, const Options& options,
                                         Logger& logger,
                                         const std::function<Solution(const Model&)>& simplex);

/// How many solves solve_substituted() has run on a substituted model, process-wide: the
/// test that holds the interior point and PDHG off the stage reads it.
[[nodiscard]] std::int64_t substitutions_applied_for_testing();

}  // namespace sankhya::presolve
