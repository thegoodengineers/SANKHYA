// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the debug-solution check (#500): every cut, every bound tightening and every
// presolve reduction of a MILP solve tested against a known feasible or optimal point.
//
// WHY. A cut that removes the optimum, or a reduction that does, gives a wrong answer that
// looks right: the search proves the second-best point optimal, the output is integral and
// feasible and the bound meets it. The rational oracle checks cut families one at a time in
// tests/unit/test_cuts.cpp; this checks them where the solver uses them, inside a real
// search, with every family and every reduction switched on together.
//
// HOW. `debug_solution` names a solution file (the `begin columns` block of a SANKHYA .sol,
// or plain `name value` lines as in a MIPLIB .sol). The point is read by column name, so it
// follows presolve's column removals without any mapping of indices. Then:
//
//   - presolve: every column a reduction fixes must be fixed at the point's value, and the
//     reduced model must still contain the point (src/core/presolve_pipeline.cpp);
//   - integral row rounding: the rounded rows must still contain it;
//   - every cut row a round appends, root or tree, must be satisfied by it;
//   - at every node whose box contains it: propagation must keep it, the node LP must not
//     be infeasible, and the node's bound must not exceed its objective;
//   - reduced-cost fixing must keep it while the incumbent is worse than it.
//
// The first check that fails prints the family, the round, the node and the row (or the
// reduction and the column) to stderr and aborts the process. Off unless the option is set,
// and nothing here runs then.
//
// WHAT "CUT OFF" MEANS FOR AN OPTIMALITY-BASED REDUCTION. Dual fixing, dominated columns,
// symmetry ordering rows and reduced-cost fixing may remove an optimal point legitimately
// when it is one of several optima: they promise to keep AN optimum, not every one. Against
// the unique optimum they must keep it, and that is what the fuzz in
// tests/unit/test_debug_solution.cpp supplies. With a debug solution that is not the unique
// optimum a failure on one of these is reported as what it is (the message names the
// reduction as optimality-based) and says to supply the unique optimum.
//
// The mechanism is the one solver user manuals describe as a debug solution; this
// implementation is ours, written from that description. The validity notion it tests is
// the textbook one: an inequality is valid for a MILP when every feasible point satisfies it
// (Nemhauser & Wolsey, "Integer and Combinatorial Optimization", Wiley 1988, ch. II.1), so
// one feasible point that violates it is a proof of invalidity.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "cuts.hpp"

namespace sankhya {
namespace presolve {
struct Result;
}  // namespace presolve

namespace mip {

/// A known point, in the column space of one model.
class DebugSolution {
 public:
  /// Read `path` and pick out a value for every column of `model` by name (a column without
  /// a name is `C<index>`, as the solution writer prints it). A column the file does not
  /// name is taken as 0, the MIPLIB .sol convention, and counted in `missing`. Returns
  /// nullopt with `error` filled when the file cannot be read or names none of the columns.
  [[nodiscard]] static std::optional<DebugSolution> load(const std::string& path,
                                                         const Model& model, std::string* error,
                                                         Index* missing = nullptr);

  explicit DebugSolution(std::vector<double> x) : x_(std::move(x)) {}

  [[nodiscard]] const std::vector<double>& x() const noexcept { return x_; }

  /// Empty when the point satisfies `cut` (sum coeff x <= rhs, to kDebugSolutionTolerance
  /// relative); otherwise the activity, the right-hand side and the excess.
  [[nodiscard]] std::string cut_violation(const Cut& cut) const;

  /// Empty when the point is inside `model`'s column bounds, integral on its integer
  /// columns and inside every row's bounds; otherwise the first thing it violates, named.
  [[nodiscard]] std::string model_violation(const Model& model) const;

  /// The first column whose box [lower, upper] excludes the point, or -1.
  [[nodiscard]] Index first_column_outside(const std::vector<double>& lower,
                                           const std::vector<double>& upper) const;

 private:
  std::vector<double> x_;
};

/// Print "debug solution cut off: <what>" to stderr and to `logger`, then abort (#500).
[[noreturn]] void debug_solution_abort(Logger& logger, const std::string& what);

/// With `debug_solution` set: the point read against `model`, or an abort when the file
/// cannot be read (a checking run that silently checks nothing is worse than none). Nullopt
/// when the option is empty.
[[nodiscard]] std::optional<DebugSolution> load_debug_solution(const Options& options,
                                                               const Model& model,
                                                               Logger& logger);

/// Presolve's side of the check (#500), called by the pipeline on a MILP with
/// `debug_solution` set: the point must be feasible for `original`; every column a record
/// fixes must be fixed at the point's value; presolve must not have proved the model
/// infeasible; and the reduced model must contain the point. Aborts on the first failure.
void check_presolve_against_debug_solution(const Model& original,
                                           const presolve::Result& reduced,
                                           const Options& options, Logger& logger);

/// solve_branch_and_bound's side (#500): the model it received must contain the point, and so
/// must the model after integral row rounding when that moved a row (`tightened`, or null).
void check_search_input_against_debug_solution(const Model& received, const Model* tightened,
                                               const DebugSolution& point, Logger& logger);

/// And the answer: a search that reports the model infeasible, or reports an optimum worse
/// than the point by more than the gap targets allow, lost it somewhere unchecked.
void check_search_result_against_debug_solution(const Model& model, const Solution& result,
                                                const Options& options,
                                                const DebugSolution& point, Logger& logger);

/// A copy of `model` with every unnamed column named `C<index>` (#500): presolve renumbers
/// columns, and the check reads the point by name, so an unnamed column must be named before
/// presolve for the reduced model to carry the same names.
[[nodiscard]] Model with_column_names(const Model& model);

namespace testing {
/// TEST SEAM (#500), and the only way to reach it: a cut appended to the root round's
/// accepted cuts when, and only when, a debug solution is active, so a test can prove that a
/// cut which removes the known optimum is caught inside a real search. Pass nullopt to clear.
void set_planted_cut(std::optional<Cut> cut);
}  // namespace testing

/// Append the planted cut, if a test set one (see testing::set_planted_cut).
void append_planted_cut(std::vector<Cut>* accepted);

}  // namespace mip
}  // namespace sankhya
