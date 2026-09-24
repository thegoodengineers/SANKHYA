// SPDX-License-Identifier: Apache-2.0
// SANKHYA - activity-based bound propagation in synchronous rounds (#510).
//
// THE ALGORITHM. Each round reads every row against the SAME column bounds (the bounds the
// round started with): per row the minimum and maximum activity over the box, and from them,
// per nonzero, the bound the row implies for that column,
//
//     a_j > 0:  x_j <= (u_i - (minact_i - a_j l_j)) / a_j,   x_j >= (l_i - (maxact_i - a_j
//     u_j)) / a_j
//
// (the mirror images for a_j < 0). A column takes the tightest bound any row implies, an
// integer column's is rounded inward, and the round ends; rounds repeat until nothing moves by
// more than kPropagationMinChange or the round limit is reached. A row whose finite activity
// already violates its side, or a column whose box crosses, proves the box empty.
// Savelsbergh, "Preprocessing and probing techniques for mixed integer programming problems",
// ORSA J. Computing 6 (1994); the synchronous (Jacobi) form is the one that parallelises,
// Sofranac, Gleixner & Pokutta, "Accelerated domain propagation for mixed-integer linear
// programs on GPUs", arXiv:2009.07785 (2020), Algorithm 1.
//
// WHY THIS FILE AND NOT THE NODE PROPAGATOR. BranchAndBound::propagate_row() updates a bound
// the moment a row implies it (Gauss-Seidel): cheaper per sweep on one core, but its result
// depends on the row order, and a GPU cannot reproduce it. This is the reference the CUDA
// propagator (src/gpu/domain_prop.cu) must match BIT FOR BIT: the same sums in the same order
// (row by row, in CSR order), the same divisions, and reductions (min, max) whose result does
// not depend on the order they are taken in.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/types.hpp"

namespace sankhya::mip {

/// A row-major copy of a model's matrix, the layout both propagators read.
struct RowMajor {
  Index rows = 0;
  Index cols = 0;
  std::vector<Index> start;  ///< rows + 1 offsets
  std::vector<Index> column;
  std::vector<double> value;
};

[[nodiscard]] RowMajor row_major(const Model& model);

struct JacobiPropagation {
  bool infeasible = false;  ///< a row or a column proved the box empty
  int rounds = 0;           ///< rounds run, including the last, quiet one
  Count tightened = 0;      ///< bound changes applied over all rounds
};

/// Tighten [lower, upper] in place over `rounds_limit` synchronous rounds at most.
/// `integrality` is the tolerance an integer column's bound is rounded with.
JacobiPropagation propagate_jacobi(const Model& model, const RowMajor& rows,
                                   std::vector<double>* lower, std::vector<double>* upper,
                                   int rounds_limit, double integrality);

/// Root propagation for the search (gpu_domain_prop): tighten `model`'s column bounds in
/// place, on the device when the build has CUDA and a card answers, otherwise with the CPU
/// reference above - the same bounds either way. A box proved empty is left as it was (the
/// search proves it again, with its own evidence). Returns the number of bounds changed.
Count propagate_root_bounds(Model* model, const Options& options, Logger& logger);

}  // namespace sankhya::mip
