// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior point's normal equations factored on the device by NVIDIA cuDSS
// (#489), an opt-in backend beside the CPU sparse LDL^T of la/ldl.hpp.
//
// References:
//   Shin et al., the MadIPM GPU interior point, arXiv 2508.16094 (paper only): the Newton
//     iteration stays on the host and only the factorization and the triangular solves of
//     the KKT or normal-equations system move to the device, analysed once per pattern and
//     refactorized every iteration.
//   NVIDIA cuDSS documentation (docs.nvidia.com/cuda/cudss): the analysis / factorization /
//     solve phases, the symmetric matrix type, the pivot epsilon and the inertia query.
//
// WHAT IT DOES. The normal equations A Theta A^T + D keep one sparsity pattern for the whole
// solve and change their values every iteration, exactly as for the CPU factor. analyze()
// uploads the pattern and runs cuDSS's reordering and symbolic factorization once;
// factorize() uploads the values and runs the numeric LDL^T; solve() runs the forward,
// diagonal and backward solves for one right-hand side. The host keeps the iteration logic,
// the assembly of the matrix and the iterative refinement against it.
//
// THE PIVOT RULE IS THE CPU FACTOR'S. la/ldl.cpp replaces a pivot not above the dual
// regularization by the regularization. cuDSS is run as a symmetric LDL^T with pivoting off
// (the order is fixed by the analysis, as on the CPU) and a static pivot epsilon equal to
// that regularization, which replaces a pivot smaller in magnitude by it, and counts it. A
// NEGATIVE pivot larger than the epsilon is left alone by cuDSS where the CPU factor would
// lift it; the inertia query exposes it, and factorize() then reports kIndefinite so that the
// caller factors that iteration's matrix on the CPU instead. Measured on the V100 with cuDSS
// 0.8.0: a symmetric matrix with a -1e-6 pivot factors with inertia (2, 1) and no replaced
// pivot; a zero or 1e-14 pivot is replaced by 1e-8 and counted.
//
// WITHOUT cuDSS. The class compiles in every build. Without SANKHYA_ENABLE_CUDSS every
// method fails with a reason and the caller keeps the CPU factor; nothing in the CPU build
// includes a CUDA or cuDSS header.
//
// LICENCE. cuDSS is used under the NVIDIA Math Libraries SDK licence; the analysis, and the
// disclaimer and U.S. Government End Users notice carried by the installed cudss.h, are
// recorded in docs/PROVENANCE.md, judgement call 28. No cuDSS file is part of this
// repository.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "sankhya/sparse.hpp"
#include "sankhya/types.hpp"

namespace sankhya::gpu {

/// What one device factorization produced.
enum class CudssOutcome {
  kFactored,    ///< usable factors, every pivot positive after the epsilon rule
  kIndefinite,  ///< factors exist but hold a negative pivot the CPU rule would have lifted
  kFailed,      ///< a cuDSS or CUDA error; the reason says which
};

/// Seconds spent in each phase, measured on the host around synchronised device work.
struct CudssTiming {
  double analysis = 0.0;
  double factorization = 0.0;
  double solve = 0.0;
  Count factorizations = 0;
  Count solves = 0;
};

class CudssFactor {
 public:
  CudssFactor();
  ~CudssFactor();
  CudssFactor(const CudssFactor&) = delete;
  CudssFactor& operator=(const CudssFactor&) = delete;
  CudssFactor(CudssFactor&&) = delete;
  CudssFactor& operator=(CudssFactor&&) = delete;

  /// True when this build carries the backend (SANKHYA_ENABLE_CUDSS).
  [[nodiscard]] static bool compiled() noexcept;

  /// Find a device and create the cuDSS handle. False, with the reason, when the build has
  /// no cuDSS, no device answers, or the library cannot be initialised.
  [[nodiscard]] bool initialize(std::string* reason);

  /// Upload the pattern of a symmetric matrix given by its LOWER triangle in CSC form (the
  /// layout of normal_equations_lower()) and run reordering and symbolic factorization.
  [[nodiscard]] bool analyze(const SparseMatrix& lower, std::string* reason);

  /// Numeric factorization of a matrix with the analysed pattern or a subset of it (the
  /// missing entries are zeros), as SparseLdl::factorize() accepts; an entry outside it fails.
  [[nodiscard]] CudssOutcome factorize(const SparseMatrix& lower, double regularization,
                                       std::string* reason);

  /// Solve with the current factors in place. On failure `b` is left as it was.
  [[nodiscard]] bool solve(double* b, std::string* reason);

  /// Pivots the last factorization replaced by the epsilon.
  [[nodiscard]] Count regularized_pivots() const noexcept;
  /// Nonzeros of the factor as cuDSS reports them after the analysis (-1 before).
  [[nodiscard]] std::int64_t factor_nonzeros() const noexcept;
  [[nodiscard]] const CudssTiming& timing() const noexcept;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace sankhya::gpu
