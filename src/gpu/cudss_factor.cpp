// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cuDSS factorization of the interior point's normal equations (#489). The design
// and the references are in cudss_factor.hpp.
//
// cuDSS is NVIDIA's; the header included below carries NVIDIA's disclaimer and U.S.
// Government End Users notice, which apply to this use of it (docs/PROVENANCE.md, judgement
// call 28). Only its documented C API is called.

#include "gpu/cudss_factor.hpp"

#include <vector>

#include "sankhya/timer.hpp"

#ifdef SANKHYA_ENABLE_CUDSS
#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>

#include <fmt/format.h>
#endif

namespace sankhya::gpu {

#ifdef SANKHYA_ENABLE_CUDSS

namespace {

const char* cudss_status_name(cudssStatus_t status) {
  switch (status) {
    case CUDSS_STATUS_SUCCESS: return "success";
    case CUDSS_STATUS_NOT_INITIALIZED: return "not initialized";
    case CUDSS_STATUS_ALLOC_FAILED: return "allocation failed";
    case CUDSS_STATUS_INVALID_VALUE: return "invalid value";
    case CUDSS_STATUS_NOT_SUPPORTED: return "not supported";
    case CUDSS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUDSS_STATUS_INTERNAL_ERROR: return "internal error";
    default: return "unknown status";
  }
}

bool cudss_ok(cudssStatus_t status, const char* what, std::string* reason) {
  if (status == CUDSS_STATUS_SUCCESS) return true;
  if (reason != nullptr) *reason = fmt::format("cuDSS {}: {}", what, cudss_status_name(status));
  return false;
}

bool cuda_ok(cudaError_t error, const char* what, std::string* reason) {
  if (error == cudaSuccess) return true;
  if (reason != nullptr) *reason = fmt::format("CUDA {}: {}", what, cudaGetErrorString(error));
  return false;
}

/// A device allocation freed on destruction.
template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  ~DeviceArray() { release(); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  bool allocate(std::size_t count, std::string* reason) {
    release();
    if (count == 0) count = 1;  // a zero-size cudaMalloc returns null, which cuDSS refuses
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc",
                 reason)) {
      data_ = nullptr;
      return false;
    }
    return true;
  }
  bool upload(const T* host, std::size_t count, std::string* reason) {
    return cuda_ok(cudaMemcpy(data_, host, count * sizeof(T), cudaMemcpyHostToDevice), "upload",
                   reason);
  }
  bool download(T* host, std::size_t count, std::string* reason) const {
    return cuda_ok(cudaMemcpy(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                   "download", reason);
  }
  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  void release() noexcept {
    if (data_ != nullptr) (void)cudaFree(data_);
    data_ = nullptr;
  }
  T* data_ = nullptr;
};

}  // namespace

struct CudssFactor::State {
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t matrix = nullptr;
  cudssMatrix_t solution = nullptr;
  cudssMatrix_t rhs = nullptr;

  DeviceArray<Index> starts;
  DeviceArray<Index> indices;
  DeviceArray<double> values;
  DeviceArray<double> x;
  DeviceArray<double> b;

  /// The analysed pattern, compared with every factorize() argument.
  std::vector<Index> pattern_starts;
  std::vector<Index> pattern_rows;
  /// The values of a matrix whose pattern is a strict subset, scattered into the analysed one.
  std::vector<double> scattered;
  Index dimension = 0;
  Index nonzeros = 0;
  bool analyzed = false;
  bool factored = false;
  Count regularized = 0;
  std::int64_t factor_nonzeros = -1;
  std::vector<double> saved;
  CudssTiming timing;

  void destroy_matrices() noexcept {
    if (matrix != nullptr) (void)cudssMatrixDestroy(matrix);
    if (solution != nullptr) (void)cudssMatrixDestroy(solution);
    if (rhs != nullptr) (void)cudssMatrixDestroy(rhs);
    matrix = solution = rhs = nullptr;
  }
  ~State() {
    destroy_matrices();
    if (data != nullptr) (void)cudssDataDestroy(handle, data);
    if (config != nullptr) (void)cudssConfigDestroy(config);
    if (handle != nullptr) (void)cudssDestroy(handle);
  }
};

CudssFactor::CudssFactor() : state_(std::make_unique<State>()) {}
CudssFactor::~CudssFactor() = default;

bool CudssFactor::compiled() noexcept {
  return true;
}

bool CudssFactor::initialize(std::string* reason) {
  State& s = *state_;
  if (s.handle != nullptr) return true;
  int devices = 0;
  if (!cuda_ok(cudaGetDeviceCount(&devices), "device count", reason)) return false;
  if (devices == 0) {
    if (reason != nullptr) *reason = "no CUDA device found";
    return false;
  }
  if (!cudss_ok(cudssCreate(&s.handle), "create", reason)) {
    s.handle = nullptr;
    return false;
  }
  if (!cudss_ok(cudssConfigCreate(&s.config), "config", reason)) {
    s.config = nullptr;
    return false;
  }
  if (!cudss_ok(cudssDataCreate(s.handle, &s.data), "data", reason)) {
    s.data = nullptr;
    return false;
  }
  // No numerical pivoting: the order is the analysis's, as on the CPU, and a small pivot is
  // replaced by the static epsilon (set per factorization) rather than swapped away.
  const cudssPivotType_t pivot = CUDSS_PIVOT_NONE;
  const cudssPivotEpsilonAlg_t epsilon_rule = CUDSS_PIVOT_EPSILON_ALG_STATIC;
  // Bit-wise identical factors on every run. Without it the V100 took stocfor1 in 89
  // iterations on one run and 91 on the next, and pilot4 in 66 and 68: the interior point's
  // end game amplifies a last-bit difference, and a solve that cannot be repeated cannot be
  // checked.
  const int deterministic = 1;
  return cudss_ok(cudssConfigSet(s.config, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic,
                                 sizeof(deterministic)),
                  "deterministic mode", reason) &&
         cudss_ok(cudssConfigSet(s.config, CUDSS_CONFIG_PIVOT_TYPE, &pivot, sizeof(pivot)),
                  "pivot type", reason) &&
         cudss_ok(cudssConfigSet(s.config, CUDSS_CONFIG_PIVOT_EPSILON_ALG, &epsilon_rule,
                                 sizeof(epsilon_rule)),
                  "pivot epsilon rule", reason);
}

bool CudssFactor::analyze(const SparseMatrix& lower, std::string* reason) {
  State& s = *state_;
  s.analyzed = false;
  s.factored = false;
  if (s.handle == nullptr && !initialize(reason)) return false;
  if (lower.num_rows() != lower.num_cols() || lower.num_rows() == 0) {
    if (reason != nullptr) *reason = "the matrix is empty or not square";
    return false;
  }
  s.destroy_matrices();
  s.dimension = lower.num_rows();
  s.nonzeros = lower.num_nonzeros();
  s.pattern_starts = lower.column_starts();
  s.pattern_rows = lower.row_indices();
  const auto n = static_cast<std::size_t>(s.dimension);
  const auto nnz = static_cast<std::size_t>(s.nonzeros);
  if (!s.starts.allocate(n + 1, reason) || !s.indices.allocate(nnz, reason) ||
      !s.values.allocate(nnz, reason) || !s.x.allocate(n, reason) || !s.b.allocate(n, reason)) {
    return false;
  }
  if (!s.starts.upload(lower.column_starts().data(), n + 1, reason) ||
      !s.indices.upload(lower.row_indices().data(), nnz, reason) ||
      !s.values.upload(lower.values().data(), nnz, reason)) {
    return false;
  }
  // The lower triangle by column IS the upper triangle by row: column j lists the rows
  // i >= j, which read as row j lists the columns i >= j. So the CSC arrays are handed over
  // unchanged as a CSR matrix with the upper view.
  const auto rows = static_cast<std::int64_t>(s.dimension);
  if (!cudss_ok(cudssMatrixCreateCsr(&s.matrix, rows, rows, static_cast<std::int64_t>(nnz),
                                     s.starts.get(), nullptr, s.indices.get(), s.values.get(),
                                     CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                                     CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO),
                "matrix", reason) ||
      !cudss_ok(cudssMatrixCreateDn(&s.solution, rows, 1, rows, s.x.get(), CUDSS_R_64F,
                                    CUDSS_LAYOUT_COL_MAJOR),
                "solution", reason) ||
      !cudss_ok(cudssMatrixCreateDn(&s.rhs, rows, 1, rows, s.b.get(), CUDSS_R_64F,
                                    CUDSS_LAYOUT_COL_MAJOR),
                "right-hand side", reason)) {
    return false;
  }
  const Timer clock;
  if (!cudss_ok(cudssExecute(s.handle, CUDSS_PHASE_ANALYSIS, s.config, s.data, s.matrix,
                             s.solution, s.rhs),
                "analysis", reason) ||
      !cuda_ok(cudaDeviceSynchronize(), "analysis", reason)) {
    return false;
  }
  s.timing.analysis += clock.elapsed_seconds();
  std::int64_t lu_nonzeros = -1;
  std::size_t written = 0;
  if (cudssDataGet(s.handle, s.data, CUDSS_DATA_LU_NNZ, &lu_nonzeros, sizeof(lu_nonzeros),
                   &written) == CUDSS_STATUS_SUCCESS) {
    s.factor_nonzeros = lu_nonzeros;
  }
  s.analyzed = true;
  return true;
}

CudssOutcome CudssFactor::factorize(const SparseMatrix& lower, double regularization,
                                    std::string* reason) {
  State& s = *state_;
  s.factored = false;
  s.regularized = 0;
  if (!s.analyzed) {
    if (reason != nullptr) *reason = "factorize() before analyze()";
    return CudssOutcome::kFailed;
  }
  // THE SAME PATTERN OR A SUBSET OF IT, as la/ldl.hpp accepts: normal_equations_lower()
  // leaves out a product whose Theta is exactly zero, so an iteration's matrix can hold fewer
  // entries than the one analysed (seen on ganges and greenbea). Such a matrix is scattered
  // into the analysed pattern with explicit zeros; an entry outside it is refused.
  if (lower.num_rows() != s.dimension || lower.num_cols() != s.dimension) {
    if (reason != nullptr) *reason = "the matrix does not have the analysed dimension";
    return CudssOutcome::kFailed;
  }
  const double* values = lower.values().data();
  if (lower.column_starts() != s.pattern_starts || lower.row_indices() != s.pattern_rows) {
    s.scattered.assign(static_cast<std::size_t>(s.nonzeros), 0.0);
    const std::vector<Index>& starts = lower.column_starts();
    const std::vector<Index>& rows = lower.row_indices();
    for (std::size_t j = 0; j < static_cast<std::size_t>(s.dimension); ++j) {
      auto slot = static_cast<std::size_t>(s.pattern_starts[j]);
      const auto slot_end = static_cast<std::size_t>(s.pattern_starts[j + 1]);
      const auto end = static_cast<std::size_t>(starts[j + 1]);
      for (auto p = static_cast<std::size_t>(starts[j]); p < end; ++p) {
        while (slot < slot_end && s.pattern_rows[slot] < rows[p]) ++slot;
        if (slot == slot_end || s.pattern_rows[slot] != rows[p]) {
          if (reason != nullptr) {
            *reason = fmt::format("entry ({}, {}) is outside the analysed pattern", rows[p], j);
          }
          return CudssOutcome::kFailed;
        }
        s.scattered[slot] = values[p];
      }
    }
    values = s.scattered.data();
  }
  const Timer clock;
  if (!s.values.upload(values, static_cast<std::size_t>(s.nonzeros), reason) ||
      !cudss_ok(cudssConfigSet(s.config, CUDSS_CONFIG_PIVOT_EPSILON, &regularization,
                               sizeof(regularization)),
                "pivot epsilon", reason) ||
      !cudss_ok(cudssExecute(s.handle, CUDSS_PHASE_FACTORIZATION, s.config, s.data, s.matrix,
                             s.solution, s.rhs),
                "factorization", reason) ||
      !cuda_ok(cudaDeviceSynchronize(), "factorization", reason)) {
    return CudssOutcome::kFailed;
  }
  ++s.timing.factorizations;
  int info = 0;
  std::size_t written = 0;
  if (!cudss_ok(cudssDataGet(s.handle, s.data, CUDSS_DATA_INFO, &info, sizeof(info), &written),
                "info", reason)) {
    return CudssOutcome::kFailed;
  }
  if (info != 0) {
    if (reason != nullptr) *reason = fmt::format("cuDSS factorization reported info {}", info);
    return CudssOutcome::kFailed;
  }
  int replaced = 0;
  if (!cudss_ok(cudssDataGet(s.handle, s.data, CUDSS_DATA_NPIVOTS, &replaced, sizeof(replaced),
                             &written),
                "pivot count", reason)) {
    return CudssOutcome::kFailed;
  }
  s.regularized = replaced;
  // (positive, negative) pivot counts of the factored matrix.
  int inertia[2] = {0, 0};
  if (!cudss_ok(cudssDataGet(s.handle, s.data, CUDSS_DATA_INERTIA, inertia, sizeof(inertia),
                             &written),
                "inertia", reason)) {
    return CudssOutcome::kFailed;
  }
  s.timing.factorization += clock.elapsed_seconds();
  if (inertia[1] != 0) {
    if (reason != nullptr) {
      *reason = fmt::format("{} negative pivot(s) of magnitude above {:.1e}", inertia[1],
                            regularization);
    }
    return CudssOutcome::kIndefinite;
  }
  s.factored = true;
  return CudssOutcome::kFactored;
}

bool CudssFactor::solve(double* b, std::string* reason) {
  State& s = *state_;
  if (!s.factored) {
    if (reason != nullptr) *reason = "solve() without factors";
    return false;
  }
  const auto n = static_cast<std::size_t>(s.dimension);
  const Timer clock;
  s.saved.assign(b, b + n);
  if (!s.b.upload(b, n, reason) ||
      !cudss_ok(cudssExecute(s.handle, CUDSS_PHASE_SOLVE, s.config, s.data, s.matrix,
                             s.solution, s.rhs),
                "solve", reason) ||
      !s.x.download(b, n, reason)) {
    std::copy(s.saved.begin(), s.saved.end(), b);
    return false;
  }
  ++s.timing.solves;
  s.timing.solve += clock.elapsed_seconds();
  return true;
}

Count CudssFactor::regularized_pivots() const noexcept {
  return state_->regularized;
}
std::int64_t CudssFactor::factor_nonzeros() const noexcept {
  return state_->factor_nonzeros;
}
const CudssTiming& CudssFactor::timing() const noexcept {
  return state_->timing;
}

#else  // !SANKHYA_ENABLE_CUDSS

namespace {
constexpr const char* kNotBuilt =
    "this build has no cuDSS backend (configure with -DSANKHYA_ENABLE_CUDA=ON "
    "-DSANKHYA_ENABLE_CUDSS=ON)";
}  // namespace

struct CudssFactor::State {
  CudssTiming timing;
};

CudssFactor::CudssFactor() : state_(std::make_unique<State>()) {}
CudssFactor::~CudssFactor() = default;

bool CudssFactor::compiled() noexcept {
  return false;
}

bool CudssFactor::initialize(std::string* reason) {
  if (reason != nullptr) *reason = kNotBuilt;
  return false;
}

bool CudssFactor::analyze(const SparseMatrix& /*lower*/, std::string* reason) {
  if (reason != nullptr) *reason = kNotBuilt;
  return false;
}

CudssOutcome CudssFactor::factorize(const SparseMatrix& /*lower*/, double /*regularization*/,
                                    std::string* reason) {
  if (reason != nullptr) *reason = kNotBuilt;
  return CudssOutcome::kFailed;
}

bool CudssFactor::solve(double* /*b*/, std::string* reason) {
  if (reason != nullptr) *reason = kNotBuilt;
  return false;
}

Count CudssFactor::regularized_pivots() const noexcept {
  return 0;
}
std::int64_t CudssFactor::factor_nonzeros() const noexcept {
  return -1;
}
const CudssTiming& CudssFactor::timing() const noexcept {
  return state_->timing;
}

#endif  // SANKHYA_ENABLE_CUDSS

}  // namespace sankhya::gpu
