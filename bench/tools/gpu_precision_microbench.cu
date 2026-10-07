// SPDX-License-Identifier: Apache-2.0
// SANKHYA - device product micro-benchmark for #982, step 1 ("measure before building").
//
// NOT YET RUN ON ANY HARDWARE. This file exists so the measurement the issue asks for can be
// taken on the laptop card without first writing any of the mixed-precision engine code; it
// does not depend on sankhya_core and reaches no sankhya:: namespace, so it can be reviewed
// and built in isolation. bench/runners/gpu_precision_microbench.py drives it and writes
// bench/results/gpu-precision-<card>-<commit>.csv in the column order #982 asks for.
//
// What it measures: A x and A^T y on one CSR matrix, timed with CUDA events, once built with
// double (CUDA_R_64F) values and once with the same pattern's values rounded to float
// (CUDA_R_32F) - the "same pattern" comparison #982 point 1 asks for, nothing else. It does
// not touch PDHG, scaling, or any solver state.
//
// Input format (read_csr_dump below): a flat binary file -
//   int64 m, int64 n, int64 nnz,
//   int32 row_ptr[m+1], int32 col_idx[nnz], double values[nnz]
// - written by bench/runners/gpu_precision_microbench.py from whatever matrix it is timing
//   (the generated 10,000 x 10,000 LP, brazil3, or a refinery ladder step), so this file does
//   not need an MPS reader of its own.
//
// Output: one CSV line per precision to stdout -
//   precision,warmup_runs,timed_runs,ax_seconds_median,atx_seconds_median,peak_bytes
// bench/runners/gpu_precision_microbench.py adds the model/instance columns and writes the
// full row #982 asks for.

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Csr {
  int64_t m = 0, n = 0, nnz = 0;
  std::vector<int32_t> row_ptr;
  std::vector<int32_t> col_idx;
  std::vector<double> values;
};

bool read_csr_dump(const std::string& path, Csr* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.read(reinterpret_cast<char*>(&out->m), sizeof(int64_t));
  f.read(reinterpret_cast<char*>(&out->n), sizeof(int64_t));
  f.read(reinterpret_cast<char*>(&out->nnz), sizeof(int64_t));
  if (!f || out->m < 0 || out->n < 0 || out->nnz < 0) return false;
  out->row_ptr.resize(static_cast<size_t>(out->m) + 1);
  out->col_idx.resize(static_cast<size_t>(out->nnz));
  out->values.resize(static_cast<size_t>(out->nnz));
  f.read(reinterpret_cast<char*>(out->row_ptr.data()),
         static_cast<std::streamsize>(out->row_ptr.size() * sizeof(int32_t)));
  f.read(reinterpret_cast<char*>(out->col_idx.data()),
         static_cast<std::streamsize>(out->col_idx.size() * sizeof(int32_t)));
  f.read(reinterpret_cast<char*>(out->values.data()),
         static_cast<std::streamsize>(out->values.size() * sizeof(double)));
  return static_cast<bool>(f) || f.eof();
}

#define CUDA_OK(expr)                                                       \
  do {                                                                      \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(_e));                                 \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

#define CS_OK(expr)                                                        \
  do {                                                                     \
    cusparseStatus_t _s = (expr);                                         \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__,     \
                   __LINE__, static_cast<int>(_s));                       \
      std::exit(1);                                                       \
    }                                                                      \
  } while (0)

/// Median of N timed repeats of A*x (non-transpose) and A^T*y (transpose), on a CSR matrix
/// whose values are already in `data_type` (CUDA_R_64F or CUDA_R_32F). `elem_bytes` is 8 or
/// 4; the caller passes pre-rounded values so no cast happens inside the timed region (#982
/// point 1 is about the product's cost at rest, not a cast it would also pay once per solver
/// iteration in the engine - that cast's cost is separately in src/gpu/pdhg_gpu.cu).
template <typename T>
void run_one_precision(const Csr& csr, cudaDataType_t data_type, int elem_bytes,
                       int warmup, int timed, double* ax_median_s, double* atx_median_s,
                       std::size_t* peak_bytes) {
  const int64_t m = csr.m, n = csr.n, nnz = csr.nnz;
  std::vector<T> vals_t(csr.values.size());
  for (size_t i = 0; i < csr.values.size(); ++i) vals_t[i] = static_cast<T>(csr.values[i]);

  int32_t *d_rowptr = nullptr, *d_colidx = nullptr;
  T *d_vals = nullptr, *d_x = nullptr, *d_y = nullptr, *d_yt = nullptr;
  CUDA_OK(cudaMalloc(&d_rowptr, csr.row_ptr.size() * sizeof(int32_t)));
  CUDA_OK(cudaMalloc(&d_colidx, csr.col_idx.size() * sizeof(int32_t)));
  CUDA_OK(cudaMalloc(&d_vals, vals_t.size() * sizeof(T)));
  CUDA_OK(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(T)));
  CUDA_OK(cudaMalloc(&d_y, static_cast<size_t>(m) * sizeof(T)));
  CUDA_OK(cudaMalloc(&d_yt, static_cast<size_t>(n) * sizeof(T)));
  CUDA_OK(cudaMemcpy(d_rowptr, csr.row_ptr.data(), csr.row_ptr.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_colidx, csr.col_idx.data(), csr.col_idx.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_vals, vals_t.data(), vals_t.size() * sizeof(T), cudaMemcpyHostToDevice));
  std::vector<T> ones_n(static_cast<size_t>(n), static_cast<T>(1));
  CUDA_OK(cudaMemcpy(d_x, ones_n.data(), ones_n.size() * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_y, 0, static_cast<size_t>(m) * sizeof(T)));
  CUDA_OK(cudaMemset(d_yt, 0, static_cast<size_t>(n) * sizeof(T)));

  cusparseHandle_t cs;
  CS_OK(cusparseCreate(&cs));
  cusparseSpMatDescr_t mat;
  CS_OK(cusparseCreateCsr(&mat, m, n, nnz, d_rowptr, d_colidx, d_vals, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, data_type));
  cusparseDnVecDescr_t vx, vy, vyt;
  CS_OK(cusparseCreateDnVec(&vx, n, d_x, data_type));
  CS_OK(cusparseCreateDnVec(&vy, m, d_y, data_type));
  CS_OK(cusparseCreateDnVec(&vyt, n, d_yt, data_type));

  const T one = static_cast<T>(1), zero = static_cast<T>(0);
  std::size_t buf_nt = 0, buf_t = 0;
  CS_OK(cusparseSpMV_bufferSize(cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vx, &zero, vy,
                                data_type, CUSPARSE_SPMV_ALG_DEFAULT, &buf_nt));
  CS_OK(cusparseSpMV_bufferSize(cs, CUSPARSE_OPERATION_TRANSPOSE, &one, mat, vy, &zero, vyt,
                                data_type, CUSPARSE_SPMV_ALG_DEFAULT, &buf_t));
  void* buf = nullptr;
  const std::size_t buf_bytes = std::max(buf_nt, buf_t);
  if (buf_bytes > 0) CUDA_OK(cudaMalloc(&buf, buf_bytes));

  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start));
  CUDA_OK(cudaEventCreate(&stop));

  auto timed_product = [&](cusparseOperation_t op, cusparseDnVecDescr_t in,
                           cusparseDnVecDescr_t out, int reps) {
    std::vector<double> samples(static_cast<size_t>(reps));
    for (int r = 0; r < warmup + reps; ++r) {
      const bool keep = r >= warmup;
      if (keep) CUDA_OK(cudaEventRecord(start));
      CS_OK(cusparseSpMV(cs, op, &one, mat, in, &zero, out, data_type,
                         CUSPARSE_SPMV_ALG_DEFAULT, buf));
      if (keep) {
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
        samples[static_cast<size_t>(r - warmup)] = static_cast<double>(ms) / 1000.0;
      } else {
        CUDA_OK(cudaDeviceSynchronize());
      }
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };

  *ax_median_s = timed_product(CUSPARSE_OPERATION_NON_TRANSPOSE, vx, vy, timed);
  *atx_median_s = timed_product(CUSPARSE_OPERATION_TRANSPOSE, vy, vyt, timed);
  *peak_bytes = csr.row_ptr.size() * sizeof(int32_t) + csr.col_idx.size() * sizeof(int32_t) +
                vals_t.size() * sizeof(T) +
                (static_cast<size_t>(n) * 2 + static_cast<size_t>(m)) * sizeof(T) + buf_bytes;

  cusparseDestroyDnVec(vx);
  cusparseDestroyDnVec(vy);
  cusparseDestroyDnVec(vyt);
  cusparseDestroySpMat(mat);
  cusparseDestroy(cs);
  cudaFree(d_rowptr);
  cudaFree(d_colidx);
  cudaFree(d_vals);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_yt);
  cudaFree(buf);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  (void)elem_bytes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <csr-dump-path> [warmup] [timed]\n", argv[0]);
    return 2;
  }
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 5;
  const int timed = argc > 3 ? std::atoi(argv[3]) : 21;  // odd, so the median is a real sample

  Csr csr;
  if (!read_csr_dump(argv[1], &csr)) {
    std::fprintf(stderr, "failed to read CSR dump: %s\n", argv[1]);
    return 1;
  }

  double ax_d = 0.0, atx_d = 0.0, ax_f = 0.0, atx_f = 0.0;
  std::size_t peak_d = 0, peak_f = 0;
  run_one_precision<double>(csr, CUDA_R_64F, 8, warmup, timed, &ax_d, &atx_d, &peak_d);
  run_one_precision<float>(csr, CUDA_R_32F, 4, warmup, timed, &ax_f, &atx_f, &peak_f);

  std::printf("precision,warmup_runs,timed_runs,ax_seconds_median,atx_seconds_median,"
             "peak_bytes\n");
  std::printf("double,%d,%d,%.9e,%.9e,%zu\n", warmup, timed, ax_d, atx_d, peak_d);
  std::printf("float,%d,%d,%.9e,%.9e,%zu\n", warmup, timed, ax_f, atx_f, peak_f);
  return 0;
}
