// SPDX-License-Identifier: Apache-2.0
// SANKHYA - write the scaled CSR the first-order engine runs on, for #982 step 1.
//
//     sankhya-dump-scaled-csr <model.mps[.gz]|model.lp> <out.csr>
//
// bench/tools/gpu_precision_microbench.cu times A x and A^T y on one matrix in double and in
// single precision. The matrix it should time is the one PDHG multiplies by, not the raw
// model: src/gpu/pdhg_gpu.cu builds it as build_scaling(model, cost, 10) (ten Ruiz rounds and
// one Pock-Chambolle alpha = 1 pass, src/la/scaling.cpp) and uploads CsrView(scaling.matrix).
// This tool does exactly that on the model as read (no presolve) and writes the result in the
// micro-benchmark's flat format:
//
//     int64 m, int64 n, int64 nnz, int32 row_ptr[m+1], int32 col_idx[nnz], double values[nnz]
//
// The scaled matrix does not depend on the cost vector (build_scaling applies the column
// multipliers to the cost only after the matrix rounds), so the cost passed here is the
// model's own in minimise space, as the engine passes it.
//
// It prints one line: rows, columns, nonzeros and the scaled matrix's largest and smallest
// absolute entry, so a dump of the wrong model is visible in the runner's log.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "la/scaling.hpp"
#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/sparse.hpp"

namespace {

// The engine's Ruiz round count (kRuizIter in src/gpu/pdhg_gpu.cu, kRuizIterations in
// src/pdhg/pdhg.cpp). Kept equal by hand: a different count would time a different matrix.
constexpr int kRuizRounds = 10;

template <typename T>
void put(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void put_all(std::ofstream& out, const std::vector<T>& values) {
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
  using namespace sankhya;
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <model> <out.csr>\n", argv[0]);
    return 2;
  }
  Model model;
  const io::ReadResult loaded = io::read_model(argv[1], &model);
  if (!loaded.ok) {
    std::fprintf(stderr, "read error: %s\n", loaded.error.c_str());
    return 1;
  }
  const double sense = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  std::vector<double> cost(model.col_cost.size());
  for (std::size_t j = 0; j < cost.size(); ++j) cost[j] = sense * model.col_cost[j];

  const Scaling scaling = build_scaling(model, cost, kRuizRounds);
  const CsrView csr(scaling.matrix);
  const std::int64_t m = csr.num_rows();
  const std::int64_t n = csr.num_cols();
  const std::int64_t nnz = static_cast<std::int64_t>(csr.values().size());
  static_assert(sizeof(Index) == sizeof(std::int32_t), "the dump format stores int32 indices");

  std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "cannot open %s for writing\n", argv[2]);
    return 1;
  }
  put(out, m);
  put(out, n);
  put(out, nnz);
  put_all(out, csr.row_starts());
  put_all(out, csr.column_indices());
  put_all(out, csr.values());
  out.close();
  if (!out) {
    std::fprintf(stderr, "write to %s failed\n", argv[2]);
    return 1;
  }
  std::printf("rows=%lld cols=%lld nnz=%lld max_abs=%.6g min_abs=%.6g\n",
              static_cast<long long>(m), static_cast<long long>(n),
              static_cast<long long>(nnz), scaling.max_abs, scaling.min_abs);
  return 0;
}
