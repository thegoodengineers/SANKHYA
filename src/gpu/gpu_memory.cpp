// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU VRAM estimation for the PDHG backend (#281).
//
// The numbers here mirror GpuState in pdhg_gpu.cu. When a field is added or removed from
// GpuState the corresponding line below must be updated to match.

#include "gpu_memory.hpp"

#include <algorithm>
#include <cstddef>

namespace sankhya::gpu {

bool is_supported_compute_capability(int major, int minor) noexcept {
  return major * 10 + minor >= kMinComputeArch;
}

std::size_t estimate_pdhg_gpu_memory(Index rows, Index cols, Count nonzeros) {
  const auto m = static_cast<std::size_t>(rows < 0 ? 0 : rows);
  const auto n = static_cast<std::size_t>(cols < 0 ? 0 : cols);
  const auto nnz = static_cast<std::size_t>(nonzeros < 0 ? 0 : nonzeros);

  // GpuState n-length double vectors (9, after #382 removed d_tmpn):
  //   d_x, d_xn, d_ext, d_dx, d_aty, d_xsum  (primal iterates + sums)
  //   d_cost, d_clo, d_chi                     (objective and column bounds)
  const std::size_t n_vecs = 9 * n * sizeof(double);

  // GpuState m-length double vectors (8, after #382 removed d_tmpm):
  //   d_y, d_yn, d_dy, d_ax, d_adx, d_ysum   (dual iterates + sums)
  //   d_rlo, d_rhi                             (row bounds)
  const std::size_t m_vecs = 8 * m * sizeof(double);

  // d_scalars[3]: mv_x, mv_y, interaction. The block partials they are summed from (#478,
  // ceil(n/256) + 2 ceil(m/256) doubles, about 1/256 of one vector each) sit inside the
  // library-overhead constant below.
  const std::size_t scalars = 3 * sizeof(double);

  // CSR matrix in device memory
  const std::size_t rowptr = (m + 1) * sizeof(int);
  const std::size_t colidx = nnz * sizeof(int);
  const std::size_t vals = nnz * sizeof(double);

  // Conservative constant for the cuSPARSE SpMV workspace. In practice a few hundred KiB;
  // 16 MiB gives headroom for larger models and allocation fragmentation.
  constexpr std::size_t kLibraryOverhead = 16ULL * 1024 * 1024;  // 16 MiB

  return n_vecs + m_vecs + scalars + rowptr + colidx + vals + kLibraryOverhead;
}

std::size_t estimate_pdhg_gpu_transpose_memory(Index cols, Count nonzeros) {
  const auto n = static_cast<std::size_t>(cols < 0 ? 0 : cols);
  const auto nnz = static_cast<std::size_t>(nonzeros < 0 ? 0 : nonzeros);
  return (n + 1) * sizeof(int) + nnz * sizeof(int) + nnz * sizeof(double);
}

std::size_t vram_reserve(std::size_t total_bytes) {
  // Reserve the larger of 10 % of total VRAM or 256 MiB so that the runtime, other
  // processes, and allocation fragmentation have headroom. On a 6144 MiB card (the RTX 4050
  // Laptop GPU the team has) that is 614 MiB. The policy is a choice, not a measurement: no
  // CUDA run has been recorded in bench/results/ yet (#19 is that measurement).
  constexpr std::size_t kMinReserve = 256ULL * 1024 * 1024;  // 256 MiB
  return std::max(kMinReserve, total_bytes / 10);
}

}  // namespace sankhya::gpu
