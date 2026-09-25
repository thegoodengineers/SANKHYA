// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU restarted PDHG for LP (#295): public declaration.
//
// Compiled only when SANKHYA_ENABLE_CUDA is ON.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::gpu {

/// Multi-GPU row-partitioned restarted PDHG.
///
/// Distributes the constraint matrix rows across `device_ids` in contiguous blocks balanced
/// by nonzeros (partition_rows_by_nonzeros). The primal iterate x is replicated on every
/// device; the dual iterate y is partitioned. A^T y is summed across the devices by an
/// all-gather of the per-device partials, device to device over P2P when every pair allows
/// it and the option gpu_peer_access is true, else staged through pinned host memory, and
/// every reduction has a fixed order: the result is bitwise reproducible run to run, and
/// the same for either transport.
///
/// Falls back to solve_pdhg_gpu() when:
///   - device_ids has one entry and gpu_partitioned is false, OR
///   - a device is absent, more than multi::kMaxDevices are listed, or any device fails to
///     allocate or initialize.
[[nodiscard]] Solution solve_pdhg_multi_gpu(const Model& model, const Options& options,
                                            const std::vector<int>& device_ids, Logger& logger,
                                            SolveControl* control = nullptr);

}  // namespace sankhya::gpu
