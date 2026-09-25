// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the batched PDHG's CUDA backend (#520): K LPs that share the matrix, iterated as
// one n x K and one m x K block, two kernels per iteration.
//
// The method, its references and the reason its bounds are safe are in
// src/pdhg/batch_pdhg.hpp; the split between the decisions (host, shared) and the arithmetic
// (this backend or the CPU reference) in src/pdhg/batch_pdhg_backend.hpp. The kernels do the
// CPU reference's arithmetic in the same order with explicitly rounded intrinsics, so the
// two return the same duals bit for bit.
//
// Compiled only when SANKHYA_ENABLE_CUDA is ON.
#pragma once

#include <memory>
#include <string>

#include "../pdhg/batch_pdhg_backend.hpp"

namespace sankhya::gpu {

/// The device backend for `data`, with the matrix, boxes and starting points uploaded; null,
/// with the reason in *why, when no device answers or an allocation or copy fails.
[[nodiscard]] std::unique_ptr<pdhg::BatchBackend> make_device_batch_backend(
    const pdhg::BatchData& data, std::string* why);

}  // namespace sankhya::gpu
