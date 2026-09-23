// SPDX-License-Identifier: Apache-2.0
// GPU PDHG on-device iteration loop via CUDA Graphs (#478).
//
// References:
//   Lu & Yang, cuPDLP.jl, arXiv:2311.12180 (host/device split, graph capture)
//   NVIDIA, CUDA C++ Programming Guide, chapter "CUDA Graphs"
#pragma once

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::gpu {

/// Run K iterations of the PDHG inner loop entirely on the device via a CUDA
/// Graph captured once and replayed, eliminating the per-iteration host
/// round-trip. Falls back to the standard per-iteration path when
/// gpu_on_device_loop=false or no CUDA device is available.
///
/// Returns the number of iterations actually performed (for accounting).
/// The caller is responsible for host-side convergence checks between blocks.
int pdhg_graph_block(int K, const Options& options);

}  // namespace sankhya::gpu
