// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the first-order QP engine's operator on the device (#493, option qp_gpu).
//
// The iteration logic stays in src/qp/qp_condat_vu.cpp; what moves to the card is the
// arithmetic of one step (src/qp/qp_operator.hpp): Q x and A^T y by cuSPARSE, the primal
// step with its projection and extrapolation, A xbar, the dual prox, the fixed-order
// reductions of the fixed-point residual and the restart distances, and the Halpern blend.
// The iterate, the anchor and the restart point live on the device between steps; only
// scalars cross to the host each iteration, and the evaluated point every 50.
//
// Compiled in every build: without SANKHYA_ENABLE_CUDA the factory returns nothing and says
// why (src/gpu/qp_device_stub.cpp), and the engine keeps the host operator.
#pragma once

#include <memory>
#include <string>

#include "../qp/qp_operator.hpp"
#include "sankhya/model.hpp"

namespace sankhya::gpu {

/// The device operator for `model`, or nullptr with `reason` set: no CUDA in the build, no
/// device, or a setup failure (memory, cuSPARSE). The model must be frozen and validated.
[[nodiscard]] std::unique_ptr<qp::QpOperator> make_qp_device_operator(const Model& model,
                                                                      std::string* reason);

}  // namespace sankhya::gpu
