// SPDX-License-Identifier: Apache-2.0
// C++17-compatible wrapper around safe_dual_bound for use in .cu files (#520).
// safe_bound.hpp uses std::span (C++20); nvcc compiles .cu with C++17.
#pragma once

#include "sankhya/model.hpp"

namespace sankhya::gpu {

/// Neumaier-Shcherbina safe dual bound for one node. `y` points to `m` dual multipliers.
/// Returns -infinity when the bound cannot be established. C++17-compatible signature.
double batch_safe_bound(const Model& model, const double* y, int m);

}  // namespace sankhya::gpu
