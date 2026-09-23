// SPDX-License-Identifier: Apache-2.0
// SANKHYA - what the machine can hold, for budgets that were laptop constants (#576).
//
// The interior point refuses an ordering whose quotient graph, or a factor whose nonzero
// count, would exceed a budget (#246). Both budgets were fixed at 1e8 entries - 400 MB and
// 1.2 GB - sized for the 7.7 GB laptop the project is developed on. On a 96 GB node the same
// constants abandoned three of the eight Mittelmann orderings in 11 to 22 s with 86 GB free
// (bench/results/mittelmann-ipm-5c7efbc.csv). A budget whose purpose is "do not run this
// machine out of memory" has to be a function of this machine.
#pragma once

#include <cstddef>

namespace sankhya {

/// Physical memory of the machine in bytes, or 0 when the platform does not say.
[[nodiscard]] std::size_t physical_memory_bytes() noexcept;

/// The ordering budget (quotient-graph list entries, 4 bytes each) for a machine with
/// `physical_bytes` of memory: a sixteenth of the memory, never below the laptop constant
/// of 1e8 entries, and the laptop constant when the memory is unknown.
[[nodiscard]] std::size_t auto_ordering_budget(std::size_t physical_bytes) noexcept;

/// The factor budget (nonzeros, 12 bytes each: an 8-byte value and a 4-byte index) for the
/// same machine: an eighth of the memory, never below the laptop constant of 1e8 nonzeros,
/// and the laptop constant when the memory is unknown.
[[nodiscard]] std::size_t auto_factor_budget(std::size_t physical_bytes) noexcept;

}  // namespace sankhya
