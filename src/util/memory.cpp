// SPDX-License-Identifier: Apache-2.0
// SANKHYA - physical memory probe and the budgets derived from it (#576). See memory.hpp.
#include "util/memory.hpp"

#include <algorithm>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sankhya {

namespace {

/// The budgets the 7.7 GB laptop shipped with (#246): the floor, so a machine with less
/// memory than that, or one that does not report it, gets exactly the behaviour that was
/// measured there.
constexpr std::size_t kLaptopOrderingEntries = 100000000;  // 4 bytes each: 400 MB
constexpr std::size_t kLaptopFactorNonzeros = 100000000;   // 12 bytes each: 1.2 GB

constexpr std::size_t kOrderingEntryBytes = 4;
constexpr std::size_t kFactorNonzeroBytes = 12;
constexpr std::size_t kOrderingShare = 16;  // a sixteenth of physical memory
constexpr std::size_t kFactorShare = 8;     // an eighth

}  // namespace

std::size_t physical_memory_bytes() noexcept {
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status) == 0) return 0;
  return static_cast<std::size_t>(status.ullTotalPhys);
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page = sysconf(_SC_PAGE_SIZE);
  if (pages <= 0 || page <= 0) return 0;
  return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page);
#endif
}

std::size_t auto_ordering_budget(std::size_t physical_bytes) noexcept {
  if (physical_bytes == 0) return kLaptopOrderingEntries;
  return std::max(kLaptopOrderingEntries,
                  physical_bytes / kOrderingShare / kOrderingEntryBytes);
}

std::size_t auto_factor_budget(std::size_t physical_bytes) noexcept {
  if (physical_bytes == 0) return kLaptopFactorNonzeros;
  return std::max(kLaptopFactorNonzeros, physical_bytes / kFactorShare / kFactorNonzeroBytes);
}

}  // namespace sankhya
