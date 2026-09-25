// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the simplex's report of how sparse its FTRAN and BTRAN results were (#464).
//
// Hall and McKinnon, "Hyper-sparsity in the revised simplex method and how to exploit it",
// Comput. Optim. Appl. 32 (2005), set the switch between the hyper-sparse and the full
// solves from the density of the results the simplex actually produces. SparseLu counts
// every solve's result density into five bins whichever path it took; this prints them at
// verbose level and hands them to the profiler, so tol::kHyperSparseDensity can be set from
// our own instances rather than from the paper's.

#include "simplex_core.hpp"

#include <array>
#include <cstdint>
#include <string>

#include <fmt/format.h>

#include "../util/profiler.hpp"

namespace sankhya::detail {

namespace {

constexpr std::array<const char*, 5> kBinNames = {"<1%", "1-5%", "5-10%", "10-30%", ">=30%"};

std::string describe(const std::array<std::int64_t, 5>& bins) {
  std::string text;
  for (std::size_t i = 0; i < bins.size(); ++i) {
    if (!text.empty()) text += ", ";
    text += fmt::format("{} {}", kBinNames[i], bins[i]);
  }
  return text;
}

}  // namespace

void Simplex::report_solve_densities() const {
  const LuSolveStats& stats = lu_.solve_stats();
  std::int64_t solves = 0;
  for (const std::int64_t count : stats.ftran_bins) solves += count;
  for (const std::int64_t count : stats.btran_bins) solves += count;
  if (solves == 0) return;
  logger_.verbose("LU result density (#464), FTRAN: {}; {} by symbolic reach",
                  describe(stats.ftran_bins), stats.ftran_hyper);
  logger_.verbose("LU result density (#464), BTRAN: {}; {} by symbolic reach",
                  describe(stats.btran_bins), stats.btran_hyper);
  if (Profiler* profiler = logger_.profiler();
      profiler != nullptr && profiler->records(ProfileMode::kDetailed)) {
    for (std::size_t i = 0; i < kBinNames.size(); ++i) {
      profiler->count(fmt::format("ftran density {}", kBinNames[i]), stats.ftran_bins[i]);
      profiler->count(fmt::format("btran density {}", kBinNames[i]), stats.btran_bins[i]);
    }
    profiler->count("ftran hyper-sparse", stats.ftran_hyper);
    profiler->count("btran hyper-sparse", stats.btran_hyper);
  }
}

}  // namespace sankhya::detail
