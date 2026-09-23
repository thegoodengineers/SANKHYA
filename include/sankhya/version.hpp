// SPDX-License-Identifier: Apache-2.0
// SANKHYA - build identification.
//
// Every benchmark CSV records the git commit (ENGINEERING_RULES.md), so the commit has to be
// reachable from inside the binary rather than from whatever shell produced the CSV.
#pragma once

namespace sankhya {

/// Semantic version of the library, e.g. "0.1.0".
[[nodiscard]] const char* version_string() noexcept;

/// Short git commit the binary was built from, or "unknown" outside a git checkout.
[[nodiscard]] const char* git_commit() noexcept;

/// CMake build type: Release, Debug, RelWithDebInfo.
[[nodiscard]] const char* build_type() noexcept;

/// Compiler identification, e.g. "GNU 13.2.0".
[[nodiscard]] const char* compiler_string() noexcept;

/// True when the binary was compiled with the CUDA backend available. Note this says
/// nothing about whether a device is present at run time.
[[nodiscard]] bool cuda_enabled() noexcept;

/// Human-readable description of the first CUDA device visible at run time, e.g.
/// "GeForce RTX 4050 Laptop GPU (compute 8.9, 6144 MiB VRAM)".
/// Returns "none (CUDA not compiled in)" when SANKHYA_ENABLE_CUDA is off, or
/// "no device: <reason>" when a device cannot be found.
[[nodiscard]] const char* cuda_device_description() noexcept;

/// One-line banner: name, version, commit, build type, CUDA status.
[[nodiscard]] const char* banner() noexcept;

/// The canonical repository, "thegoodengineers/SANKHYA". The name SANKHYA is a common
/// Sanskrit word and other projects use it too, so every artefact a reader may meet out of
/// context (the CLI's version output, every solution file) names this one (#538).
[[nodiscard]] const char* repository() noexcept;

/// The canonical repository's URL, "https://github.com/thegoodengineers/SANKHYA".
[[nodiscard]] const char* repository_url() noexcept;

}  // namespace sankhya
