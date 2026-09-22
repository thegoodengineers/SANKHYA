#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Is the binary about to be demonstrated the one this tree builds?
#
# Sourced by the demos and by scripts/preflight.sh for the functions below; run directly it
# answers the same question for one binary, which is the form to reach for before a
# demonstration:  scripts/binary_provenance.sh build/sankhya.exe
#
# WHY THIS EXISTS. demo/run_sih_demo.sh, demo/run_demo.sh and scripts/preflight.sh all take
# the first binary they find, and `build/` is first on that list. A `build/` left from an
# earlier commit runs perfectly well and answers with THAT commit's solver, so the demo
# reports whatever was true then and says nothing about the gap. Measured here on
# 2026-09-22: a `build/` linked ten days earlier failed six of the nine committed Netlib
# instances with "verifier: REJECTED", because the postsolve basis fix (#345) landed after
# it. The same run against a binary built from the tree passed nine of nine. Nothing on the
# screen named the binary as the cause, and the only honest reading of it was that the
# solver was broken - in front of whoever is watching.
#
# The commit is compiled in (SANKHYA_GIT_COMMIT, CMakeLists.txt) and printed by `version`,
# so one invocation settles it. Everything uncertain stays SILENT: no git, not a checkout, a
# version string this cannot parse, a build from an export with no commit at all. This
# reports a fact it has established and nothing else, because a warning that fires on a
# machine where it cannot be true is a warning people learn to scroll past.

# The short commit compiled into a binary, or nothing.
sankhya_binary_commit() {
  "$1" version 2>/dev/null | sed -n 's/^SANKHYA [^(]*(\([0-9a-fA-F]\{7,\}\).*/\1/p' | head -1
}

# The short commit of the checkout this script is running in, or nothing.
sankhya_repo_commit() {
  git rev-parse --short HEAD 2>/dev/null
}

# Echoes "<binary commit> <HEAD commit>" when the two are known AND differ; echoes nothing
# otherwise, including when they agree. The caller decides how loudly to say it: preflight
# degrades, the demos print a block above the first result.
sankhya_binary_staleness() {
  local binary_commit head_commit
  binary_commit="$(sankhya_binary_commit "$1")"
  head_commit="$(sankhya_repo_commit)"
  [ -n "$binary_commit" ] || return 0
  [ -n "$head_commit" ] || return 0
  [ "$binary_commit" = "$head_commit" ] && return 0
  printf '%s %s\n' "$binary_commit" "$head_commit"
}

# Executed rather than sourced: report on the binary named, and exit non-zero when it is
# from another commit, so this can gate a script that wants to stop rather than warn.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  binary="${1:-build/sankhya.exe}"
  [ -x "$binary" ] || { printf 'not executable: %s\n' "$binary" >&2; exit 2; }
  staleness="$(sankhya_binary_staleness "$binary")"
  if [ -z "$staleness" ]; then
    printf '%s is built from this checkout (%s)\n' "$binary" "$(sankhya_repo_commit)"
    exit 0
  fi
  printf '%s was linked at %s; this tree is at %s\n' \
    "$binary" "${staleness% *}" "${staleness#* }" >&2
  exit 1
fi
