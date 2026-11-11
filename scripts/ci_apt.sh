#!/usr/bin/env bash
# Install apt packages on a CI runner, and survive the package mirror being unreachable.
#
#   scripts/ci_apt.sh ninja-build zlib1g-dev
#
# On 11 Nov 2026 the hosted runners could not resolve azure.archive.ubuntu.com for over an
# hour and every job died here before building anything. So: three attempts at apt, and if
# the mirror is still down, accept each package that the runner image already carries
# (dpkg -s) and take ninja from PyPI, which is a different host. A package that is neither
# installed nor replaceable still fails the step, so nothing is silently skipped.
set -u

for attempt in 1 2 3; do
  if sudo apt-get update && sudo apt-get install -y "$@"; then
    exit 0
  fi
  echo "ci_apt: attempt $attempt failed" >&2
  [ "$attempt" = 3 ] || sleep 20
done

echo "ci_apt: apt is unreachable; falling back to what the image has" >&2
missing=0
for pkg in "$@"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "ci_apt: $pkg is already installed"
  elif [ "$pkg" = ninja-build ]; then
    if command -v ninja >/dev/null 2>&1 || pip install ninja; then
      echo "ci_apt: ninja $(ninja --version) from $(command -v ninja)"
    else
      missing=1
    fi
  else
    echo "ci_apt: $pkg is not installed and has no fallback" >&2
    missing=1
  fi
done
exit "$missing"
