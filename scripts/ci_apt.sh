#!/usr/bin/env bash
# Install apt packages on a CI runner, and survive the package mirror being unreachable.
#
#   scripts/ci_apt.sh ninja-build zlib1g-dev
#
# On 11 Nov 2026 the hosted runners could not resolve azure.archive.ubuntu.com for over an
# hour and every job died here before building anything. So: three attempts at apt, the
# later ones against kernel.org's mirror, and if that is down too, accept each package that the runner image already carries
# (dpkg -s, or its command on PATH), take ninja from PyPI, which is a different host, and
# stand a pass-through in for ccache. A package that is neither present nor replaceable
# still fails the step, so nothing is silently skipped.
set -u

for attempt in 1 2 3; do
  if sudo apt-get update && sudo apt-get install -y "$@"; then
    exit 0
  fi
  echo "ci_apt: attempt $attempt failed" >&2
  # The second and third attempts go to kernel.org's mirror of the same archive, a different
  # host and DNS name. apt still checks every index against Ubuntu's signing keys.
  echo "http://mirrors.kernel.org/ubuntu/" | sudo tee /etc/apt/apt-mirrors.txt >/dev/null
  sudo sed -i 's|https\?://security.ubuntu.com/ubuntu/\?|http://mirrors.kernel.org/ubuntu/|g'     /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null
  [ "$attempt" = 3 ] || sleep 10
done

echo "ci_apt: apt is unreachable; falling back to what the image has" >&2
missing=0
for pkg in "$@"; do
  cmd=${pkg%-build} # ninja-build installs `ninja`; for the others the command is the package name
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "ci_apt: $pkg is already installed"
  elif command -v "$cmd" >/dev/null 2>&1; then
    echo "ci_apt: $cmd is on the image at $(command -v "$cmd")"
  elif [ "$pkg" = ninja-build ] && pip install ninja; then
    echo "ci_apt: ninja $(ninja --version) from PyPI"
  elif [ "$pkg" = ccache ]; then
    # ccache only makes the build faster. A pass-through keeps the launcher and the stats
    # calls in ci.yml working: flags are accepted and ignored, a compile line is run as is.
    printf '#!/bin/sh
case "$1" in -*) exit 0 ;; esac
exec "$@"
' | sudo tee /usr/local/bin/ccache >/dev/null
    sudo chmod +x /usr/local/bin/ccache
    echo "ci_apt: ccache replaced by a pass-through; this build is uncached"
  else
    echo "ci_apt: $pkg is not installed and has no fallback" >&2
    missing=1
  fi
done
exit "$missing"
