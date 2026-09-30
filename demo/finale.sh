#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The finale walk (#758); the steps are in demo/finale.py.  demo/finale.sh [--dry]
set -euo pipefail
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
exec "$PYTHON" "$(dirname "${BASH_SOURCE[0]}")/finale.py" "$@"
