# SPDX-License-Identifier: Apache-2.0
"""Vercel entry point: the same FastAPI app as server.py, found one directory up."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import app  # noqa: E402,F401
