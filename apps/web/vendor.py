# SPDX-License-Identifier: Apache-2.0
"""Copy what the Vercel function needs from the repository into apps/web/vendor.

The checker (tools/verify_solution*.py) and the demo models (demo/*.mps) live at the
repository root; a Vercel function only sees its project directory. Run before deploying:

    python apps/web/vendor.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for sub, pattern in (("tools", "verify_solution*.py"), ("demo", "*.mps")):
    dst = HERE / "vendor" / sub
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / sub).glob(pattern)):
        shutil.copyfile(f, dst / f.name)
        print("vendored", f.relative_to(REPO).as_posix())
