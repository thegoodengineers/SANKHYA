#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DIAG, not for merge: the rival's own log on one MIPLIB 2017 model, as MIP or LP relaxation."""
import sys

import highspy

name, relax = sys.argv[1], sys.argv[2] == "relax"
h = highspy.Highs()
h.setOptionValue("threads", 1)
h.setOptionValue("time_limit", 60.0)
h.setOptionValue("solve_relaxation", relax)
h.readModel(f"data/miplib2017-bench/{name}.mps.gz")
h.run()
info = h.getInfo()
print("DIAG-HIGHS", name, sys.argv[2], h.modelStatusToString(h.getModelStatus()),
      info.objective_function_value, "simplex iterations", info.simplex_iteration_count,
      "run time", h.getRunTime(), flush=True)
