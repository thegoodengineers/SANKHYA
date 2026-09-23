# SPDX-License-Identifier: Apache-2.0
"""Optional adapters over the SANKHYA Python bindings for other modelling libraries (#536).

Nothing in this subpackage is imported by ``sankhya`` itself - none of PuLP, Pyomo or CVXPY
is a dependency of the core package, only of whichever adapter module you import. A planner
who already has a PuLP, Pyomo or CVXPY model changes one line - the solver call - rather
than rewriting the model:

    import sankhya.adapters.pulp_solver as sankhya_pulp
    problem.solve(sankhya_pulp.SANKHYA(msg=False))

    import sankhya.adapters.cvxpy_solver as sankhya_cvxpy
    problem.solve(method=sankhya_cvxpy.SANKHYA)

    import sankhya.adapters.pyomo_plugin  # registers "sankhya" with Pyomo's SolverFactory
    pyomo.environ.SolverFactory("sankhya").solve(model)

Each adapter builds a `sankhya.Model` directly from the calling library's own in-memory
representation - variables, constraints, the objective - rather than by writing an MPS file
and reading it back, so the round trip costs one pass over the model, not a parser.
"""
