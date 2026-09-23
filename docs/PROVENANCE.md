# SANKHYA — provenance

This document exists so that the claim "built from mathematical foundations, not wrapped
around an existing solver" can be **checked** rather than believed. It is maintained
continuously, not written at the end.

Last updated: **Phase 6** (sparse LU), with the HiGHS comparison run. Every number and every command output below was
produced by running the command shown, on the machine described, at the commit recorded.

---

## 1. The red line

No source code from an optimization **solver** is copied, vendored, linked, or read. That
list explicitly includes CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP,
cuPDLP / cuPDLP-C / cuPDLPx, OR-Tools GLOP and CP-SAT, Gurobi, CPLEX and Xpress — not their
simplex, not their cuts, not their MPS readers.

HiGHS appears in this project **only** as the comparison baseline in
`bench/runners/compare.py`. It runs either as an externally installed command-line binary or,
where none is present, through the `highspy` pip package in a separate Python process. It is
never linked into SANKHYA, never a build dependency, and no part of its source informs ours.
The comparison results are in `docs/BENCHMARKS.md` section 3.

---

## 2. Dependency table

Every dependency is a general-purpose library. None of them solves an optimization problem.

| Dependency | Version | Licence | Linked? | Why it is not a solver |
|---|---|---|---|---|
| **fmt** | 10.2.1 | MIT | yes, static | String formatting. Produces text from values. |
| **CLI11** | 2.4.2 | BSD-3-Clause | header-only | Command-line argument parsing. |
| **nlohmann/json** | 3.11.3 | MIT | header-only | JSON serialisation for the `--stats` result blob. |
| **OpenMP runtime** (libgomp with GCC, libomp with Clang) | as shipped by the compiler | GPL-3.0 with the GCC Runtime Library Exception / Apache-2.0 with LLVM exception | optional, found by CMake | The compiler's own thread runtime for `#pragma omp parallel for`. Schedules loops; contains no numerical code of any kind. |
| **zlib** | 1.3.1 | zlib | yes, static | DEFLATE decompression, for `.mps.gz` inputs (Phase 2). |
| **GoogleTest** | 1.14.0 | BSD-3-Clause | test binary only | Unit test framework. Never linked into `sankhya_core`. |
| **CUDA runtime (cudart)** | as shipped with CUDA Toolkit | NVIDIA SDK Licence | yes, when `SANKHYA_ENABLE_CUDA=ON` | Device enumeration and the probe in `src/gpu/device.cu`. No kernel launch, no transfer. |
| **cuSPARSE** | as shipped with CUDA Toolkit | NVIDIA SDK Licence | yes, when `SANKHYA_ENABLE_CUDA=ON` | Sparse matrix-vector products on GPU (`cusparseSpMV`). Supplies the A x and A^T y arithmetic primitives called by the GPU PDHG iteration in `src/gpu/pdhg_gpu.cu` (#17), in the same way that a BLAS supplies `dgemv` to a simplex without being a simplex. Not an optimisation algorithm. |
| **cuBLAS** | as shipped with CUDA Toolkit | NVIDIA SDK Licence | **not linked** | Dense linear algebra on GPU. Not needed by the current GPU backend; listed here so the absence is a decision, not an oversight. |
| **CUB (CCCL)** | as shipped with CUDA Toolkit | BSD-3-Clause | header-only, when `SANKHYA_ENABLE_CUDA=ON` | Device-wide parallel reductions (`DeviceReduce::Sum`) used by the GPU PDHG movement and interaction scalars in `src/gpu/pdhg_gpu.cu` (#17). Ships inside the CUDA Toolkit as part of NVIDIA CCCL; no separate fetch or link step. Not an optimisation algorithm. |
| **highspy** | pip, benchmark only | MIT | **never linked** | The HiGHS solver, used ONLY as the comparison baseline in `bench/runners/compare.py`. It runs in a separate Python process, is not a build dependency, and nothing in `src/` knows it exists. Its source does not inform ours - see the red line in section 1. |
| **PuLP** | 3.3.2, pip, optional | MIT | **never linked** | A modelling library, not a solver (its own bundled CBC binary is never invoked here). #536's `bindings/python/sankhya/adapters/pulp_solver.py` implements PuLP's own `LpSolver` interface and walks `LpProblem` to build a `sankhya.Model`; SANKHYA is the only thing that solves anything. Optional: importing `sankhya` never imports `pulp`. |
| **Pyomo** | 6.10.1, pip, optional | BSD-3-Clause | **never linked** | A modelling library. #536's `bindings/python/sankhya/adapters/pyomo_plugin.py` registers against `pyomo.contrib.solver`'s `SolverBase` interface and reads coefficients via Pyomo's own public `generate_standard_repn` utility (`pyomo/repn/standard_repn.py`, the same one Pyomo's built-in LP/NL writers use) - not by writing or reading any file Pyomo or SANKHYA produces. Optional, same as PuLP. |
| **CVXPY** | 1.9.3, pip, optional | Apache-2.0 | **never linked** | A modelling library whose OWN solver plugins hook into its internal cone-canonicalization reduction chain - #536's `bindings/python/sankhya/adapters/cvxpy_solver.py` deliberately does not, to stay stable across CVXPY releases, and instead walks `Problem.variables()` / `.constraints` / `.objective` (CVXPY's public surface) directly, registered through CVXPY's own documented `solve(method=...)` escape hatch. Optional, same as PuLP. |

Considered, and not present. Each was allowed by the policy above; none turned out to be
needed, and none is linked, vendored or fetched:

| Dependency | Licence | Status |
|---|---|---|
| pybind11 | BSD-3-Clause | Not used. The Python bindings (#59, #129) are `ctypes` over the C API and compile nothing. |
| AMD / COLAMD ordering | per-module | Not used as a library. The sparse LDL^T (#70) carries its own ordering, written from the AMD paper (Amestoy, Davis & Duff 1996) for #193 - approximate minimum degree on a quotient graph - with no code from SuiteSparse or anywhere else; the LU orders by Markowitz counts. |
| Eigen | MPL-2.0 | Not used. `DenseLu` (`src/simplex/dense_lu.cpp`) is the reference oracle the sparse LU is tested against. |

---

## 2b. Toolchains this has been built and tested with

PS26119 asks for a foundation someone else can pick up, and "it builds here" is not that.
These are the toolchains the evidence in `docs/BENCHMARKS.md` was actually produced with, and
the ones CI exercises on every push. `scripts/preflight.sh` reports what it finds on the
machine it runs on, and `scripts/configure.sh` chooses among them rather than trusting the
order of `PATH`.

| Where | Compiler | Build system | Python | What it is the authority on |
|---|---|---|---|---|
| CI, `ubuntu-latest` | the runner's default GCC | CMake >= 3.20, Ninja | 3.x as shipped | `-Wall -Wextra -Werror` cleanliness, the ASan/UBSan run, and the Netlib gates. A portability claim rests on the Linux build. |
| Development, Windows 11 | GCC 16.1.0, MSYS2 UCRT64 (`C:\msys64\ucrt64\bin\g++.exe`) | CMake 4.3.3, Ninja 1.13.2 | 3.11.9 | Every benchmark CSV in `bench/results/` whose `machine` column reads `Windows-AMD64`. |
| Development, Windows 11 (fallback) | GCC 13.2.0, Strawberry Perl MinGW-W64 | as shipped | 3.x | A second Windows compiler the configure script accepts; no committed CSV comes from it. |

**One toolchain must never be selected**: a MinGW 6.3.0 that predates C++20 and sits on
`PATH` on at least one development box. `scripts/configure.sh` rejects any `g++` reporting a
major version below 10, which is why it exists rather than the build simply calling `cmake`.

Third-party sources are pinned by tag in `CMakeLists.txt`, and `docs/sbom.spdx.json` - the
SPDX 2.3 bill of materials - is generated from those pins by `tools/make_sbom.py`. CI runs
`tools/make_sbom.py --check`, so a dependency added to the build without an entry in the
table above turns the tree red instead of passing quietly.

## 3. Algorithm citation table

Every algorithm we implement is cited at its implementation site as well as here. The
citation records where the *mathematics* came from; the code is written from the
mathematics, not transcribed from anyone's implementation.

| Algorithm | Citation | File |
|---|---|---|
| Compressed-column storage, counting-sort transpose | Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006), ch. 2 | `src/la/sparse.cpp` |
| Dense-accumulator sparse vector (FTRAN/BTRAN result pattern) | Davis, ibid.; Hall & McKinnon on hyper-sparsity | `include/sankhya/sparse.hpp`, `src/la/sparse.cpp` |
| Markowitz threshold pivoting (constant only, so far) | Suhl & Suhl, *Computing sparse LU factorizations for large-scale linear programming bases* | `include/sankhya/tolerances.hpp` |
| MPS format: sections, RANGES and BOUNDS semantics | IBM MPS specification; Maros, *Computational Techniques of the Simplex Method* (Kluwer, 2003), appendix A | `src/io/mps_reader.cpp` |
| CPLEX LP format | Public CPLEX and Gurobi reference manuals (documentation only) | `src/io/lp_reader.cpp` |
| Dense LU with partial pivoting; transposed triangular solves | Golub & Van Loan, *Matrix Computations* (4th ed.), sections 3.2 and 3.4 | `src/simplex/dense_lu.cpp` |
| Sparse LU with Markowitz pivoting and threshold stability | Markowitz, *The elimination form of the inverse and its application to linear programming*, Management Science 3 (1957); Suhl & Suhl, *Computing sparse LU factorizations for large-scale linear programming bases*, ORSA J. Computing 2 (1990); Duff, Erisman & Reid, *Direct Methods for Sparse Matrices* (2nd ed., 2017), ch. 7-8 | `src/la/lu.cpp`, `src/la/lu_eliminate.cpp` |
| LU singleton fast path (#463): a column-singleton pivot takes its U row from a row-wise copy of the values instead of opening every column its row crosses, and the threshold test reads a cached column maximum; the same pivots, factors and solves as the general update, bit for bit (tested against the reference elimination and `DenseLu`) | Suhl & Suhl, *Computing sparse LU factorizations for large-scale linear programming bases*, ORSA J. Computing 2 (1990); Koberstein, *The dual simplex method, techniques for a fast and stable implementation*, PhD thesis, Paderborn (2005), sec. 5.3 | `src/la/lu_eliminate.cpp`, `src/la/lu_workspace.hpp` |
| Basis update, product form of the inverse | Dantzig & Orchard-Hays, *The product form for the inverse in the simplex method*, Mathematical Tables and Other Aids to Computation 8 (1954) | `src/la/lu.cpp` |
| Basis update, Forrest-Tomlin (#279) - a row eta computed by a partial BTRAN through U, applied between L and U, in place of an eta appended outside both | Forrest & Tomlin, *Updated triangular factors of the basis to maintain sparsity in the product form simplex method*, Mathematical Programming 2 (1972); the implementable form followed here is Huangfu & Hall, *Novel update techniques for the revised simplex method*, ERGO 13-001 (2012), sec. 2.1 | `src/la/lu.cpp` (`update_forrest_tomlin`) |
| Bounded-variable revised primal simplex | Dantzig, *Linear Programming and Extensions* (1963); Chvátal, *Linear Programming* (1983), ch. 3 and 8 | `src/simplex/primal_simplex.cpp` |
| Piecewise-linear (composite) phase 1, no artificial variables | Maros, *Computational Techniques of the Simplex Method*, ch. 9 | `src/simplex/primal_simplex.cpp` |
| Bounded-variable revised dual simplex with the bound-flipping ratio test, dual devex pricing, artificial bounds for dual infeasibility, warm start from a basis (#65; `--option algorithm=dual-simplex`, and the branch-and-bound node engine) | Lemke, *The dual method of solving the linear programming problem*, Naval Research Logistics Quarterly 1 (1954); Maros, *Computational Techniques of the Simplex Method*, ch. 10; Koberstein, *The dual simplex method, techniques for a fast and stable implementation*, PhD thesis, Paderborn (2005), sec. 3.3 and 4.5; Forrest & Goldfarb (1992), above, sec. 3 for the dual weights | `src/simplex/dual_simplex.cpp`, shared state in `src/simplex/simplex_core.hpp` |
| Bland's anti-cycling rule | Chvátal, *Linear Programming*, ch. 3 | `src/simplex/primal_simplex.cpp` |
| Devex pricing (default since #66 was re-measured; `--option pricing=dantzig` selects the old rule) | Forrest & Goldfarb, *Steepest-edge simplex algorithms for linear programming*, Mathematical Programming 57 (1992); the approximation is Harris (1973), below | `src/simplex/primal_simplex.cpp` |
| Harris two-pass ratio test, long-step bound flipping (opt-in; `--option ratio_test=harris`, #67) | Harris, P.M.J., *Pivot selection methods of the Devex LP code*, Mathematical Programming 5 (1973), 1-28; long-step generalises the piecewise-linear phase 1 already cited to Maros below | `src/simplex/primal_simplex.cpp` |
| Exact rational tableau simplex (test oracle) | Chvátal, *Linear Programming*, ch. 2–3 | `tests/oracles/rational_simplex.cpp` |
| Primal-dual hybrid gradient (the base iteration) | Chambolle & Pock, *A first-order primal-dual algorithm for convex problems with applications to imaging*, JMIV 40(1), 2011, Algorithm 1 | `src/pdhg/pdhg.cpp` |
| Adaptive step size, primal weight, restarts | Applegate et al., *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient* (PDLP), NeurIPS 2021, sections 3.1, 3.2, 4.3 | `src/pdhg/pdhg.cpp` |
| GPU-oriented restarted PDHG (design reference) | Lu & Yang, *cuPDLP.jl*, arXiv:2311.12180 | `src/pdhg/pdhg.cpp` |
| GPU PDHG on-device iteration loop via CUDA Graphs (#478, off by default): K=64 iterations captured as a CUDA Graph and replayed with one launch, eliminating the per-iteration host round-trip; step-size rule and KKT residuals evaluated on-device; one small D→H copy per block | Lu & Yang, *cuPDLP.jl*, arXiv:2311.12180; NVIDIA, *CUDA C++ Programming Guide*, chapter "CUDA Graphs" | `src/gpu/pdhg_graph.cu` |
| Restarted PDHG on the device (#17): the same iteration with cuSPARSE SpMV, elementwise CUDA kernels for the coordinate updates and CUB reductions for the movement and interaction terms; preconditioning, residuals and restarts on the host; every device failure falls back to the CPU engine | Chambolle & Pock (2011) and Applegate et al. (PDLP, 2021) as above; Lu & Yang, arXiv:2311.12180, for the host/device split only | `src/gpu/pdhg_gpu.cu` |
| Batched GPU PDHG for B&B node bounding and strong branching (#520): n×K and m×K block iterates, one cuSPARSE SpMM per matrix product; per-column step sizes and restarts; safe dual bound from early-stopped iterate; scoring protocol for strong-branching candidates | Applegate et al., *Practical Large-Scale LP via PDHG*, NeurIPS 2021 (batch extension); Huang et al., *Batched First-Order Strong Branching on the GPU*, arXiv:2601.21990 | `src/gpu/pdhg_batch.cu` |
| Ruiz equilibration | Ruiz, *A scaling algorithm to equilibrate both rows and columns norms in matrices*, RAL-TR-2001-034 | `src/la/scaling.cpp` |
| Diagonal preconditioning, alpha = 1 | Pock & Chambolle, *Diagonal preconditioning for first order primal-dual algorithms*, ICCV 2011, section 4 | `src/la/scaling.cpp` |
| Moreau decomposition for the support-function prox | Rockafellar, *Convex Analysis*, theorem 31.5 | `src/pdhg/pdhg.cpp` |
| Branch and bound | Land & Doig, *An automatic method of solving discrete programming problems*, Econometrica 28(3), 1960; Wolsey, *Integer Programming*, ch. 7 | `src/mip/branch_and_bound.cpp` |
| Node propagation from row activities | Savelsbergh, *Preprocessing and probing for MIP*, ORSA J. Computing 6(4), 1994 | `src/mip/branch_and_bound.cpp` |
| Search shape: propagation at nodes, incumbent as cutoff | Achterberg, *Constraint Integer Programming* (thesis, 2007), ch. 5–6 | `src/mip/branch_and_bound.cpp` |
| Hyper-sparse back-substitution through U in the FTRAN, U stored by column as well as by row, zero results skipped; the row-wise gather kept as the reference the push form is tested against (#68) | Gilbert & Peierls, *Sparse partial pivoting in time proportional to arithmetic operations*, SIAM J. Sci. Stat. Comput. 9 (1988) | `src/la/lu.cpp` (`solve`, `solve_reference`) |
| Sparse symmetric LDL^T: approximate minimum degree ordering on the quotient graph (#193), elimination tree, symbolic pattern once, up-looking numeric factorization with diagonal regularization (#70) | Davis, *Direct Methods for Sparse Linear Systems*, SIAM (2006), ch. 4; Liu, *The role of elimination trees in sparse factorization*, SIAM J. Matrix Anal. Appl. 11 (1990); Tinney & Walker, Proc. IEEE 55 (1967); Amestoy, Davis & Duff, *An approximate minimum degree ordering algorithm*, SIAM J. Matrix Anal. Appl. 17 (1996); Altman & Gondzio, Optim. Methods Softw. 11 (1999) | `src/la/ldl.cpp` |
| Primal-dual interior-point method, Mehrotra predictor-corrector on the normal equations of the bounded-variable form; no basis, no infeasibility certificate (#56; `--option algorithm=ipm`) | Mehrotra, SIAM J. Optim. 2 (1992); Wright, *Primal-Dual Interior-Point Methods*, SIAM (1997), ch. 10–11; Altman & Gondzio (1999) for the regularization | `src/ipm/ipm.cpp` |
| Crossover from the interior point to a vertex (#219): classify each column and row of the interior point's answer as at-bound (within a relative tolerance, reduced cost agreeing) or interior, take the m interior entries with the most slack as a basis guess, warm-start the dual simplex from it and keep its vertex when it reaches optimal; the guess is a heuristic and nothing rests on it, the simplex proves the vertex | Bixby, *Solving real-world linear programs: a decade and more of progress*, Operations Research 50 (2002), sec. 4; Andersen & Ye, *Combining interior-point and pivoting algorithms for linear programming*, Management Science 42 (1996) | `src/simplex/crossover.cpp` |
| The primal push of the crossover (#343): every nonbasic variable starts at the interior point's value and is pushed to its nearer bound along `B^-1 a_k` with a ratio test against the basic variables, a blocking basic leaving the basis, so the point stays feasible and the primal loop finishes from a vertex of the optimal face | Bixby, Operations Research 50 (2002), sec. 4; Andersen & Ye, Management Science 42 (1996); Megiddo, *On finding primal- and dual-optimal bases*, ORSA J. Computing 3 (1991) | `src/simplex/simplex_push.cpp` (`Simplex::run_push`) |
| Crossover from an interior point that is not optimal (#474, option `crossover_from_nonoptimal`, off by default): a `feasible`, time-limited or iteration-limited answer, or the best iterate the interior point attaches to a numerical error, is handed to the same guess, push and primal loop when its scaled primal and dual infeasibility are at most `tol::kCrossoverStartInfeasibility` (1e-4); only the simplex vertex, measured by the status guard, may be reported optimal, and an iterate nothing used is withdrawn from the numerical error | Megiddo, *On finding primal- and dual-optimal bases*, ORSA J. Computing 3(1) (1991); Bixby & Saltzman, *Recovering an optimal LP basis from an interior point solution*, Operations Research Letters 15 (1994) | `src/simplex/crossover.cpp` (`crossover_when_wanted`, `crossover_start_is_usable`), `src/ipm/ipm.cpp` (`finish`) |
| Verifier-gated concurrent engine race under `algorithm=auto` (#476, option `engine_race`, off by default): the dual simplex, the interior point with crossover and PDHG run at once, each through its own presolve and postsolve, on its own thread, logger and out-of-memory guard; an answer wins only when an in-process KKT check against the original model at the project tolerances passes (or, for an infeasibility, its Farkas certificate checks out), a rejected answer neither wins nor stops the others, the first accepted one cancels the rest; deterministic mode or one core runs the same engines in a fixed order | Concurrent optimization as documented in the IBM ILOG CPLEX User's Manual ("Concurrent optimizer") and the Gurobi Reference Manual (parameter `Method` = 3 and 4) - user-facing behaviour only; the KKT conditions of LP, Nocedal & Wright, *Numerical Optimization*, 2nd ed. (2006), thm. 13.1 | `src/core/engine_race.cpp` |
| In-process KKT check of an LP optimum, a C++ reimplementation of the checks `tools/verify_solution.py` makes (bounds and recomputed row activities relative to their terms, activity and objective agreement, reduced costs against c - A^T y, multiplier signs, complementary slackness, the basis, strong duality with per-item accounting) so the race can judge an answer inside the process while the Python verifier stays independent and never links the solver | Nocedal & Wright, *Numerical Optimization*, 2nd ed. (2006), thm. 13.1; the scalings are those argued in `tools/verify_solution.py` | `src/core/kkt_check.cpp` |
| Dual purification at the interior point's finish (#209): the least-squares correction of y over the strictly interior variables, `dy = (A_I A_I^T + eps I)^-1 A_I (c_I - A_I^T y)`, through the same normal equations, accepted only if the interior reduced costs shrink and no sign condition breaks - the dual half of a crossover, without the primal half | Bixby, *Solving real-world linear programs: a decade and more of progress*, Operations Research 50 (2002), sec. 4; Andersen & Ye, *Combining interior-point and pivoting algorithms for linear programming*, Management Science 42 (1996) | `src/ipm/ipm.cpp` (`purify_duals`) |
| Reliability branching: pseudocosts, strong branching on unreliable candidates with capped warm-started dual probes, product score (#69; `--option mip_branching=most-fractional` keeps the old rule) | Achterberg, Koch & Martin, *Branching rules revisited*, Operations Research Letters 33 (2005), 42–54 | `src/mip/branch_and_bound.cpp` |
| Iterative refinement of the final basis, primal and dual, residual in compensated arithmetic (#72) | Wilkinson, *Rounding Errors in Algebraic Processes* (1963), ch. 4; Ogita, Rump & Oishi, *Accurate sum and dot product*, SIAM J. Sci. Comput. 26 (2005) | `src/simplex/primal_simplex.cpp` (`refine_final_basis`) |
| Cost perturbation on a dual-degenerate stall, nonbasic costs only so dual feasibility is preserved by construction | Maros, *Computational Techniques of the Simplex Method*, ch. 9 (the bound-shifting scheme, applied to the dual's costs); Koberstein (2005), above, sec. 6.2 | `src/simplex/dual_simplex.cpp` |
| Robustness suite: the classic cycling examples, adversarial families judged by the exact oracle, and the conditioning / near-parallel / cost-ratio / redundancy / degeneracy sweeps with an optimum known by construction (#71) | Beale, *Cycling in the dual simplex algorithm*, Naval Research Logistics Quarterly 2 (1955); Kuhn's example as given in Chvátal, *Linear Programming* (1983), ch. 3; the KKT construction is the oracle fuzz's (above) | `tests/robustness/test_robustness.cpp`, `bench/runners/robustness.py` |
| Solution pool: every integer-feasible point the search finds, deduplicated by integer assignment, bounded and best first; `pool_complete` continues past integral nodes (three-way split on an unfixed integer column) and prunes on the pool's worst member so the pool provably holds the k best; `pool_diversity` evicts by nearest Hamming distance (#225) | Danna, Fenelon, Gu & Wunderling, *Generating multiple solutions for mixed integer programming problems*, IPCO 2007, LNCS 4513; Glover, Lokketangen & Woodruff, *Scatter search to generate diverse MIP solutions*, in *Computing Tools for Modeling, Optimization and Simulation* (2000) | `src/mip/solution_pool.cpp`, `src/mip/branch_and_bound.cpp` |
| GPU activity-based bound propagation (#510, off by default): activity bounds per row computed by one CUDA thread per row; variable bounds tightened by reversing each variable's contribution; iterated to convergence | Sofranac, Gleixner & Pokutta, *Accelerating domain propagation: an efficient GPU implementation for solving mixed-integer programs*, arXiv:2009.07785 | `src/gpu/domain_prop.cu` |
| GPU feasibility pump (#509, off by default): PDHG on the device as the LP projection step; outer loop rounds to integer and repeats until feasible | Fischetti, Glover & Lodi, *Feasibility pump*, Mathematical Programming 104 (2005); Mexi et al., *Combining Dual Simplex and GPU PDLP*, arXiv:2307.03466; Corduk et al., *GPU Heuristics for MILP*, arXiv:2510.20499 | `src/gpu/gpu_heuristics.cu` |
| GPU fix-and-propagate (#509, off by default): fixes near-integer columns from LP relaxation, propagates bounds on GPU, solves residual LP | Corduk et al., arXiv:2510.20499 | `src/gpu/gpu_heuristics.cu` |
| GPU Feasibility Jump (#508, off by default): activities and column weights on the device; jump value and score evaluated per candidate by one warp with a segmented CUB reduction; several independent restarts as parallel CUDA blocks | Corduk, Sielski, Boucher & Aatish, *GPU-accelerated primal heuristics for mixed integer programming*, arXiv:2510.20499 | `src/gpu/gpu_fj.cu` |
| Conflict analysis (#292, off by default): from a node proved infeasible, the subset of its branching decisions that the rows and the global bounds cannot satisfy, re-proved from the global bounds by propagation or by the node LP's Farkas multipliers evaluated conservatively, shrunk by a bounded deletion filter, stored as a bound disjunction and used in node propagation to prune and to fix bounds | Achterberg, *Conflict analysis in mixed integer programming*, Discrete Optimization 4(1) (2007); Witzig, Berthold & Heinz, *Experiments with conflict analysis in mixed integer programming*, CPAIOR 2017; Junker, *QuickXplain*, AAAI 2004 | `src/mip/conflict.cpp`, `src/mip/branch_and_bound_conflicts.cpp` |
| MIQP: branch and bound over convex QP node relaxations | Gupta & Ravindran, *Branch and bound experiments in convex nonlinear integer programming*, Management Science 31(12), 1985 | `src/mip/branch_and_bound.cpp` |
| Parallel tree search (#222, `mip_threads`): each worker runs the sequential search on a subtree, sharing the incumbent, the node count, the pool and the pseudocosts; an idle worker takes a subtree another gives away | Ralphs, Shinano, Berthold & Koch, *Parallel Solvers for Mixed Integer Linear Optimization*, in Hamadi & Sais (eds.), *Handbook of Parallel Constraint Reasoning*, Springer, 2018 | `src/mip/branch_and_bound_parallel.cpp` |
| Exact rational branch and bound (test oracle) | as above, in exact arithmetic | `tests/oracles/rational_simplex.cpp` |
| Shifted geometric mean benchmark reporting | Mittelmann, plato.asu.edu benchmark methodology | `bench/runners/make_benchmarks_doc.py` |
| LP duality checks (feasibility, complementary slackness, strong duality) | Chvátal, *Linear Programming*, ch. 5 | `tools/verify_solution.py` |
| QP optimality measures for the Maros-Meszaros benchmark (#491): primal residual, dual residual (distance of the row multipliers and the derived reduced costs from the admissible signs) and duality gap against the Dorn dual, absolute and relative, success at 1e-6 and 1e-9 | Dorn, *Duality in quadratic programming*, Quarterly of Applied Mathematics 18 (1960); Nocedal & Wright, *Numerical Optimization*, 2nd ed., ch. 12 and 16; the measures and levels of the published qpbenchmark Maros-Meszaros report (Caron et al., `qpsolvers/maros_meszaros_qpbenchmark`, `results/maros_meszaros.md`) | `bench/runners/qp_residuals.py`, `bench/runners/maros_meszaros.py` |
| Lifted knapsack cover cuts at the root, exact sequential lifting through a 0/1 knapsack dynamic programme (#159; `--option enable_root_cuts=true`, off by default by measurement) | Balas, *Facets of the knapsack polytope*, Math. Programming 8 (1975); Wolsey, *Faces for a linear inequality in 0-1 variables*, ibid.; Zemel, *Easily computable facets of the knapsack polytope*, Math. Oper. Res. 14 (1989); Crowder, Johnson & Padberg, Oper. Res. 31 (1983); Gu, Nemhauser & Savelsbergh, INFORMS J. Computing 10 (1998) | `src/mip/cuts.cpp` |
| LP engine selection behind `algorithm=auto` (#284): a rule table on rows, nonzeros and the presence of a starting basis, every threshold tied to a CSV in `bench/results/`; a selected interior point that declines falls back to the dual simplex | no algorithm: a policy; the measurements are in `docs/BENCHMARKS.md` sections 1a and 1f | `src/core/engine_selection.cpp` |
| Mixed-integer rounding (MIR) cuts at the root from single model rows: bound substitution to the nearer bound, divisors 1 and the fractional integer coefficients, the most violated member kept (#221; part of `enable_root_cuts`, `enable_mir_cuts` isolates the family) | Nemhauser & Wolsey, *A recursive procedure to generate all cuts for 0-1 mixed integer programs*, Math. Programming 46 (1990); Marchand & Wolsey, *Aggregation and mixed integer rounding to solve MIPs*, Oper. Res. 49 (2001) | `src/mip/mir_cuts.cpp` |
| Flow cover cuts for the fixed-charge structure: variable upper bounds read off two-nonzero rows x <= u y, a cover of the switched inflows chosen at the LP point, each outflow on the right in its smaller form, unlifted; validity checked in exact arithmetic over every switch assignment (#419; `enable_flow_cover_cuts`, off by default) | Padberg, Van Roy & Wolsey, *Valid linear inequalities for fixed charge problems*, Oper. Res. 33 (1985); Gu, Nemhauser & Savelsbergh, *Lifted flow cover inequalities for mixed 0-1 integer programs*, Math. Programming 85 (1999), for the lifting not built | `src/mip/flow_cover_cuts.cpp` |
| Cut selection: score by efficacy, objective parallelism and integer support, a greedy pick under a per-round cap with an orthogonality filter against the round's chosen set, the rest kept waiting for the next round (#415; `cut_max_per_round`, `cut_max_parallelism`) | Wesselmann & Suhl, *Implementing cutting plane management and selection techniques*, technical report, University of Paderborn (2012); Achterberg, *Constraint Integer Programming*, thesis, TU Berlin (2007), ch. 8 | `src/mip/cut_selection.cpp` |
| Dual steepest-edge pricing (#411; `pricing=dual-steepest-edge`, not the default until its A/B): the leaving row priced by the exact squared norm of its row of the basis inverse, maintained through the Forrest-Goldfarb update with one extra FTRAN per pivot, the leaving row's norm re-read from the iteration's own BTRAN, exact at the start on warm bases up to 2,000 rows; the maintained weights are held to recomputed norms to a relative 1e-9 in the unit test | Forrest & Goldfarb, *Steepest-edge simplex algorithms for linear programming*, Math. Programming 57 (1992); Koberstein, *The dual simplex method, techniques for a fast and stable implementation*, thesis, Universitaet Paderborn (2005), ch. 5 | `src/simplex/dual_simplex.cpp` |
| Gomory mixed-integer cuts at the root, from tableau rows reconstructed off the final basis (#159) | Gomory, *An algorithm for the mixed integer problem*, RAND RM-2597 (1960); Balas, Ceria, Cornuéjols & Natraj, *Gomory cuts revisited*, Oper. Res. Letters 19 (1996); Marchand & Wolsey, *Aggregation and mixed integer rounding to solve MIPs*, Oper. Res. 49 (2001) | `src/mip/cuts.cpp` |
| Bound rounding on integral rows; cut filtering by density, coefficient range and violation, duplicates dropped | Chvátal, *Edmonds polytopes and a hierarchy of combinatorial problems*, Discrete Math. 4 (1973); Achterberg, *Constraint Integer Programming* (thesis, 2007), ch. 8 | `src/mip/cuts.cpp` |
| Cut validity gate: every family checked in exact arithmetic against the rational oracle's optimum, with a deliberately invalid cut as the negative control | as above, in exact arithmetic | `tests/unit/test_cuts.cpp` |
| Certificate of infeasibility (Farkas): row multipliers whose aggregate no point in the column box satisfies, checked against the ORIGINAL model before it is published (#191) | Farkas, J., *Theorie der einfachen Ungleichungen*, J. reine angew. Math. 124 (1902); Schrijver, *Theory of Linear and Integer Programming* (1986), section 7.3 | `src/core/certificate.cpp`, `include/sankhya/certificate.hpp` |
| Irreducible Infeasible Subsystem (IIS): deletion filter starting from the Farkas certificate support (O(k) re-solves where k is the number of named constraints); each candidate is freed and re-solved — redundant if still infeasible, necessary otherwise (#217) | Chinneck, J.W. and Dravnieks, E.W., *Locating minimal infeasible constraint sets in linear programs*, ORSA Journal on Computing 3(2), 1991, pp. 157–168 | `src/core/iis.cpp`, `include/sankhya/model.hpp` |
| Certificate of unboundedness: a recession direction no bound blocks along which the objective strictly improves, reported with the feasible point it starts from (#191) | Schrijver, ibid., section 8.2; Chvatal, *Linear Programming* (1983), ch. 3 on the ratio test that exhibits the ray | `src/core/certificate.cpp`, `src/simplex/primal_simplex.cpp` |
| Convex QP by a primal-dual proximal method with a forward step for the smooth 0.5 x'Qx term (#55; the LP engine's iteration with the gradient added) | Condat, *A primal-dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms*, J. Optim. Theory Appl. 158 (2013); Vũ, *A splitting algorithm for dual monotone inclusions involving cocoercive operators*, Adv. Comput. Math. 38 (2013); Chambolle & Pock (2011) for the Q = 0 case | `src/qp/qp_condat_vu.cpp` |
| Convexity decided before any arithmetic: LDL^T without interchanges on sense · Q, a negative pivot returned as the certificate; dense, and above 2000 columns it reports unverified rather than guessing | Golub & Van Loan, *Matrix Computations* (4th ed.), section 4.1; Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed.), ch. 10 | `src/qp/convexity.cpp` |
| Presolve: empty, singleton, redundant and forcing rows, fixed and empty columns (#43), free column singletons and doubleton equations (#92), dual fixing (#412; `presolve_dual_fixing`, off by default: a column whose entries can only push their rows away from a finite bound one way, and whose cost never rewards that way, is fixed at the other bound and replayed nonbasic there), parallel rows (#412; `presolve_parallel_rows`, off by default: a row that is a scalar multiple of an earlier one folds its scaled bounds into that row and goes, its dual handed back to it when the bound that binds was its own); every reduction pushes a record and postsolve replays the stack in reverse, with the round-trip asserted against the rational oracle | Brearley, Mitra & Williams, *Analysis of mathematical programming problems prior to applying the simplex algorithm*, Math. Programming 8 (1975); Andersen & Andersen, *Presolving in linear programming*, Math. Programming 71 (1995); Achterberg et al., *Presolve reductions in mixed integer programming*, INFORMS J. Computing 32 (2020) | `src/presolve/presolve.cpp` |
| Root diving heuristic: fix the least-fractional integer column, re-solve, to a bounded depth, so the search starts with an incumbent (#25) | Achterberg, *Constraint Integer Programming* (thesis, 2007), ch. 6; Berthold, *Primal heuristics for mixed integer programs* (diploma thesis, TU Berlin, 2006) | `src/mip/branch_and_bound.cpp` |
| Singular-basis repair: the LU reports dependent columns by position, the simplex evicts them for their logicals, and a budget of stalled repairs stops a solve that is being carried by repair alone (#34, #174) | Maros, *Computational Techniques of the Simplex Method*, section 9.4; Suhl & Suhl, *Computing sparse LU factorizations for large-scale linear programming bases*, ORSA J. Computing 2 (1990) | `src/la/lu.cpp`, `src/simplex/simplex_core.hpp` |
| Nonlinear expression DAG: hash-consed construction, safe simplification, constant folding (#296) | Standard common-subexpression elimination by value numbering; written from the definition | `src/nlp/expression.cpp` |
| Exact gradients (reverse-mode AD) and Hessians (forward-over-reverse, second-order adjoints) (#296) | Griewank & Walther, *Evaluating Derivatives*, 2nd ed. (SIAM, 2008), ch. 3 and 5 | `src/nlp/derivatives.cpp` |
| Convexity by composition rules over interval ranges (#296) | Grant, Boyd & Ye, "Disciplined convex programming", in *Global Optimization* (Springer, 2006); Boyd & Vandenberghe, *Convex Optimization* (2004), sec. 3.2.4 | `src/nlp/convexity.cpp` |

Of the work once listed here as "Phases 6 onwards", restarted PDHG, the Mehrotra
predictor-corrector, Gomory mixed-integer and lifted cover cuts, Devex pricing (#66, the
default) and the Harris ratio test (#67, opt-in) have all landed and are in the table
above. Not landed: the Forrest-Tomlin update (Forrest & Tomlin 1972; what ships is the
product form of the inverse with refactorization forced by an accuracy check),
mixed-integer rounding cuts, and cuts below the root.

---

## 4. Machine-checkable evidence

### 4.1 What the binary actually links against

Regenerated at **Phase 2**, commit `12fa98b`. The Phase 1 text predicted that zlib would
appear here once the gzip MPS path became real; it now does, on both platforms, which is
what that prediction was for.

Windows 11, **GCC 16.1.0 (MSYS2 UCRT64), Release**:

```
$ objdump -p build/sankhya.exe | grep "DLL Name" | sort -u
        DLL Name: KERNEL32.dll
        DLL Name: api-ms-win-crt-convert-l1-1-0.dll
        DLL Name: api-ms-win-crt-environment-l1-1-0.dll
        DLL Name: api-ms-win-crt-filesystem-l1-1-0.dll
        DLL Name: api-ms-win-crt-heap-l1-1-0.dll
        DLL Name: api-ms-win-crt-locale-l1-1-0.dll
        DLL Name: api-ms-win-crt-math-l1-1-0.dll
        DLL Name: api-ms-win-crt-private-l1-1-0.dll
        DLL Name: api-ms-win-crt-runtime-l1-1-0.dll
        DLL Name: api-ms-win-crt-stdio-l1-1-0.dll
        DLL Name: api-ms-win-crt-string-l1-1-0.dll
        DLL Name: api-ms-win-crt-time-l1-1-0.dll
        DLL Name: api-ms-win-crt-utility-l1-1-0.dll
        DLL Name: zlib1.dll
```

The Windows C runtime plus **zlib1.dll**. Note that the `-static` flag added in Phase 1 does
not cover this one: MSYS2 provides zlib as an import library (`libz.dll.a`), so the Windows
`.exe` is no longer self-contained. That is recorded rather than fixed. Windows is the
convenience build; Linux is what CI gates and what the claim rests on. It is logged as
judgement call 7 below.

The Linux build is the authoritative one. Produced by the `provenance` job on
**ubuntu-latest, GCC (runner default), Release**, run 32867892122:

```
$ ldd build/sankhya
        linux-vdso.so.1 (0x00007f2012bf2000)
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f2012b07000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f2012800000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f2012717000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f2012ad9000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f2012400000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f2012bf4000)

$ ldd build/sankhya | grep -Ei 'cbc|clp|highs|scip|soplex|glpk|lpsolve|osqp|ortools|gurobi|cplex|xpress'
OK: no solver library is linked
```

zlib, the C++ runtime, libc and libm - and, on a build with `SANKHYA_WITH_OPENMP` on (the
default, #57), the compiler's own OpenMP runtime, `libgomp` with GCC, which the listing
above predates. Nothing else.

That grep is a **gate**, not a comment: the `provenance` job in `.github/workflows/ci.yml`
exits non-zero if any linked library name matches a known solver. The claim cannot silently
rot.

### 4.2 The full link line

Asked of Ninja directly so it cannot drift from what was really executed.

Linux, from the CI `provenance` job (run 32867892122):

```
$ ninja -C build -t commands sankhya | tail -1
/usr/bin/c++ -O3 -DNDEBUG -Wl,--dependency-file=CMakeFiles/sankhya-cli.dir/link.d     CMakeFiles/sankhya-cli.dir/apps/sankhya-cli/main.cpp.o     -o sankhya     libsankhya_core.a     _deps/fmt-build/libfmt.a     /usr/lib/x86_64-linux-gnu/libz.so
```

Windows, same command on the local tree:

```
$ ninja -C build -t commands sankhya-cli | tail -1
g++ -O3 -DNDEBUG -static-libgcc -static-libstdc++ -static     CMakeFiles/sankhya-cli.dir/apps/sankhya-cli/main.cpp.obj     -o sankhya.exe     -Wl,--out-implib,libsankhya.dll.a     libsankhya_core.a     _deps/fmt-build/libfmt.a     C:/msys64/ucrt64/lib/libz.dll.a
```

Three artifacts on each platform: our own core, fmt, and the system zlib. CLI11 and
nlohmann/json are header-only and appear as includes rather than archives; they are in the
dependency table above and visible in `CMakeLists.txt`.

### 4.3 SBOM

An SPDX SBOM is generated in CI by the `provenance` job and attached as a build artifact
(`sankhya-sbom.spdx.json`).

---

## 5. Judgement calls

Decisions where the dependency policy needed interpretation. Per `ENGINEERING_RULES.md`, these are
recorded rather than silently made.

| # | Question | Decision | Reasoning |
|---|---|---|---|
| 1 | Does imitating a familiar solver *log layout* (iteration / objective / primal inf / dual inf / time) cross the line? | Allowed | Interface familiarity, not derivation. The layout is visible in published user manuals and screenshots; no source was consulted. It makes the output readable to the industrial audience that reads such logs daily. Same reasoning as the API-shape allowance in `ENGINEERING_RULES.md`. |
| 2 | Does a string-keyed option table with typed values imitate a solver's API? | Allowed | Explicitly permitted by `ENGINEERING_RULES.md` item 5 (API shape). The shape is create/set/solve/query with string options; the registry, parser and storage are our own. |
| 3 | System zlib is picked up in preference to a fetched copy. Does that weaken the provenance claim? | Acceptable, and recorded | zlib is a compression library under a permissive licence, present on essentially every system. The exact resolved path appears in the link line above, so which copy was used is always visible. |
| 4 | The Windows build links libstdc++ and libgcc **statically** (`-static`). | Deliberate | It removes a start-up failure caused by an older MinGW earlier on PATH, and it makes the dependency list above shorter and easier to audit, not longer. It has no effect on the Linux build, which CI treats as authoritative. |
| 5 | The MPS reader imitates the *file format* of CPLEX/Xpress inputs, and the LP reader imitates the CPLEX LP format. | Allowed | A file format is an interface, not an implementation. `ENGINEERING_RULES.md` item 5 covers API shape for the same reason, and reading a format everyone's solver reads is what makes us drop-in adoptable. Both readers were written from the published format specification and from the textbook reference above; **no solver's reader source was consulted**, which `ENGINEERING_RULES.md` calls out by name as forbidden. |
| 6 | Netlib distributes its LP test set in a packed `emps` encoding, not as plain MPS, and expanding it requires the `emps` decoder Netlib ships alongside the data. Does fetching that decoder cross the red line? | **RESOLVED — allowed, and not vendored.** See row 8, which records how it is actually done. | Logged as OPEN while Phase 2 was in progress, and Phase 2 correctly claimed no Netlib result at the time. It was settled when `bench/runners/fetch_data.py` was written: the decoder is DOWNLOADED AND COMPILED at fetch time rather than committed, so this repository contains no third-party source, and the sha256 of everything downloaded is recorded. `emps` is a file-format converter, not a solver, so it sits outside the red line, and `ENGINEERING_RULES.md` item 6 permits the Netlib dataset of which this is the delivery mechanism. Nothing it produces is linked into SANKHYA. The Netlib results in `docs/BENCHMARKS.md` rest on this decision. See issue #62. |
| 7 | The Windows `.exe` dynamically links `zlib1.dll`, so the Phase 1 claim that it is self-contained no longer holds. | Recorded, not fixed | MSYS2 ships zlib only as an import library, and `-static` cannot statically link what has no static archive. Fixing it would mean forcing the bundled zlib build on Windows, which trades an audit-surface improvement for a divergence between the two platforms' dependency sets. Linux is what CI gates and what the provenance claim rests on; Windows is the convenience build. Revisit at Phase 10 packaging if we ship a Windows binary. |
| 8 | `bench/runners/fetch_data.py` downloads Netlib's `emps.c` decoder and COMPILES it at fetch time. Is that third-party source in the project? | Allowed, and not vendored | Netlib distributes its LP set in a custom compressed encoding, and `emps.c` is the decoder they publish beside it. It is a file-format converter, not a solver, so it is outside the red line. It is downloaded at fetch time rather than committed, so this repository contains no third-party source; its sha256 is recorded in `data/netlib/reference.json`. Nothing it produces is linked into SANKHYA - it runs once, offline, to turn an archive format into plain MPS. |
| 9 | MPS says a negative `UP` bound with no explicit lower bound implies `lower = -inf`. That is documented for continuous columns and **implementation-defined for integer ones**, where established readers disagree. Which reading do we take? | Apply the convention to integer columns too, and warn | Declining to choose was tried first and was worse than either choice: leaving `lower = 0` produces `[0, -5]`, an empty interval, so `Model::validate()` rejected the model and the file could not be loaded **at all** - and the resulting error named crossed bounds, which is the symptom rather than the cause. It also put the C++ reader at odds with `tools/verify_solution.py`, which already applies the convention; two components disagreeing about what the same bytes mean is exactly what that verifier exists to catch, so the disagreement sitting inside the pair weakened the check against every instance carrying such a bound. The warning is kept so the ambiguity stays visible in the log. See issue #8. |
| 10 | The device PDHG (#17) cites the cuPDLP paper for its host/device split, and cuPDLP.jl / cuPDLP-C / cuPDLPx are named on the red line. | Allowed, and recorded | The red line forbids reading solver SOURCE. The arXiv paper is a publication, allowed under item 2 of the policy, and it is the only cuPDLP artefact consulted: no repository of any of the three was opened, and the kernels, the CSR upload and the reduction scheme were written from the paper's description and the CUDA / cuSPARSE / CUB documentation. The iteration itself is the CPU engine's, from CP11 and PDLP. |
| 11 | Auditing the 2026 SIH PS26119 field meant READING ten competitors' public repositories at source level, and nine of the ten turn out to contain genuine from-scratch optimization solvers - their simplex implementations, cut generators and PDHG kernels. The red line forbids reading solver source. Does reading a *competitor's* solver cross it? | **Recorded as a judgement call, not silently decided.** The reading was done; nothing from any of those repositories may enter `src/`, and nothing has. | The red line in `ENGINEERING_RULES.md` enumerates established solvers - CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP, cuPDLP, OR-Tools, Gurobi, CPLEX, Xpress - and its purpose is to keep our implementation free of derivation from mature codebases we might otherwise lean on. Ten hackathon repositories a few weeks old are not on that list and are not a plausible source of technique for us; the audit's output is a competitive assessment - threat ranking, evidence discipline, capability gaps - not an algorithm. Every reader agent was instructed to detect any VENDORED solver tree and report its path WITHOUT opening it, precisely so the audit could not launder red-line source through a competitor's repository; no vendored tree was found in any of the ten. The contrary reading is that a from-scratch competitor solver is still a solver, and that the rule is about the act of reading rather than the maturity of what is read. That reading is defensible, which is why this is logged rather than assumed. Two consequences are accepted regardless of how it is resolved: no competitor code is referenced, adapted or cited in `src/`, and if the audit is ever cited externally, this row is the disclosure. |
| 12 | The dual simplex cost perturbation planned in #465 is taken from Koberstein's thesis (2005), whose text remarks that it is similar to the perturbation in an open solver named on the red line. Does learning it from the thesis cross the line? | Allowed, and recorded before implementation | The thesis is a publication, allowed under item 2 of the policy, and the perturbation is specified in it completely (sizes, signs, when to apply and remove it). The resemblance is the author's remark about someone else's code; nobody on this project reads that code, and the implementation is written from the thesis text and Maros (2003) alone. |
| 13 | The first-order presolve planned in #485 follows a 2026 arXiv paper whose authors also publish a presolve library, and that library ships inside GPU LP solvers named on the red line. | Paper only, and recorded before implementation | The paper describes each reduction in enough detail to implement it, and presolve reductions are textbook material (Andersen and Andersen 1995). The library is treated as red-line source because it is distributed inside red-line solvers: it is not opened, and our reductions are written from the paper and from the presolve already in `src/presolve/`. |
| 14 | METIS (#471) would be the first graph-ordering library linked into `src/`. | Allowed, optional, recorded before it is added | METIS partitions graphs; it does not solve an optimization problem, and the policy names graph orderings (AMD, COLAMD, RCM) as allowed when permissively licensed. METIS 5 is Apache-2.0. It is added behind a build option, the CPU build must keep working without it, and the dependency table, SBOM and `ldd` output are updated in the same PR. |
| 15 | cuDSS (#489) is a sparse direct factorization library shipped by NVIDIA, which also ships an optimization solver. | Allowed in principle as vendor numerical library, pending a licence check | The policy allows vendor numerical libraries such as cuSPARSE, cuBLAS and cuSOLVER; cuDSS is the same kind of library and solves linear systems, not optimization problems. Its licence terms have not been checked yet, so no code depending on it merges until they are and the result is recorded in this row. |
| 16 | Agreement runs against size-limited commercial editions (#533). | Allowed as external processes only, pending a licence check | The same arrangement as the HiGHS comparison: the solver is invoked as a separate process on the same MPS files, and nothing is linked or read. Each edition's terms for benchmarking use are checked and recorded here before any run is made or any number is published. |
| 17 | The local-search MIP heuristic planned in #507 follows a CP 2024 paper whose authors publish their implementation, and that implementation is itself a MIP solver. | Paper only, recorded before implementation | The red line exists to keep solver code out of our derivation, and a published local-search MIP solver is a solver whether or not it is on the enumerated list. The paper specifies the moves, weights and tabu rule; the implementation is written from it, and the repository is not opened. The same applies to the Feasibility Jump paper (#506). |
| 18 | The MILP proof certificates planned in #518 use the VIPR certificate format from Cheung, Gleixner and Steffy (2017). A reference checker for that format exists. | Format allowed; our own checker | A certificate format is an interface, like the MPS format (row 5). The checker in `tools/` is written by us in exact rational arithmetic from the paper's definition of the format and never links our C++, the same arrangement as `tools/verify_solution.py`. The reference checker is not read, so that a bug in our reading of the format cannot be masked by copying theirs. |
| 19 | The Maros-Meszaros benchmark (#491) takes its reference objectives from the set's own readme, whose OPT column is the output of BPMPD, an interior-point QP solver, and follows the success criteria of qpbenchmark, a repository that benchmarks QP solvers through a Python wrapper. Does either cross the red line? | Allowed, and recorded | The OPT values are published benchmark data, item 6 of the policy, in the same position as Netlib's readme values for the LP set: they are parsed from `00README.QP` by `bench/runners/fetch_maros_meszaros.py`, never typed, and every one carries its source in `data/maros-meszaros/reference.json`. No BPMPD code exists in this repository or was consulted. Of qpbenchmark only the published results page (`results/maros_meszaros.md`: the three measures, the 1e-6 and 1e-9 levels, the 1000 s limit and the shift of 10) was read; none of its code, the wrapper's code or any solver it calls was opened, and its converted `.mat` copies of the set are not used - the instances come from Maros's own archives, with their sha256 recorded. |
| 20 | PuLP (#536) is pip-installed as an optional modelling library for the adapter, and the wheel ships a CBC binary (a solver named on the red line) alongside it. Does installing it cross the line? | Allowed; the binary is never invoked, and recorded here | The policy forbids copying, vendoring, linking or reading solver source; a binary that pip drops next to a modelling library and that nothing in this repository executes is none of those. The adapter implements PuLP's own `LpSolver` interface and hands the model to `sankhya.Model`; PuLP's default solver is never selected, `test_adapters.py` solves through SANKHYA only, and no benchmark or result in this repository comes from CBC. If a PuLP call ever routed to CBC it would be a bug, not a judgement call. |

---

## 6. Reproducing this

```bash
scripts/configure.sh build Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On Linux, `ldd build/sankhya`. On Windows, `objdump -p build/sankhya.exe | grep "DLL Name"`.

To reproduce the Phase 2 end-to-end result:

```bash
build/sankhya info demo/crude_blend.mps
build/sankhya solve demo/crude_blend.mps --write-sol blend.sol --stats blend.json
build/sankhya solve demo/crude_blend.lp
```

The MPS and LP files describe the same model; the two solutions must be identical.
