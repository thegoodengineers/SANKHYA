#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The jury Q&A of docs/REPORT.md: 25 questions, each answer with the CSV or document section
it rests on. `{slot}` names are filled by tools/make_report.py from the numbers it computes
for the headline table, so an answer cannot carry a figure its CSV has lost. Kept apart from
the generator so each file stays readable."""
from __future__ import annotations

import re

QA = [
    ("Is this built on top of an existing solver?",
     "No. No source of any optimization solver is copied, linked or read, and a CI job dumps "
     "what the binary links against and fails if a solver library appears. Rival solvers are "
     "run only as separate processes, to compare against.",
     "`docs/PROVENANCE.md` sections 1 (the red line), 2 (dependency table) and 4.1 (what the "
     "binary links against)."),
    ("Which algorithms are implemented, and where do they come from?",
     "Revised primal and dual simplex, restarted PDHG on CPU and CUDA, an interior point "
     "method, presolve with postsolve, branch and cut, convex QP engines and a local NLP "
     "interior point. Each is written from the literature and carries its citation above "
     "the implementation.",
     "`docs/PROVENANCE.md` section 3 (algorithm citation table: algorithm, paper, file); "
     "`docs/PS26119_COVERAGE.md`, \"Algorithms the PS names\"."),
    ("How do I know an answer is right, not just plausible?",
     "Every solve writes a `.sol` file, and `tools/verify_solution.py`, a separate Python "
     "program that shares no code with the solver, re-reads the model and re-derives "
     "feasibility, dual feasibility, complementarity and the objective. On the full Netlib "
     "set it accepts {netlib_verified} of {netlib_n} answers.",
     "{netlib_csv} at `{netlib_commit}`; `docs/BENCHMARKS.md` section 1c."),
    ("The verifier is yours too. Why should I trust it?",
     "Three independent anchors. The Netlib answers are graded against Koch's optima, "
     "computed in exact rational arithmetic: {netlib_exact} of {netlib_n} within 1e-6. The "
     "same verifier judges the rivals' answers in the head-to-head run and rejected "
     "{h2h_netlib_rejected} of theirs on Netlib, each named with the violated condition. And "
     "HiGHS, as a separate process, agrees on the objective on {compare_agree} of "
     "{compare_n} medium-tier instances.",
     "{netlib_csv}; {h2h_netlib_csv} and `docs/BENCHMARKS.md` section 4a.1; {compare_csv} "
     "and section 4."),
    ("How fast is it against HiGHS on Netlib?",
     "Slower in the aggregate. Shifted geometric mean with a 10 s shift over {h2h_netlib_n} "
     "instances: SANKHYA {h2h_netlib_sankhya_sgm} s, HiGHS {h2h_netlib_highs_sgm} s. Instance "
     "by instance, SANKHYA is faster on {h2h_netlib_wins}, ties on {h2h_netlib_ties} (within "
     "10%, or both under 0.1 s) and is slower on {h2h_netlib_losses}. We do not headline a "
     "win count: most pairs are ties, because most Netlib instances finish in milliseconds.",
     "{h2h_netlib_csv} at `{h2h_netlib_commit}`; `docs/BENCHMARKS.md` section 4a.1."),
    ("And on larger LPs?",
     "On the Kennington set HiGHS is clearly ahead: SGM {h2h_kennington_sankhya_sgm} s "
     "against {h2h_kennington_highs_sgm} s, SANKHYA faster on {h2h_kennington_wins} of "
     "{h2h_kennington_compared} and slower on {h2h_kennington_losses}, a median ratio of "
     "{h2h_kennington_median}. On Mittelmann's LPs SANKHYA finishes {mittelmann_passed} of "
     "{mittelmann_n} inside the limit. Speed at size is the main open work.",
     "{h2h_kennington_csv}, section 4a.2; {mittelmann_csv}, section 1d of "
     "`docs/BENCHMARKS.md`."),
    ("How does the QP engine compare?",
     "On the Maros-Meszaros convex QPs in the head-to-head run SANKHYA reports optimal on "
     "{h2h_maros_sankhya_solved} of {h2h_maros_n}, at the published objective on "
     "{h2h_maros_sankhya_matched}, and the verifier accepts {h2h_maros_sankhya_counted} of "
     "those; HiGHS reports optimal on {h2h_maros_highs_solved}, at the published objective "
     "on {h2h_maros_highs_matched}, and the verifier accepts {h2h_maros_highs_counted}, the "
     "rest failing its 1e-7 dual feasibility or 1e-6 gap conditions by small margins (each "
     "named in section 4a.3), so the SGM of {h2h_maros_sankhya_sgm} s against "
     "{h2h_maros_highs_sgm} s rests on the verifier's tolerances as much as on speed. In "
     "SANKHYA's own run {maros_matched} of {maros_n} reach the published objective and "
     "{maros_rejected} `optimal` answers are rejected by the verifier.",
     "{h2h_maros_csv}, `docs/BENCHMARKS.md` section 4a.3; {maros_csv}, section 2b."),
    ("What does the GPU actually buy?",
     "On real models on an {gpu_gpu}, the CUDA first-order engine is {gpu_worst} to "
     "{gpu_best} the speed of the faster CPU arm of the same engine on the same host, over "
     "{gpu_cells} model-tolerance cells of which the card ends {gpu_limits} at the time limit "
     "(section 6 prints every cell). It is "
     "a speed-up of our own first-order method, measured against our own CPU code; it is not "
     "a claim to beat a simplex or barrier solver on the same model.",
     "{gpu_csv} at `{gpu_commit}`; `docs/BENCHMARKS.md` sections 1g and 1g.3."),
    ("Where does the GPU lose?",
     "On small models, where launch and transfer costs are not amortised; on root domain "
     "propagation and batched node bounds for MILP, which lose to the CPU and stay off; and "
     "on the largest refinery year, where the card is further along but {gpu_limits} of its "
     "{gpu_cells} runs still end at the time limit.",
     "{gpu_csv}; `docs/BENCHMARKS.md` sections 1g, 1g.5 and 1g.10; "
     "`docs/NEGATIVE-RESULTS.md` section 1, \"The GPU crossover as a single-run figure\"."),
    ("Does it reach a million variables?",
     "On generated models with an optimum exact by construction, on the CPU: "
     "{million_done} of {million_n} engine runs end optimal and verified ({million_list}). "
     "The runs that do not finish are in the same table with the reason. The GPU arms at "
     "this size have not been run.",
     "{million_csv} at `{million_commit}`; `docs/BENCHMARKS.md` section 1f.5."),
    ("Has it solved a real refinery model?",
     "No real industrial model has been made available to us. The refinery evidence is a "
     "generated multi-period planning model (crudes, tanks, yields, capacities, quality "
     "budgets, deliveries) whose optimum is known by construction, plus scenario sweeps on "
     "it. We say so wherever the number appears.",
     "`docs/case_studies/refinery.md`; `docs/BENCHMARKS.md` sections 1f.3 and 6."),
    ("How strong is the MILP side?",
     "Behind mature solvers, and reported that way. On the MIPLIB 2017 easy instances we "
     "run, {miplib_matched} of {miplib_n} reach the published optimum and {miplib_proved} "
     "also prove it. The search finds good incumbents more often than it closes the bound.",
     "{miplib_csv} at `{miplib_commit}`; `docs/BENCHMARKS.md` section 2."),
    ("What do I get when a model is infeasible?",
     "A Farkas certificate, written to the solution file and checked by the verifier: a "
     "verdict without a certificate is not counted as a pass. On Netlib's infeasible set "
     "{infeasible_passed} of {infeasible_n} verdicts carry a certificate the verifier "
     "accepts.",
     "{infeasible_csv} at `{infeasible_commit}`; `docs/BENCHMARKS.md` section 3a."),
    ("What is a \"safe bound\", and why does it matter to a planner?",
     "A lower bound on the true optimum computed from the reported duals with outward "
     "rounding (Neumaier and Shcherbina), so it holds despite floating-point error, and "
     "re-derived exactly by the verifier. Of {certified_optimal} optimal Netlib answers, "
     "{certified_tight} are certified within 1e-6 of the true optimum this way, "
     "{certified_loose} have a looser finite bound and {certified_none} have none, and the "
     "file then claims nothing.",
     "{certified_csv} at `{certified_commit}`; `docs/BENCHMARKS.md` section 1c.2."),
    ("Can the shadow prices be trusted for pricing decisions?",
     "With the `exact` option the duals, reduced costs and ranges are re-derived from the "
     "basis in rational arithmetic and each floating-point value is marked certified or "
     "corrected. Over {exact_done} of {exact_n} LPs where the derivation finished in its "
     "budget, {exact_corrected} of {exact_values} floating-point sensitivity values were "
     "corrected. Degenerate rows get a left and a right shadow price instead of one number.",
     "{exact_csv} at `{exact_commit}`; `docs/PROVENANCE.md` section 3, \"Certified "
     "sensitivity\"."),
    ("What happens on badly scaled models?",
     "On {stress_sankhya_n} scaled-Netlib and adversarial LPs SANKHYA is correct on "
     "{stress_sankhya_correct}, wrong on {stress_sankhya_wrong} and declines "
     "{stress_sankhya_failed}. HiGHS {stress_highs_version} on its default options is "
     "{stress_highs_correct} correct, {stress_highs_wrong} wrong, {stress_highs_failed} "
     "failed on the same files; its defaults drop the tiny coefficients these files carry, "
     "so this measures defaults on extreme scaling, not HiGHS in general.",
     "{stress_csv} at `{stress_commit}`; `docs/BENCHMARKS.md` section 5b."),
    ("What does the solver do when it cannot solve a model?",
     "It declines: the status is a limit, `numerical_error` or `feasible`, never `optimal`. "
     "The {stress_sankhya_failed} stress-set runs above and the Maros-Meszaros runs short of "
     "`optimal` ({maros_optimal} of {maros_n} are optimal) are named instance by instance.",
     "{stress_csv}; {maros_csv}; `docs/ARCHITECTURE.md` section 8 (resource limits and what "
     "each one means)."),
    ("Does it handle non-convex QP?",
     "No, and it refuses rather than approximates: convexity is decided by an LDL^T test "
     "before any arithmetic, and a non-convex Hessian is returned with the negative pivot as "
     "the reason. A local optimum is never reported as a global one.",
     "`docs/PS26119_COVERAGE.md`, \"Problem classes\"; `docs/BENCHMARKS.md` section 2d."),
    ("What about NLP and MINLP?",
     "A local interior point for NLP and branch and bound for MINLP proved convex. "
     "Hock-Schittkowski: {hs_matched} of {hs_n} at the published objective, {hs_verified} "
     "accepted by the checker. Convex MINLPLib: {minlp_matched} of {minlp_n}. The status "
     "says `locally_optimal` unless the model is proved convex.",
     "{hs_csv}; {minlp_csv}; `docs/BENCHMARKS.md` section 2e."),
    ("Can a planner run what-if scenarios?",
     "Yes: `sankhya scenarios` re-solves a model over a set of changes from a warm start, "
     "and each scenario's answer is verified; {scenarios_verified} of {scenarios_n} in the "
     "committed run. `tools/report.py` prints binding constraints and shadow prices by the "
     "model's own names.",
     "{scenarios_csv} at `{scenarios_commit}`; `README.md`, \"Use\"; `docs/FINALE.md`."),
    ("How are the benchmark numbers kept honest?",
     "Every run writes a CSV with the instance's sha256, both objectives, the gap, status, "
     "time, the git commit and the machine. `docs/BENCHMARKS.md` and this report are "
     "generated from those files by scripts, and CI fails if a CSV's commit is not on "
     "`main` or if either document is out of date.",
     "`docs/BENCHMARKS.md`, opening section; `bench/runners/check_result_stamps.py`; "
     "`tools/make_report.py --check`."),
    ("Can I reproduce the results myself?",
     "Yes. `scripts/reproduce.sh` takes a fresh clone through the build, the tests, the "
     "Netlib run with independent verification and the HiGHS comparison, offline, and CI "
     "runs the same script on a machine that has never built the project.",
     "`README.md`, \"Reproduce everything\"; `docs/ARCHITECTURE.md` section 7 "
     "(reproducibility, and what is actually promised)."),
    ("What did you try that did not work?",
     "Tree cut rounds, flow cover strengthening, OpenMP in the LP engines and the "
     "Forrest-Tomlin update were built, measured and demoted; the single-run GPU crossover "
     "figure was withdrawn; and each withdrawn claim is kept by name with the number that "
     "withdrew it.",
     "`docs/NEGATIVE-RESULTS.md` sections 1 to 4."),
    ("How would this be integrated into an existing planning system?",
     "Through a C API shaped like the established solvers' (create, set, solve, query), "
     "Python bindings, a CLI, and MPS and LP readers. No GUI is part of the deliverable.",
     "`include/sankhya/sankhya.h`; `README.md`, \"Use\"; `docs/PS26119_COVERAGE.md`, "
     "\"Expected solution\"."),
    ("How do MIQP, NLP or a new engine get added later?",
     "Every engine implements one entry point, `solve(const Model&, const Options&) -> "
     "Solution`, behind a dispatcher; a new engine registers there and gets the same "
     "quality audit, status guard and verifier as the existing ones.",
     "`docs/ADDING_AN_ENGINE.md`; `docs/ARCHITECTURE.md` sections 5 and 13."),
]
CITES = re.compile(r"\.csv|docs/|README\.md|\.py|\.h`")


