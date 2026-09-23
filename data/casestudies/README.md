# SANKHYA case studies — sources

`data/casestudies/generate.py` documents the algebra of each model in its own header
comments (visible at the top of the committed `.mps` files). This file adds what those
headers don't: the published formulation each model is drawn from, and what is preserved
versus simplified for this repository.

**All numeric parameters below (costs, capacities, demands, quality specs) are invented for
this repository — a small, hand-checkable instance in the shape of the cited formulation,
not data transcribed from the source.** None of it is real MRPL data. Where a source paper's
own numbers would have been usable directly, that is noted; none currently are, because the
published examples are all larger than is useful for an instance a judge can read end to end.

---

## Power system dispatch — `power_dispatch.mps`

### Source

Garver, L.L. (1962). "Power generation scheduling by integer programming — development of
theory." *IEEE Transactions on Power Apparatus and Systems*, PAS-81(3), 730–735.

Garver's paper is the origin of formulating unit commitment as an integer program: each
generating unit gets a binary on/off variable, coupled to its power output through a
minimum-stable-generation constraint, with a start-up cost charged only when a unit turns
on. `power_dispatch.mps` keeps exactly that structure — the DEMAND, RESERVE, and per-unit
CAPMX/CAPMN constraints in `generate.py` are Garver's coupling constraints under different
names. What it simplifies: a single dispatch period rather than a multi-period horizon, and
no ramp-rate or minimum-up/-down-time constraints, both of which are standard extensions in
modern unit commitment literature but are not part of Garver's original 1962 formulation
either.

---

## Supply chain distribution — `supply_chain.mps` (and `ill_conditioned.mps`, its rescaled variant)

### Source

Dantzig, G.B. (1963). *Linear Programming and Extensions.* Princeton University Press.

This is the classical balanced transportation problem in its standard textbook form —
supply and demand equality rows, a linear per-unit freight cost, total supply equal to
total demand by construction. There is essentially nothing to simplify: this *is* the
formulation, chosen deliberately because balanced supply/demand makes the constraint matrix
structurally rank-deficient (rank 6 on 7 rows), which is the numerical property PS26119
asks about (`generate.py`'s own header explains the degeneracy in detail). `ill_conditioned.mps`
is not a separate model — it is `supply_chain.mps` under an exact, invertible row/column
rescaling, generated so the ill-conditioning demo has a known-correct answer to check
against rather than an unverifiable one.

---

## Production planning — `lot_sizing.mps` (and `lot_sizing_relaxed.mps`, its LP relaxation)

### Source

Pochet, Y., Wolsey, L.A. (2006). *Production Planning by Mixed Integer Programming.*
Springer.

The single-item, multi-period lot-sizing problem with a per-period fixed setup cost,
per-unit production cost, and per-unit holding cost is the base model this entire book
builds from (their uncapacitated lot-sizing chapter). `lot_sizing.mps` keeps that exact
structure: the BAL inventory-balance rows and the big-M LINK constraints in `generate.py`
are the standard big-M setup/production coupling the book uses to introduce why that
formulation has a weak LP bound. What it simplifies: a single item and a single facility,
with no capacity limit beyond the big-M link itself — Pochet & Wolsey's later chapters
extend to multi-item, capacitated and multi-echelon variants that this instance does not
attempt.

---

## Crude blending — `demo/crude_blend.mps`

### Source

Rigby, B., Lasdon, L.S., Waren, A.D. (1995). "The evolution of Texaco's blending systems."
*Interfaces*, 25(5), 64–83.

This paper describes Texaco's production blending optimization systems, which combine
multiple crude/component streams into finished pools subject to quality specifications
(the paper's systems handle properties like octane, vapor pressure and sulphur content
much as `crude_blend.mps` handles sulphur and viscosity here) and report the shadow price
of each binding specification back to the planner — exactly the role the row duals play in
this repository's demo. What it simplifies: three crude streams into a single diesel pool
with one throughput window, one commitment and one quality spec, versus Texaco's
multi-pool, multi-period, refinery-wide system. This instance is a single snapshot of the
same kind of decision, not a scaled-down version of their actual model.

---

## Crude blending with price impact — `demo/crude_blend_qp.mps`

### Source

Beale, E.M.L. (1959). "On quadratic programming." *Naval Research Logistics Quarterly*,
6(3), 227–243.

The blending structure is inherited unchanged from `demo/crude_blend.mps` above (Rigby,
Lasdon & Waren) — same three crudes, same throughput window, same diesel commitment, same
sulphur specification. What is added is the term that makes it a QP, and Beale is cited for
the problem class rather than for this instance: a convex quadratic objective over linear
constraints, which is what his paper is about.

The quadratic term itself is the standard convexification of a purchasing decision and is
**not** taken from a specific published instance. Modelling the marginal delivered price of
a crude as rising linearly in the volume lifted, `price(x) = base + p·x`, makes total spend
the integral rather than the product — `base·x + 0.5·p·x²` — so the margin loses `0.5·p·x²`
per crude. In the QPS convention (`c'x + 0.5 x'Qx`, lower triangle only) that is exactly
`Q_jj = -p_j` with the linear margins unchanged, which is why the file carries no extra
factor. `Q` is negative definite, so under `OBJSENSE MAXIMIZE` the objective is strictly
concave and the QP is convex with a unique optimum.

What it simplifies versus a real purchasing model: price impact is per-crude and
independent, with no cross-terms between grades competing in the same market, and no
term structure — a single snapshot rather than a forward curve. The off-diagonal machinery
those cross-terms would need is exercised in the reader's unit tests instead
(`tests/unit/test_mps_reader.cpp`), not here.

**The price-impact slopes (0.012, 0.020, 0.016 $/bbl per kbbl/day) are invented for this
repository**, chosen so the quadratic term is large enough to move the answer visibly and
small enough to leave the problem hand-checkable. As with every other model in this
directory, none of it is real MRPL data.

---

## Crude blending with a volume discount — `demo/crude_blend_discount_qp.mps` (refused)

The price-impact instance above with one sign changed: Bonny Light comes with a volume
discount, its delivered price falling by 0.020 $/bbl for every kbbl/day lifted, so the
spend on it is `base·x − 0.5·0.020·x²` and the margin *gains* `0.5·0.020·BN²`. In the QPS
convention that is `Q_BN,BN = +0.020`, and under `OBJSENSE MAXIMIZE` the matrix the
convexity test looks at, `−Q = diag(0.012, −0.020, 0.016)`, has a negative eigenvalue. The
objective is neither concave nor convex, the problem has local optima at corners of the
feasible set, and nothing in a printed answer distinguishes one of them from the global
optimum.

SANKHYA refuses the file (`model_error`, exit code 5) with the pivot that proves it, rather
than reporting whichever local point an iteration reached as optimal. It is in this
directory to make that stated position demonstrable: `demo/run_sih_demo.sh` section 3 runs
it. The slope is invented, like the rest of the instance.

---

## The Haverly pooling problem — `demo/pooling_haverly.mps` (not read)

### Source

Haverly, C.A. (1978). "Studies of the behavior of recursion for the pooling problem."
*ACM SIGMAP Bulletin*, 25, 19–28. Case 1.

Two crudes, A (3 % sulphur, $6) and B (1 %, $16), are mixed in one pool of unknown quality
`Q` and then blended with a direct stream C (2 %, $10) into two products: X sells at $9 with
at most 2.5 % sulphur and demand 100, Y at $15 with at most 1.5 % and demand 200. The pool
quality is a decision variable, and the sulphur it carries into a product is `Q` times the
flow — a bilinear term in a constraint, which is what makes pooling non-convex. The
published global optimum is a profit of 400; the local optimum of 100 that recursion
finds from a poor start is the classic trap the paper is about.

The bilinear terms are written in `QCMATRIX` sections, the CPLEX/Gurobi QPS extension for
quadratic constraints, one per row with the full symmetric matrix listed. SANKHYA reads a
quadratic *objective* only: the file is refused at read time, by name, with a message
saying why — not skipped, which would solve the linear remainder and report it as this
model. `demo/run_sih_demo.sh` section 3 shows the refusal. The data are the paper’s; the
file is a transcription, not real MRPL data.

---

## Refinery scheduling — not currently represented

### Source

Pinto, J.M., Joly, M., Moro, L.F.L. (2000). "Planning and scheduling models for refinery
operations." *Computers & Chemical Engineering*, 24(9–10), 2259–2276.

This describes refinery production scheduling (as distinct from the blending decision
`crude_blend.mps` covers). There is no case study in this repository that implements that
formulation — `data/casestudies/` and `demo/` currently contain exactly the five models
above, not a sixth. Recorded here rather than attached to an unrelated `.mps` file, in case
a scheduling case study is added later.
