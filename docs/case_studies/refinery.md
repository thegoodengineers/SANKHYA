# Case study: multi-period refinery planning

**SYNTHETIC DATA NOTICE.** Every numerical coefficient in the model files
produced by `bench/case_studies/refinery/generator.py` is synthetic.
Yields, qualities, costs, capacities and demand figures are pseudo-randomly
generated from a fixed seed.  They do not represent any real refinery, crude
oil stream, product specification or market price.  No real company data are
included.

---

## What a planner sees

A crude refinery buys several crude streams on the spot or contract market,
runs them through a crude distillation unit (CDU) and a set of downstream
processing units, and sells the resulting products (naphtha, middle distillate,
fuel oil, ...) subject to minimum-quality specifications and delivery
commitments.  The plan covers T periods (weeks or months); crude and product
inventories carry over from one period to the next, and unused production
capacity in one period cannot be banked.

The planner's decisions are:
- How much of each crude to purchase each period.
- How much of each crude to run through the CDU each period.
- How much of each product to sell, and how much to hold in tank.

The objective is to minimise total cost: crude purchase costs and inventory
holding charges, minus product revenue.

When the refinery must commit to a crude purchase in advance (for example,
to secure a tanker slot), the purchase decision becomes binary: either the
crude is ordered or it is not.  That variant is a mixed-integer program (MILP).

---

## Model structure

The model follows the multi-period refinery LP/MILP structure described in
Floudas and Lin (2005) and the staircase LP class analysed in Pochet and Wolsey
(2006).

### Variables

| Name | Meaning | Type |
|---|---|---|
| `BUY_{k}_{t}` | barrels of crude k purchased in period t | continuous, ≥ 0 |
| `RUN_{k}_{t}` | barrels of crude k run through the CDU in period t | continuous, ∈ [0, CDU\_cap] |
| `CS_{k}_{t}` | crude k stock at end of period t | continuous, ∈ [0, tank\_k] |
| `PROD_{j}_{t}` | barrels of product j made in period t | continuous, ≥ 0 |
| `SELL_{j}_{t}` | barrels of product j sold in period t | continuous, ∈ [0, demand\_{j,t}] |
| `PS_{j}_{t}` | product j stock at end of period t | continuous, ∈ [0, tank\_j] |
| `ZDEC_{k}_{t}` | 1 iff crude k is purchased in period t (**MILP only**) | binary |

### Constraints

| Name pattern | Meaning | Sense |
|---|---|---|
| `CBAL_{k}_{t}` | crude k inventory balance: CS = CS\_prev + BUY − RUN | equality |
| `CDU_{t}` | CDU throughput capacity: Σ\_k RUN\_k\_t ≤ cap | ≤ |
| `MAKE_{j}_{t}` | production from yields: PROD\_j\_t = Σ\_k Y\[j,k\] RUN\_k\_t | equality |
| `PBAL_{j}_{t}` | product j inventory balance: PS = PS\_prev + PROD − SELL | equality |
| `UNIT_{u}_{t}` | unit u capacity: Σ\_k H\[u,k\] RUN\_k\_t ≤ cap\_u\_t | ≤ |
| `SPEC_{s}_{t}` | quality budget for spec s: Σ\_k q\[s,k\] Y\[j\_s,k\] RUN\_k\_t ≤ budget | ≤ |
| `COMMIT_{j}_{t}` | delivery commitment: SELL\_j\_t ≥ commit\_{j,t} | ≥ |
| `ZLINK_{k}_{t}` | big-M linking (**MILP only**): BUY\_k\_t ≤ M · ZDEC\_k\_t | ≤ |

The quality constraint uses the *mass-budget* form rather than a
concentration cap, following the same motivation as
`bench/runners/generate_refinery_lp.py`: a concentration cap that is exactly
met at an integer plan is generally not a short decimal, and the LP optimum
must be expressible in exact decimal arithmetic for the KKT verification to
be exact.

The inventory balance rows create the *staircase* sparsity pattern
characteristic of multi-period planning models: period t's columns appear in
both period t's rows and period (t+1)'s balance rows, and nowhere else.  This
is the structure that a direct LU factorisation exploits (#206, #193).

### Objective

Minimise:
```
  Σ_{k,t}  c_buy[k]   · BUY_{k,t}
− Σ_{j,t}  p_sell[j]  · SELL_{j,t}
+ Σ_{k,t}  h_crude[k] · CS_{k,t}
+ Σ_{j,t}  h_prod[j]  · PS_{j,t}
```

All coefficients are synthetic decimals with small denominators, so the LP
optimum is exact when verified in rational arithmetic.

---

## Size ladder

| Size | Periods | Crudes | Products | Units | Specs | Commits | ~rows | ~cols |
|---|---|---|---|---|---|---|---|---|
| small  | 4  | 3  | 4  | 5  | 3  | 2 | ~120  | ~160  |
| medium | 12 | 8  | 8  | 8  | 6  | 4 | ~900  | ~1200 |
| large  | 52 | 15 | 12 | 10 | 10 | 8 | ~9000 | ~13000 |

Row and column counts are approximate; the exact numbers depend on the seed.

Generate all sizes (LP and MILP) with:

```bash
python bench/case_studies/refinery/run_case_study.py --generate-only
```

Solve with a built sankhya binary:

```bash
python bench/case_studies/refinery/run_case_study.py --sankhya build/sankhya \
    --out bench/case_studies/refinery/results.csv
```

---

## Results

**Placeholder — run the case study to fill this table.**

The table below will be populated by `run_case_study.py`.  Column meanings:
`solve_s` is wall-clock solve time in seconds; `status` is the solver status
(`optimal`, `infeasible`, etc.); `objective` is the reported optimal value.

| size | type | rows | cols | nonzeros | binary | status | objective | solve_s |
|---|---|---|---|---|---|---|---|---|
| small  | lp   | — | — | — | 0 | — | — | — |
| small  | milp | — | — | — | — | — | — | — |
| medium | lp   | — | — | — | 0 | — | — | — |
| medium | milp | — | — | — | — | — | — | — |
| large  | lp   | — | — | — | 0 | — | — | — |
| large  | milp | — | — | — | — | — | — | — |

---

## Interpretation

### Binding quality specifications

A `SPEC` row with a nonzero shadow price (dual multiplier) means the quality
budget for that product is tight: the blend is exactly at its specification
limit.  Relaxing the budget by one unit reduces cost by the shadow price.
In a real refinery this tells the blender whether there is value in sourcing a
sweeter crude or in upgrading the treating unit.

### Value of CDU capacity

The shadow price of `CDU_{t}` quantifies how much the optimal cost would
decrease if one additional barrel of throughput were available in period t.
A consistently high value across periods signals a capacity bottleneck; a
value of zero means the unit is not the limiting constraint in that period.

### Binary purchase decisions (MILP)

The MILP objective is always at least as large as the LP relaxation optimum
(since the LP is a relaxation).  The gap between the two indicates how much
the discrete purchase commitment costs the planner compared with the ideal
fractional schedule.  A small gap suggests the LP relaxation is a reliable
planning tool even when decisions are binary.

---

## References

- Floudas, C.A. and Lin, X. (2005). Mixed Integer Linear Programming in
  Process Scheduling: Modeling, Algorithms, and Applications. *Annals of
  Operations Research*, 139(1), 131–162.

- Pochet, Y. and Wolsey, L.A. (2006). *Production Planning by Mixed Integer
  Programming*. Springer Series in Operations Research and Financial
  Engineering.

See also `bench/runners/generate_refinery_lp.py` for the production-scale
refinery generator used in `docs/BENCHMARKS.md`, which follows the same
mathematical structure and carries a KKT-verified analytic optimum.
