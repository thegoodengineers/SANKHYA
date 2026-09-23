* SANKHYA demo instance - crude blending with a VOLUME DISCOUNT, which is NOT a convex QP.
*
* The same file as demo/crude_blend_qp.mps with one sign changed. There every price rises
* with the volume lifted, so the margin loses 0.5 * p_j * x_j^2 per crude and the QP is
* convex. Here Bonny Light comes with a volume discount instead: its delivered price FALLS
* by 0.020 $/bbl for every kbbl/day taken, the spend on it is base * x - 0.5 * 0.020 * x^2,
* and the margin GAINS 0.5 * 0.020 * BN^2. In the QPS convention that is Q_BN,BN = +0.020.
*
* Under OBJSENSE MAXIMIZE the convexity test looks at -Q = diag(0.012, -0.020, 0.016): one
* negative eigenvalue, so the objective is neither concave nor convex. The margin curves
* UPWARD along BN, a maximiser wants BN at a bound, and the problem has local optima at
* corners of the feasible set that a first-order or interior-point iteration can stop at.
* Nothing distinguishes such a point from the global optimum once it is printed.
*
* SANKHYA therefore REFUSES this file (status model_error, exit code 5) with the pivot that
* proves it, rather than reporting whichever local point an iteration landed on as optimal.
* demo/run_sih_demo.sh section 3 shows the refusal; src/qp/convexity.cpp is the test.
*
* Everything else is inherited unchanged from crude_blend_qp.mps. All numeric parameters
* are invented for this repository; none of it is real MRPL data.
NAME          CRUDEBLENDDISC
OBJSENSE
    MAXIMIZE
ROWS
 N  MARGIN
 G  THRUPUT
 E  DIESEL
 L  SULPHUR
COLUMNS
    AL        MARGIN       2.40   THRUPUT      1.00
    AL        DIESEL       0.30   SULPHUR      0.80
    BN        MARGIN       1.60   THRUPUT      1.00
    BN        DIESEL       0.45   SULPHUR     -0.86
    MU        MARGIN       1.64   THRUPUT      1.00
    MU        DIESEL       0.38   SULPHUR     -0.22
RHS
    RHS       THRUPUT     90.00   DIESEL      40.00
    RHS       SULPHUR      0.00
RANGES
    RNG       THRUPUT     30.00
BOUNDS
 LO BND       AL          10.00
 UP BND       BN          45.00
 UP BND       MU          60.00
QUADOBJ
    AL        AL          -0.012
    BN        BN           0.020
    MU        MU          -0.016
ENDATA
