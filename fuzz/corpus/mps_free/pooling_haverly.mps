* SANKHYA demo instance - the Haverly pooling problem, which this solver does NOT read.
*
* Haverly, C.A. (1978). Studies of the behavior of recursion for the pooling problem.
* ACM SIGMAP Bulletin 25, 19-28. Case 1.
*
* Two crudes, A (3 % sulphur, 6 $/unit) and B (1 %, 16 $/unit), are mixed in one pool of
* unknown quality Q before being blended with a direct stream C (2 %, 10 $/unit) into two
* products: X sells at 9 with at most 2.5 % sulphur and demand 100, Y sells at 15 with at
* most 1.5 % and demand 200. The pool quality Q is a decision variable, and the sulphur
* carried from the pool into a product is Q times the flow - a BILINEAR term in a
* CONSTRAINT, which is what makes pooling non-convex (the published global optimum is a
* profit of 400; the local optimum of 100 that recursion finds from a poor start is the
* classic trap the paper is about).
*
* Columns: A, B (into the pool), PX, PY (pool to X, Y), CX, CY (C to X, Y), Q (pool % S).
*   BALANCE   A + B - PX - PY = 0
*   QUALITY   3 A + B - Q*PX - Q*PY = 0                                        (bilinear)
*   SPECX     Q*PX + 2 CX <= 2.5 (PX + CX)   ->   Q*PX - 0.5 PX - 0.5 CX <= 0   (bilinear)
*   SPECY     Q*PY + 2 CY <= 1.5 (PY + CY)   ->   Q*PY + 0.5 PY + 0.5 CY <= 0   (bilinear)
*   DEMX      PX + CX <= 100          DEMY      PY + CY <= 200
*
* The bilinear terms are written in QCMATRIX sections, the CPLEX/Gurobi QPS extension for
* quadratic constraints: one section per row, the full symmetric matrix listed, so a term
* Q*PX with coefficient c appears as the two entries (Q, PX) and (PX, Q) of c/2 each.
*
* SANKHYA reads a quadratic OBJECTIVE only. A QCMATRIX section is refused at read time,
* by name, with a message saying why - not skipped, which would silently solve a different
* (linear) model and report its optimum as this one. demo/run_sih_demo.sh section 3 shows
* the refusal. The data are from the paper; none of it is real MRPL data.
NAME          HAVERLY1
OBJSENSE
    MAXIMIZE
ROWS
 N  PROFIT
 E  BALANCE
 E  QUALITY
 L  SPECX
 L  SPECY
 L  DEMX
 L  DEMY
COLUMNS
    A         PROFIT      -6.00   BALANCE      1.00
    A         QUALITY      3.00
    B         PROFIT     -16.00   BALANCE      1.00
    B         QUALITY      1.00
    PX        PROFIT       9.00   BALANCE     -1.00
    PX        SPECX       -0.50   DEMX         1.00
    PY        PROFIT      15.00   BALANCE     -1.00
    PY        SPECY        0.50   DEMY         1.00
    CX        PROFIT      -1.00   SPECX       -0.50
    CX        DEMX         1.00
    CY        PROFIT       5.00   SPECY        0.50
    CY        DEMY         1.00
    Q         PROFIT       0.00
RHS
    RHS       DEMX       100.00   DEMY       200.00
BOUNDS
 LO BND       Q            1.00
 UP BND       Q            3.00
QCMATRIX   QUALITY
    Q         PX          -0.50
    PX        Q           -0.50
    Q         PY          -0.50
    PY        Q           -0.50
QCMATRIX   SPECX
    Q         PX           0.50
    PX        Q            0.50
QCMATRIX   SPECY
    Q         PY           0.50
    PY        Q            0.50
ENDATA
