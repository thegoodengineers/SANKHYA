g3 0 1 0	# problem nvs03
 2 2 1 0 0	# vars, constraints, objectives, ranges, eqns
 1 1	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 1 2 1	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 1 0 1	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2	# nonzeros in Jacobian, gradients
 3 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
b
0 0 200
0 0 200
x2
0 100
1 100
r
2 0
2 -4.5
C0
o2
n-.1
o5
v0
n2
C1
n0
O0 0
o0
o5
o0
n-8
v0
n2
o5
o0
n-2
v1
n2
k1
2
J0 2
0 0
1 1
J1 2
0 -.333333333333333
1 -1
G0 2
0 0
1 0
