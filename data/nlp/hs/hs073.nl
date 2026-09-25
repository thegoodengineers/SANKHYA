g3 1 1 0	# problem hs073
 4 3 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 12 4 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c2_constr2
o2	#*
n1.645
o39	#sqrt
o54	# sumlist
4	# (n)
o2	#*
n0.28
o5	#^
v0	#x[1]
n2.0
o2	#*
n0.19
o5	#^
v1	#x[2]
n2.0
o2	#*
n20.5
o5	#^
v2	#x[3]
n2.0
o2	#*
n0.62
o5	#^
v3	#x[4]
n2.0
C1	#c1_constr1
n0
C2	#c3_constr3
n0
O0 0	#obj
n0
x4	# initial guess
0 1.0	#x[1]
1 1.0	#x[2]
2 1.0	#x[3]
3 1.0	#x[4]
r	#3 ranges (rhs's)
1 -21.0	#c2_constr2
2 5.0	#c1_constr1
4 1.0	#c3_constr3
b	#4 bounds (on variables)
2 0.0	#x[1]
2 0.0	#x[2]
2 0.0	#x[3]
2 0.0	#x[4]
k3	#intermediate Jacobian column lengths
3
6
9
J0 4	#c2_constr2
0 -12.0
1 -11.9
2 -41.8
3 -52.1
J1 4	#c1_constr1
0 2.3
1 5.6
2 11.1
3 1.3
J2 4	#c3_constr3
0 1
1 1
2 1
3 1
G0 4	#obj
0 24.55
1 26.75
2 39.0
3 40.5
