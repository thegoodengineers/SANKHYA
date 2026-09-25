g3 1 1 0	# problem hs076
 4 3 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 4 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 10 4 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
C1	#c2_constr2
n0
C2	#c3_constr3
n0
O0 0	#obj
o54	# sumlist
6	# (n)
o5	#^
v0	#x[1]
n2.0
o2	#*
n0.5
o5	#^
v1	#x[2]
n2.0
o5	#^
v2	#x[3]
n2.0
o2	#*
n0.5
o5	#^
v3	#x[4]
n2.0
o16	#-
o2	#*
v0	#x[1]
v2	#x[3]
o2	#*
v2	#x[3]
v3	#x[4]
x4	# initial guess
0 0.5	#x[1]
1 0.5	#x[2]
2 0.5	#x[3]
3 0.5	#x[4]
r	#3 ranges (rhs's)
1 5.0	#c1_constr1
1 4.0	#c2_constr2
2 1.5	#c3_constr3
b	#4 bounds (on variables)
2 0.0	#x[1]
2 0.0	#x[2]
2 0.0	#x[3]
2 0.0	#x[4]
k3	#intermediate Jacobian column lengths
2
5
8
J0 4	#c1_constr1
0 1
1 2.0
2 1
3 1
J1 4	#c2_constr2
0 3.0
1 1
2 2.0
3 -1
J2 2	#c3_constr3
1 1
2 4.0
G0 4	#obj
0 -1
1 -3.0
2 1
3 -1
