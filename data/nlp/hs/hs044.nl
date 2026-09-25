g3 1 1 0	# problem hs044
 4 6 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 4 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 12 4 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
C1	#c2_constr2
n0
C2	#c3_constr3
n0
C3	#c4_constr4
n0
C4	#c5_constr5
n0
C5	#c6_constr6
n0
O0 0	#obj
o54	# sumlist
4	# (n)
o16	#-
o2	#*
v0	#x[1]
v2	#x[3]
o2	#*
v0	#x[1]
v3	#x[4]
o2	#*
v1	#x[2]
v2	#x[3]
o16	#-
o2	#*
v1	#x[2]
v3	#x[4]
x4	# initial guess
0 0.0	#x[1]
1 0.0	#x[2]
2 0.0	#x[3]
3 0.0	#x[4]
r	#6 ranges (rhs's)
1 8.0	#c1_constr1
1 12.0	#c2_constr2
1 12.0	#c3_constr3
1 8.0	#c4_constr4
1 8.0	#c5_constr5
1 5.0	#c6_constr6
b	#4 bounds (on variables)
2 0.0	#x[1]
2 0.0	#x[2]
2 0.0	#x[3]
2 0.0	#x[4]
k3	#intermediate Jacobian column lengths
3
6
9
J0 2	#c1_constr1
0 1
1 2.0
J1 2	#c2_constr2
0 4.0
1 1
J2 2	#c3_constr3
0 3.0
1 4.0
J3 2	#c4_constr4
2 2.0
3 1
J4 2	#c5_constr5
2 1
3 2.0
J5 2	#c6_constr6
2 1
3 1
G0 4	#obj
0 1
1 -1
2 -1
3 0
