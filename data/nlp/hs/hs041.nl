g3 1 1 0	# problem hs041
 4 5 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 8 3 	# nonzeros in Jacobian, obj. gradient
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
O0 0	#obj
o0	#+
o16	#-
o2	#*
o2	#*
v0	#x[1]
v1	#x[2]
v2	#x[3]
n2.0
x4	# initial guess
0 2.0	#x[1]
1 2.0	#x[2]
2 2.0	#x[3]
3 2.0	#x[4]
r	#5 ranges (rhs's)
4 0.0	#c1_constr1
1 1.0	#c2_constr2
1 1.0	#c3_constr3
1 1.0	#c4_constr4
1 2.0	#c5_constr5
b	#4 bounds (on variables)
2 0.0	#x[1]
2 0.0	#x[2]
2 0.0	#x[3]
2 0.0	#x[4]
k3	#intermediate Jacobian column lengths
2
4
6
J0 4	#c1_constr1
0 1
1 2.0
2 2.0
3 -1
J1 1	#c2_constr2
0 1
J2 1	#c3_constr3
1 1
J3 1	#c4_constr4
2 1
J4 1	#c5_constr5
3 1
G0 3	#obj
0 0
1 0
2 0
