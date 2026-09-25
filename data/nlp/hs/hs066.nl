g3 1 1 0	# problem hs066
 3 5 1 3 0 	# vars, constraints, objectives, ranges, eqns
 2 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 7 2 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o16	#-
o44	#exp
v0	#x[1]
C1	#c2_constr2
o16	#-
o44	#exp
v1	#x[2]
C2	#c3_constr3
n0
C3	#c4_constr4
n0
C4	#c5_constr5
n0
O0 0	#obj
n0
x3	# initial guess
0 0.0	#x[1]
1 1.05	#x[2]
2 2.9	#x[3]
r	#5 ranges (rhs's)
2 0.0	#c1_constr1
2 0.0	#c2_constr2
0 0.0 100.0	#c3_constr3
0 0.0 100.0	#c4_constr4
0 0.0 10.0	#c5_constr5
b	#3 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
k2	#intermediate Jacobian column lengths
2
5
J0 2	#c1_constr1
0 0
1 1
J1 2	#c2_constr2
1 0
2 1
J2 1	#c3_constr3
0 1
J3 1	#c4_constr4
1 1
J4 1	#c5_constr5
2 1
G0 2	#obj
0 -0.8
2 0.2
