g3 1 1 0	# problem hs017
 2 4 1 1 0 	# vars, constraints, objectives, ranges, eqns
 2 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 2 2 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 2 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o5	#^
v1	#x[2]
n2.0
C1	#c2_constr2
o5	#^
v0	#x[1]
n2.0
C2	#c3_constr3
n0
C3	#c4_constr4
n0
O0 0	#obj
o0	#+
o2	#*
n100.0
o5	#^
o0	#+
v1	#x[2]
o16	#-
o5	#^
v0	#x[1]
n2.0
n2.0
o5	#^
o0	#+
o2	#*
n-1
v0	#x[1]
n1.0
n2.0
x2	# initial guess
0 -2.0	#x[1]
1 1.0	#x[2]
r	#4 ranges (rhs's)
2 0.0	#c1_constr1
2 0.0	#c2_constr2
0 -0.5 0.5	#c3_constr3
1 1.0	#c4_constr4
b	#2 bounds (on variables)
3	#x[1]
3	#x[2]
k1	#intermediate Jacobian column lengths
3
J0 2	#c1_constr1
0 -1
1 0
J1 2	#c2_constr2
0 0
1 -1
J2 1	#c3_constr3
0 1
J3 1	#c4_constr4
1 1
G0 2	#obj
0 0
1 0
