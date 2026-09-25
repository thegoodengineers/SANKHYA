g3 1 1 0	# problem hs039
 4 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 2 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 1 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o0	#+
o16	#-
o5	#^
v0	#x[1]
n3.0
o16	#-
o5	#^
v1	#x[3]
n2.0
C1	#c2_constr2
o0	#+
o5	#^
v0	#x[1]
n2.0
o16	#-
o5	#^
v2	#x[4]
n2.0
O0 0	#obj
n0
x4	# initial guess
0 2.0	#x[1]
1 2.0	#x[3]
2 2.0	#x[4]
3 2.0	#x[2]
r	#2 ranges (rhs's)
4 0.0	#c1_constr1
4 0.0	#c2_constr2
b	#4 bounds (on variables)
3	#x[1]
3	#x[3]
3	#x[4]
3	#x[2]
k3	#intermediate Jacobian column lengths
2
3
4
J0 3	#c1_constr1
0 0
1 0
3 1
J1 3	#c2_constr2
0 0
2 0
3 -1
G0 1	#obj
0 -1
