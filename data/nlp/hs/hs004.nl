g3 1 1 0	# problem hs004
 2 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 1 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
C1	#c2_constr2
n0
O0 0	#obj
o2	#*
n0.3333333333333333
o5	#^
o0	#+
v0	#x[1]
n1.0
n3.0
x2	# initial guess
0 1.125	#x[1]
1 0.125	#x[2]
r	#2 ranges (rhs's)
2 1.0	#c1_constr1
2 0.0	#c2_constr2
b	#2 bounds (on variables)
3	#x[1]
3	#x[2]
k1	#intermediate Jacobian column lengths
1
J0 1	#c1_constr1
0 1
J1 1	#c2_constr2
1 1
G0 2	#obj
0 0
1 1
