g3 1 1 0	# problem hs011
 2 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 1 2 1 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o5	#^
v0	#x[1]
n2.0
O0 0	#obj
o0	#+
o0	#+
o5	#^
o0	#+
v0	#x[1]
n-5.0
n2.0
o5	#^
v1	#x[2]
n2.0
n-25.0
x2	# initial guess
0 4.9	#x[1]
1 0.1	#x[2]
r	#1 ranges (rhs's)
1 0	#c1_constr1
b	#2 bounds (on variables)
3	#x[1]
3	#x[2]
k1	#intermediate Jacobian column lengths
1
J0 2	#c1_constr1
0 0
1 -1
G0 2	#obj
0 0
1 0
