g3 1 1 0	# problem hs006
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 1 1 1 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 1 	# nonzeros in Jacobian, obj. gradient
 9 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr
o2	#*
n10.0
o16	#-
o5	#^
v0	#x[1]
n2.0
O0 0	#obj
o5	#^
o0	#+
o2	#*
n-1
v0	#x[1]
n1.0
n2.0
x2	# initial guess
0 -1.2	#x[1]
1 1.0	#x[2]
r	#1 ranges (rhs's)
4 0.0	#c1_constr
b	#2 bounds (on variables)
3	#x[1]
3	#x[2]
k1	#intermediate Jacobian column lengths
1
J0 2	#c1_constr
0 0
1 10.0
G0 1	#obj
0 0
