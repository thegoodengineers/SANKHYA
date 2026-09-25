g3 1 1 0	# problem hs027
 3 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 1 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o5	#^
v0	#x[3]
n2.0
O0 0	#obj
o0	#+
o2	#*
n0.01
o5	#^
o0	#+
v1	#x[1]
n-1.0
n2.0
o5	#^
o0	#+
v2	#x[2]
o16	#-
o5	#^
v1	#x[1]
n2.0
n2.0
x3	# initial guess
0 2.0	#x[3]
1 2.0	#x[1]
2 2.0	#x[2]
r	#1 ranges (rhs's)
4 -1.0	#c1_constr1
b	#3 bounds (on variables)
3	#x[3]
3	#x[1]
3	#x[2]
k2	#intermediate Jacobian column lengths
1
2
J0 2	#c1_constr1
0 0
1 1
G0 2	#obj
1 0
2 0
