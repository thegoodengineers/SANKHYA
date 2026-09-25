g3 1 1 0	# problem hs026
 3 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 3 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 3 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o0	#+
o2	#*
o0	#+
o5	#^
v1	#x[2]
n2.0
n1.0
v0	#x[1]
o5	#^
v2	#x[3]
n4.0
O0 0	#obj
o0	#+
o5	#^
o0	#+
v0	#x[1]
o2	#*
n-1
v1	#x[2]
n2.0
o5	#^
o0	#+
v1	#x[2]
o2	#*
n-1
v2	#x[3]
n4.0
x3	# initial guess
0 -2.6	#x[1]
1 2.0	#x[2]
2 2.0	#x[3]
r	#1 ranges (rhs's)
4 3.0	#c1_constr1
b	#3 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
k2	#intermediate Jacobian column lengths
1
2
J0 3	#c1_constr1
0 0
1 0
2 0
G0 3	#obj
0 0
1 0
2 0
