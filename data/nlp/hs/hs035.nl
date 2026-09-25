g3 1 1 0	# problem hs035
 3 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 3 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
O0 0	#obj
o0	#+
o54	# sumlist
5	# (n)
o2	#*
n2.0
o5	#^
v0	#x[1]
n2.0
o2	#*
n2.0
o5	#^
v1	#x[2]
n2.0
o5	#^
v2	#x[3]
n2.0
o2	#*
o2	#*
n2.0
v0	#x[1]
v1	#x[2]
o2	#*
o2	#*
n2.0
v0	#x[1]
v2	#x[3]
n9.0
x3	# initial guess
0 0.5	#x[1]
1 0.5	#x[2]
2 0.5	#x[3]
r	#1 ranges (rhs's)
1 3.0	#c1_constr1
b	#3 bounds (on variables)
2 0.0	#x[1]
2 0.0	#x[2]
2 0.0	#x[3]
k2	#intermediate Jacobian column lengths
1
2
J0 3	#c1_constr1
0 1
1 1
2 2.0
G0 3	#obj
0 -8.0
1 -6.0
2 -4.0
