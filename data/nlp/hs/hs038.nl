g3 1 1 0	# problem hs038
 4 0 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 4 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 4 	# nonzeros in Jacobian, obj. gradient
 3 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0	#obj
o54	# sumlist
6	# (n)
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
o2	#*
n90.0
o5	#^
o0	#+
v3	#x[4]
o16	#-
o5	#^
v2	#x[3]
n2.0
n2.0
o5	#^
o0	#+
o2	#*
n-1
v2	#x[3]
n1.0
n2.0
o2	#*
n10.1
o0	#+
o5	#^
o0	#+
v1	#x[2]
n-1.0
n2.0
o5	#^
o0	#+
v3	#x[4]
n-1.0
n2.0
o2	#*
o2	#*
n19.8
o0	#+
v1	#x[2]
n-1.0
o0	#+
v3	#x[4]
n-1.0
x4	# initial guess
0 -3.0	#x[1]
1 -1.0	#x[2]
2 -3.0	#x[3]
3 -1.0	#x[4]
r	#0 ranges (rhs's)
b	#4 bounds (on variables)
0 -10.0 10.0	#x[1]
0 -10.0 10.0	#x[2]
0 -10.0 10.0	#x[3]
0 -10.0 10.0	#x[4]
k3	#intermediate Jacobian column lengths
0
0
0
G0 4	#obj
0 0
1 0
2 0
3 0
