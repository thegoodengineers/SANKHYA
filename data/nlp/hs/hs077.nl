g3 1 1 0	# problem hs077
 5 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 2 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 5 4 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 5 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o0	#+
o2	#*
o5	#^
v0	#x[1]
n2.0
v2	#x[4]
o41	#sin
o0	#+
v2	#x[4]
o2	#*
n-1
v3	#x[5]
C1	#c2_constr2
o2	#*
o5	#^
v1	#x[3]
n4.0
o5	#^
v2	#x[4]
n2.0
O0 0	#obj
o54	# sumlist
5	# (n)
o5	#^
o0	#+
v0	#x[1]
n-1.0
n2.0
o5	#^
o0	#+
v0	#x[1]
o2	#*
n-1
v4	#x[2]
n2.0
o5	#^
o0	#+
v1	#x[3]
n-1.0
n2.0
o5	#^
o0	#+
v2	#x[4]
n-1.0
n4.0
o5	#^
o0	#+
v3	#x[5]
n-1.0
n6.0
x5	# initial guess
0 2.0	#x[1]
1 2.0	#x[3]
2 2.0	#x[4]
3 2.0	#x[5]
4 2.0	#x[2]
r	#2 ranges (rhs's)
4 2.8284271247461903	#c1_constr1
4 9.414213562373096	#c2_constr2
b	#5 bounds (on variables)
3	#x[1]
3	#x[3]
3	#x[4]
3	#x[5]
3	#x[2]
k4	#intermediate Jacobian column lengths
1
2
4
5
J0 3	#c1_constr1
0 0
2 0
3 0
J1 3	#c2_constr2
1 0
2 0
4 1
G0 5	#obj
0 0
1 0
2 0
3 0
4 0
