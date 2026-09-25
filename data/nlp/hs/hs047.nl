g3 1 1 0	# problem hs047
 5 3 1 0 3 	# vars, constraints, objectives, ranges, eqns
 3 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 5 4 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 8 5 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o0	#+
o5	#^
v1	#x[2]
n2.0
o5	#^
v2	#x[3]
n3.0
C1	#c2_constr2
o16	#-
o5	#^
v2	#x[3]
n2.0
C2	#c3_constr3
o2	#*
v0	#x[1]
v3	#x[5]
O0 0	#obj
o54	# sumlist
4	# (n)
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
n3.0
o5	#^
o0	#+
v2	#x[3]
o2	#*
n-1
v4	#x[4]
n4.0
o5	#^
o0	#+
v4	#x[4]
o2	#*
n-1
v3	#x[5]
n4.0
x5	# initial guess
0 2.0	#x[1]
1 1.4142135623730951	#x[2]
2 -1.0	#x[3]
3 0.5	#x[5]
4 0.5857864376269049	#x[4]
r	#3 ranges (rhs's)
4 3.0	#c1_constr1
4 1.0	#c2_constr2
4 1.0	#c3_constr3
b	#5 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
3	#x[5]
3	#x[4]
k4	#intermediate Jacobian column lengths
2
4
6
7
J0 3	#c1_constr1
0 1
1 0
2 0
J1 3	#c2_constr2
1 1
2 0
4 1
J2 2	#c3_constr3
0 0
3 0
G0 5	#obj
0 0
1 0
2 0
3 0
4 0
