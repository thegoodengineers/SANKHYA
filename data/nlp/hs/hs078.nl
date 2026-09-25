g3 1 1 0	# problem hs078
 5 3 1 0 3 	# vars, constraints, objectives, ranges, eqns
 3 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 5 5 5 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 11 5 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o54	# sumlist
5	# (n)
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
o5	#^
v2	#x[3]
n2.0
o5	#^
v3	#x[4]
n2.0
o5	#^
v4	#x[5]
n2.0
C1	#c2_constr2
o0	#+
o2	#*
v1	#x[2]
v2	#x[3]
o16	#-
o2	#*
o2	#*
n5.0
v3	#x[4]
v4	#x[5]
C2	#c3_constr3
o0	#+
o5	#^
v0	#x[1]
n3.0
o5	#^
v1	#x[2]
n3.0
O0 0	#obj
o2	#*
o2	#*
o2	#*
o2	#*
v0	#x[1]
v1	#x[2]
v2	#x[3]
v3	#x[4]
v4	#x[5]
x5	# initial guess
0 -2.0	#x[1]
1 1.5	#x[2]
2 2.0	#x[3]
3 -1.0	#x[4]
4 -1.0	#x[5]
r	#3 ranges (rhs's)
4 10.0	#c1_constr1
4 0.0	#c2_constr2
4 -1.0	#c3_constr3
b	#5 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
3	#x[4]
3	#x[5]
k4	#intermediate Jacobian column lengths
2
5
7
9
J0 5	#c1_constr1
0 0
1 0
2 0
3 0
4 0
J1 4	#c2_constr2
1 0
2 0
3 0
4 0
J2 2	#c3_constr3
0 0
1 0
G0 5	#obj
0 0
1 0
2 0
3 0
4 0
