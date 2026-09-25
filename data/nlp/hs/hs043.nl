g3 1 1 0	# problem hs043
 4 3 1 0 0 	# vars, constraints, objectives, ranges, eqns
 3 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 4 4 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 12 4 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o54	# sumlist
4	# (n)
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
C1	#c2_constr2
o54	# sumlist
4	# (n)
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
n2.0
o5	#^
v3	#x[4]
n2.0
C2	#c3_constr3
o54	# sumlist
3	# (n)
o2	#*
n2.0
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
o5	#^
v2	#x[3]
n2.0
O0 0	#obj
o54	# sumlist
4	# (n)
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
o2	#*
n2.0
o5	#^
v2	#x[3]
n2.0
o5	#^
v3	#x[4]
n2.0
x4	# initial guess
0 0.0	#x[1]
1 0.0	#x[2]
2 0.0	#x[3]
3 0.0	#x[4]
r	#3 ranges (rhs's)
1 8.0	#c1_constr1
1 10.0	#c2_constr2
1 5.0	#c3_constr3
b	#4 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
3	#x[4]
k3	#intermediate Jacobian column lengths
3
6
9
J0 4	#c1_constr1
0 1
1 -1
2 1
3 -1
J1 4	#c2_constr2
0 -1
1 0
2 0
3 -1
J2 4	#c3_constr3
0 2.0
1 -1
2 0
3 -1
G0 4	#obj
0 -5.0
1 -5.0
2 -21.0
3 7.0
