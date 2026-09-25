g3 1 1 0	# problem hs071
 4 2 1 0 1 	# vars, constraints, objectives, ranges, eqns
 2 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 4 4 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 8 4 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o2	#*
o2	#*
o2	#*
v0	#x[1]
v1	#x[2]
v2	#x[3]
v3	#x[4]
C1	#c2_constr2
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
O0 0	#obj
o2	#*
o2	#*
v0	#x[1]
v3	#x[4]
o54	# sumlist
3	# (n)
v0	#x[1]
v1	#x[2]
v2	#x[3]
x4	# initial guess
0 1.0	#x[1]
1 5.0	#x[2]
2 5.0	#x[3]
3 1.0	#x[4]
r	#2 ranges (rhs's)
2 25.0	#c1_constr1
4 40.0	#c2_constr2
b	#4 bounds (on variables)
0 1.0 5.0	#x[1]
0 1.0 5.0	#x[2]
0 1.0 5.0	#x[3]
0 1.0 5.0	#x[4]
k3	#intermediate Jacobian column lengths
2
4
6
J0 4	#c1_constr1
0 0
1 0
2 0
3 0
J1 4	#c2_constr2
0 0
1 0
2 0
3 0
G0 4	#obj
0 0
1 0
2 1
3 0
