g3 1 1 0	# problem hs032
 3 2 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 1 3 1 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o16	#-
o5	#^
v0	#x[1]
n3.0
C1	#c2_constr2
n0
O0 0	#obj
o0	#+
o5	#^
o54	# sumlist
3	# (n)
v0	#x[1]
o2	#*
n3.0
v1	#x[2]
v2	#x[3]
n2.0
o2	#*
n4.0
o5	#^
o0	#+
v0	#x[1]
o2	#*
n-1
v1	#x[2]
n2.0
x3	# initial guess
0 0.1	#x[1]
1 0.7	#x[2]
2 0.2	#x[3]
r	#2 ranges (rhs's)
2 3.0	#c1_constr1
4 1.0	#c2_constr2
b	#3 bounds (on variables)
2 0.0	#x[1]
2 0.0	#x[2]
2 0.0	#x[3]
k2	#intermediate Jacobian column lengths
2
4
J0 3	#c1_constr1
0 0
1 6.0
2 4.0
J1 3	#c2_constr2
0 1
1 1
2 1
G0 3	#obj
0 0
1 0
2 0
