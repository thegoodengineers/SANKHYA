g3 1 1 0	# problem hs061
 3 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 2 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 3 2 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o2	#*
n-2.0
o5	#^
v0	#x[2]
n2.0
C1	#c2_constr2
o16	#-
o5	#^
v1	#x[3]
n2.0
O0 0	#obj
o54	# sumlist
3	# (n)
o2	#*
n4.0
o5	#^
v2	#x[1]
n2.0
o2	#*
n2.0
o5	#^
v0	#x[2]
n2.0
o2	#*
n2.0
o5	#^
v1	#x[3]
n2.0
x3	# initial guess
0 0.0	#x[2]
1 0.0	#x[3]
2 0.0	#x[1]
r	#2 ranges (rhs's)
4 7.0	#c1_constr1
4 11.0	#c2_constr2
b	#3 bounds (on variables)
3	#x[2]
3	#x[3]
3	#x[1]
k2	#intermediate Jacobian column lengths
1
2
J0 2	#c1_constr1
0 0
2 3.0
J1 2	#c2_constr2
1 0
2 4.0
G0 3	#obj
0 16.0
1 -24.0
2 -33.0
