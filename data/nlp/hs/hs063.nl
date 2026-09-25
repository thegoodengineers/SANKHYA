g3 1 1 0	# problem hs063
 3 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 3 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c2_constr2
o54	# sumlist
3	# (n)
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
o5	#^
v2	#x[3]
n2.0
C1	#c1_constr1
n0
O0 0	#obj
o0	#+
o54	# sumlist
5	# (n)
o16	#-
o5	#^
v0	#x[1]
n2.0
o2	#*
n-2.0
o5	#^
v1	#x[2]
n2.0
o16	#-
o5	#^
v2	#x[3]
n2.0
o16	#-
o2	#*
v0	#x[1]
v1	#x[2]
o16	#-
o2	#*
v0	#x[1]
v2	#x[3]
n1000.0
x3	# initial guess
0 2.0	#x[1]
1 2.0	#x[2]
2 2.0	#x[3]
r	#2 ranges (rhs's)
4 25.0	#c2_constr2
4 56.0	#c1_constr1
b	#3 bounds (on variables)
2 0.0	#x[1]
2 0.0	#x[2]
2 0.0	#x[3]
k2	#intermediate Jacobian column lengths
2
4
J0 3	#c2_constr2
0 0
1 0
2 0
J1 3	#c1_constr1
0 8.0
1 14.0
2 7.0
G0 3	#obj
0 0
1 0
2 0
