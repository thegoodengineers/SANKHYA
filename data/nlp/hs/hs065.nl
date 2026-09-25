g3 1 1 0	# problem hs065
 3 4 1 3 0 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 3 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
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
C1	#c2_constr2
n0
C2	#c3_constr3
n0
C3	#c4_constr4
n0
O0 0	#obj
o54	# sumlist
3	# (n)
o5	#^
o0	#+
v0	#x[1]
o2	#*
n-1
v1	#x[2]
n2.0
o2	#*
n0.1111111111111111
o5	#^
o54	# sumlist
3	# (n)
v0	#x[1]
v1	#x[2]
n-10.0
n2.0
o5	#^
o0	#+
v2	#x[3]
n-5.0
n2.0
x3	# initial guess
0 -5.0	#x[1]
1 5.0	#x[2]
2 0.0	#x[3]
r	#4 ranges (rhs's)
1 48.0	#c1_constr1
0 -4.5 4.5	#c2_constr2
0 -4.5 4.5	#c3_constr3
0 -5.0 5.0	#c4_constr4
b	#3 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
k2	#intermediate Jacobian column lengths
2
4
J0 3	#c1_constr1
0 0
1 0
2 0
J1 1	#c2_constr2
0 1
J2 1	#c3_constr3
1 1
J3 1	#c4_constr4
2 1
G0 3	#obj
0 0
1 0
2 0
