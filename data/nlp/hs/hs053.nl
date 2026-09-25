g3 1 1 0	# problem hs053
 5 3 1 0 3 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 5 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 7 5 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
C1	#c2_constr2
n0
C2	#c3_constr3
n0
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
o54	# sumlist
3	# (n)
v1	#x[2]
v2	#x[3]
n-2.0
n2.0
o5	#^
o0	#+
v3	#x[4]
n-1.0
n2.0
o5	#^
o0	#+
v4	#x[5]
n-1.0
n2.0
x5	# initial guess
0 2.0	#x[1]
1 2.0	#x[2]
2 2.0	#x[3]
3 2.0	#x[4]
4 2.0	#x[5]
r	#3 ranges (rhs's)
4 0.0	#c1_constr1
4 0.0	#c2_constr2
4 0.0	#c3_constr3
b	#5 bounds (on variables)
0 -10.0 10.0	#x[1]
0 -10.0 10.0	#x[2]
0 -10.0 10.0	#x[3]
0 -10.0 10.0	#x[4]
0 -10.0 10.0	#x[5]
k4	#intermediate Jacobian column lengths
1
3
4
5
J0 2	#c1_constr1
0 1
1 3.0
J1 3	#c2_constr2
2 1
3 1
4 -2.0
J2 2	#c3_constr3
1 1
4 -1
G0 5	#obj
0 0
1 0
2 0
3 0
4 0
