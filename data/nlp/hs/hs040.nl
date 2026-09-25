g3 1 1 0	# problem hs040
 4 3 1 0 3 	# vars, constraints, objectives, ranges, eqns
 3 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 4 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 7 4 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o0	#+
o5	#^
v0	#x[1]
n3.0
o5	#^
v1	#x[2]
n2.0
C1	#c2_constr2
o2	#*
o5	#^
v0	#x[1]
n2.0
v2	#x[4]
C2	#c3_constr3
o5	#^
v2	#x[4]
n2.0
O0 0	#obj
o2	#*
o2	#*
o2	#*
o2	#*
n-1
v0	#x[1]
v1	#x[2]
v3	#x[3]
v2	#x[4]
x4	# initial guess
0 0.8	#x[1]
1 0.8	#x[2]
2 0.8	#x[4]
3 0.8	#x[3]
r	#3 ranges (rhs's)
4 1.0	#c1_constr1
4 0.0	#c2_constr2
4 0.0	#c3_constr3
b	#4 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[4]
3	#x[3]
k3	#intermediate Jacobian column lengths
2
4
6
J0 2	#c1_constr1
0 0
1 0
J1 3	#c2_constr2
0 0
2 0
3 -1
J2 2	#c3_constr3
1 -1
2 0
G0 4	#obj
0 0
1 0
2 0
3 0
