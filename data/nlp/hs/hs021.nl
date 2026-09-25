g3 1 1 0	# problem hs021
 2 3 1 2 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
C1	#c2_constr2
n0
C2	#c3_constr3
n0
O0 0	#obj
o0	#+
o0	#+
o2	#*
n0.01
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
n-100.0
x2	# initial guess
0 -1.0	#x[1]
1 -1.0	#x[2]
r	#3 ranges (rhs's)
2 10.0	#c1_constr1
0 2.0 50.0	#c2_constr2
0 -50.0 50.0	#c3_constr3
b	#2 bounds (on variables)
3	#x[1]
3	#x[2]
k1	#intermediate Jacobian column lengths
2
J0 2	#c1_constr1
0 10.0
1 -1
J1 1	#c2_constr2
0 1
J2 1	#c3_constr3
1 1
G0 2	#obj
0 0
1 0
