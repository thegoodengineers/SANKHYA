g3 1 1 0	# problem hs037
 3 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
C1	#c2_constr2
n0
O0 0	#obj
o2	#*
o2	#*
o2	#*
n-1
v0	#x[1]
v1	#x[2]
v2	#x[3]
x3	# initial guess
0 10.0	#x[1]
1 10.0	#x[2]
2 10.0	#x[3]
r	#2 ranges (rhs's)
1 72.0	#c1_constr1
2 0.0	#c2_constr2
b	#3 bounds (on variables)
0 0.0 42.0	#x[1]
0 0.0 42.0	#x[2]
0 0.0 42.0	#x[3]
k2	#intermediate Jacobian column lengths
2
4
J0 3	#c1_constr1
0 1
1 2.0
2 2.0
J1 3	#c2_constr2
0 1
1 2.0
2 2.0
G0 3	#obj
0 0
1 0
2 0
