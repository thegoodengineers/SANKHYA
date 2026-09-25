g3 1 1 0	# problem hs064
 3 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 3 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 3 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o54	# sumlist
3	# (n)
o3	# /
n4.0
v0	#x[1]
o3	# /
n32.0
v1	#x[2]
o3	# /
n120.0
v2	#x[3]
O0 0	#obj
o54	# sumlist
3	# (n)
o3	# /
n50000.0
v0	#x[1]
o3	# /
n72000.0
v1	#x[2]
o3	# /
n144000.0
v2	#x[3]
x3	# initial guess
0 1.0	#x[1]
1 1.0	#x[2]
2 1.0	#x[3]
r	#1 ranges (rhs's)
1 1.0	#c1_constr1
b	#3 bounds (on variables)
2 1e-05	#x[1]
2 1e-05	#x[2]
2 1e-05	#x[3]
k2	#intermediate Jacobian column lengths
1
2
J0 3	#c1_constr1
0 0
1 0
2 0
G0 3	#obj
0 5.0
1 20.0
2 10.0
