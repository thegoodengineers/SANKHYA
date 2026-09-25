g3 1 1 0	# problem hs062
 3 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 3 3 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
n0
O0 0	#obj
o2	#*
n-32.174
o54	# sumlist
3	# (n)
o2	#*
n255.0
o43	#log
o3	# /
o54	# sumlist
4	# (n)
v0	#x[1]
v1	#x[2]
v2	#x[3]
n0.03
o54	# sumlist
4	# (n)
o2	#*
n0.09
v0	#x[1]
v1	#x[2]
v2	#x[3]
n0.03
o2	#*
n280.0
o43	#log
o3	# /
o54	# sumlist
3	# (n)
v1	#x[2]
v2	#x[3]
n0.03
o54	# sumlist
3	# (n)
o2	#*
n0.07
v1	#x[2]
v2	#x[3]
n0.03
o2	#*
n290.0
o43	#log
o3	# /
o0	#+
v2	#x[3]
n0.03
o0	#+
o2	#*
n0.13
v2	#x[3]
n0.03
x3	# initial guess
0 0.7	#x[1]
1 0.2	#x[2]
2 0.1	#x[3]
r	#1 ranges (rhs's)
4 1.0	#c1_constr1
b	#3 bounds (on variables)
0 0.0 1.0	#x[1]
0 0.0 1.0	#x[2]
0 0.0 1.0	#x[3]
k2	#intermediate Jacobian column lengths
1
2
J0 3	#c1_constr1
0 1
1 1
2 1
G0 3	#obj
0 0
1 0
2 0
