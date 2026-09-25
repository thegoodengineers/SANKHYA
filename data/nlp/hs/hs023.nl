g3 1 1 0	# problem hs023
 2 5 1 0 0 	# vars, constraints, objectives, ranges, eqns
 4 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 2 2 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 10 2 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c2_constr2
o0	#+
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
C1	#c3_constr3
o0	#+
o2	#*
n9.0
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
C2	#c4_constr4
o5	#^
v0	#x[1]
n2.0
C3	#c5_constr5
o5	#^
v1	#x[2]
n2.0
C4	#c1_constr1
n0
O0 0	#obj
o0	#+
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
x2	# initial guess
0 3.0	#x[1]
1 1.0	#x[2]
r	#5 ranges (rhs's)
2 1.0	#c2_constr2
2 9.0	#c3_constr3
2 0.0	#c4_constr4
2 0.0	#c5_constr5
2 1.0	#c1_constr1
b	#2 bounds (on variables)
0 -50.0 50.0	#x[1]
0 -50.0 50.0	#x[2]
k1	#intermediate Jacobian column lengths
5
J0 2	#c2_constr2
0 0
1 0
J1 2	#c3_constr3
0 0
1 0
J2 2	#c4_constr4
0 0
1 -1
J3 2	#c5_constr5
0 -1
1 0
J4 2	#c1_constr1
0 1
1 1
G0 2	#obj
0 0
1 0
