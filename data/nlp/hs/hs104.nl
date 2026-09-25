g3 1 1 0	# problem hs104
 8 6 1 0 0 	# vars, constraints, objectives, ranges, eqns
 6 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 8 4 4 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 21 4 	# nonzeros in Jacobian, obj. gradient
 5 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_c1
o16	#-
o2	#*
o2	#*
n0.0588
v6	#x[5]
v2	#x[7]
C1	#c2_c2
o16	#-
o2	#*
o2	#*
n0.0588
v7	#x[6]
v3	#x[8]
C2	#c3_c3
o54	# sumlist
3	# (n)
o16	#-
o3	# /
o2	#*
n4.0
v4	#x[3]
v6	#x[5]
o16	#-
o3	# /
n2.0
o2	#*
o5	#^
v4	#x[3]
n0.71
v6	#x[5]
o16	#-
o3	# /
o2	#*
n0.0588
v2	#x[7]
o5	#^
v4	#x[3]
n1.3
C3	#c4_c4
o54	# sumlist
3	# (n)
o16	#-
o3	# /
o2	#*
n4.0
v5	#x[4]
v7	#x[6]
o16	#-
o3	# /
n2.0
o2	#*
o5	#^
v5	#x[4]
n0.71
v7	#x[6]
o16	#-
o3	# /
o2	#*
n0.0588
v3	#x[8]
o5	#^
v5	#x[4]
n1.3
C4	#c5_c5
o0	#+
o2	#*
o2	#*
n0.4
o5	#^
v0	#x[1]
n0.67
o5	#^
v2	#x[7]
n-0.67
o2	#*
o2	#*
n0.4
o5	#^
v1	#x[2]
n0.67
o5	#^
v3	#x[8]
n-0.67
C5	#c6_c6
o0	#+
o2	#*
o2	#*
n0.4
o5	#^
v0	#x[1]
n0.67
o5	#^
v2	#x[7]
n-0.67
o2	#*
o2	#*
n0.4
o5	#^
v1	#x[2]
n0.67
o5	#^
v3	#x[8]
n-0.67
O0 0	#obj
o0	#+
o0	#+
o2	#*
o2	#*
n0.4
o5	#^
v0	#x[1]
n0.67
o5	#^
v2	#x[7]
n-0.67
o2	#*
o2	#*
n0.4
o5	#^
v1	#x[2]
n0.67
o5	#^
v3	#x[8]
n-0.67
n10.0
x8	# initial guess
0 6.0	#x[1]
1 3.0	#x[2]
2 1.0	#x[7]
3 0.5	#x[8]
4 0.4	#x[3]
5 0.2	#x[4]
6 6.0	#x[5]
7 6.0	#x[6]
r	#6 ranges (rhs's)
2 -1.0	#c1_c1
2 -1.0	#c2_c2
2 -1.0	#c3_c3
2 -1.0	#c4_c4
2 -9.9	#c5_c5
1 -5.8	#c6_c6
b	#8 bounds (on variables)
0 0.1 10.0	#x[1]
0 0.1 10.0	#x[2]
0 0.1 10.0	#x[7]
0 0.1 10.0	#x[8]
0 0.1 10.0	#x[3]
0 0.1 10.0	#x[4]
0 0.1 10.0	#x[5]
0 0.1 10.0	#x[6]
k7	#intermediate Jacobian column lengths
4
7
11
15
16
17
19
J0 3	#c1_c1
0 -0.1
2 0
6 0
J1 4	#c2_c2
0 -0.1
1 -0.1
3 0
7 0
J2 3	#c3_c3
2 0
4 0
6 0
J3 3	#c4_c4
3 0
5 0
7 0
J4 4	#c5_c5
0 -1
1 -1
2 0
3 0
J5 4	#c6_c6
0 -1
1 -1
2 0
3 0
G0 4	#obj
0 -1
1 -1
2 0
3 0
