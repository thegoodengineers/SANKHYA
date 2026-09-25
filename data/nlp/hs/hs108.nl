g3 1 1 0	# problem hs108
 9 14 1 0 0 	# vars, constraints, objectives, ranges, eqns
 13 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 9 9 9 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 40 9 	# nonzeros in Jacobian, obj. gradient
 7 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_c1
o0	#+
o16	#-
o5	#^
v2	#x[3]
n2.0
o16	#-
o5	#^
v3	#x[4]
n2.0
C1	#c2_c2
o0	#+
o16	#-
o5	#^
v4	#x[5]
n2.0
o16	#-
o5	#^
v5	#x[6]
n2.0
C2	#c3_c3
o16	#-
o5	#^
v8	#x[9]
n2.0
C3	#c4_c4
o0	#+
o16	#-
o5	#^
v0	#x[1]
n2.0
o16	#-
o5	#^
o0	#+
v1	#x[2]
o2	#*
n-1
v8	#x[9]
n2.0
C4	#c5_c5
o0	#+
o16	#-
o5	#^
o0	#+
v0	#x[1]
o2	#*
n-1
v4	#x[5]
n2.0
o16	#-
o5	#^
o0	#+
v1	#x[2]
o2	#*
n-1
v5	#x[6]
n2.0
C5	#c6_c6
o0	#+
o16	#-
o5	#^
o0	#+
v0	#x[1]
o2	#*
n-1
v6	#x[7]
n2.0
o16	#-
o5	#^
o0	#+
v1	#x[2]
o2	#*
n-1
v7	#x[8]
n2.0
C6	#c7_c7
o0	#+
o16	#-
o5	#^
o0	#+
v2	#x[3]
o2	#*
n-1
v6	#x[7]
n2.0
o16	#-
o5	#^
o0	#+
v3	#x[4]
o2	#*
n-1
v7	#x[8]
n2.0
C7	#c8_c8
o0	#+
o16	#-
o5	#^
o0	#+
v2	#x[3]
o2	#*
n-1
v4	#x[5]
n2.0
o16	#-
o5	#^
o0	#+
v3	#x[4]
o2	#*
n-1
v5	#x[6]
n2.0
C8	#c9_c9
o0	#+
o16	#-
o5	#^
v6	#x[7]
n2.0
o16	#-
o5	#^
o0	#+
v7	#x[8]
o2	#*
n-1
v8	#x[9]
n2.0
C9	#c10_c10
o0	#+
o2	#*
v0	#x[1]
v3	#x[4]
o16	#-
o2	#*
v1	#x[2]
v2	#x[3]
C10	#c11_c11
o2	#*
v2	#x[3]
v8	#x[9]
C11	#c12_c12
o2	#*
o2	#*
n-1
v4	#x[5]
v8	#x[9]
C12	#c13_c13
o0	#+
o2	#*
v4	#x[5]
v7	#x[8]
o16	#-
o2	#*
v5	#x[6]
v6	#x[7]
C13	#c14_c14
n0
O0 0	#obj
o2	#*
n-0.5
o54	# sumlist
6	# (n)
o2	#*
v0	#x[1]
v3	#x[4]
o16	#-
o2	#*
v1	#x[2]
v2	#x[3]
o2	#*
v2	#x[3]
v8	#x[9]
o16	#-
o2	#*
v4	#x[5]
v8	#x[9]
o2	#*
v4	#x[5]
v7	#x[8]
o16	#-
o2	#*
v5	#x[6]
v6	#x[7]
x9	# initial guess
0 1.0	#x[1]
1 1.0	#x[2]
2 1.0	#x[3]
3 1.0	#x[4]
4 1.0	#x[5]
5 1.0	#x[6]
6 1.0	#x[7]
7 1.0	#x[8]
8 1.0	#x[9]
r	#14 ranges (rhs's)
2 -1.0	#c1_c1
2 -1.0	#c2_c2
2 -1.0	#c3_c3
2 -1.0	#c4_c4
2 -1.0	#c5_c5
2 -1.0	#c6_c6
2 -1.0	#c7_c7
2 -1.0	#c8_c8
2 -1.0	#c9_c9
2 0.0	#c10_c10
2 0.0	#c11_c11
2 0.0	#c12_c12
2 0.0	#c13_c13
2 0.0	#c14_c14
b	#9 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
3	#x[4]
3	#x[5]
3	#x[6]
3	#x[7]
3	#x[8]
3	#x[9]
k8	#intermediate Jacobian column lengths
4
8
13
17
22
26
30
34
J0 2	#c1_c1
2 0
3 0
J1 2	#c2_c2
4 0
5 0
J2 1	#c3_c3
8 0
J3 3	#c4_c4
0 0
1 0
8 0
J4 4	#c5_c5
0 0
1 0
4 0
5 0
J5 4	#c6_c6
0 0
1 0
6 0
7 0
J6 4	#c7_c7
2 0
3 0
6 0
7 0
J7 4	#c8_c8
2 0
3 0
4 0
5 0
J8 3	#c9_c9
6 0
7 0
8 0
J9 4	#c10_c10
0 0
1 0
2 0
3 0
J10 2	#c11_c11
2 0
8 0
J11 2	#c12_c12
4 0
8 0
J12 4	#c13_c13
4 0
5 0
6 0
7 0
J13 1	#c14_c14
8 1
G0 9	#obj
0 0
1 0
2 0
3 0
4 0
5 0
6 0
7 0
8 0
