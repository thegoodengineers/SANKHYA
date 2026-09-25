g3 1 1 0	# problem hs113
 10 8 1 0 0 	# vars, constraints, objectives, ranges, eqns
 5 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 5 10 5 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 32 10 	# nonzeros in Jacobian, obj. gradient
 5 5	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c4_c4
o54	# sumlist
3	# (n)
o2	#*
n-3.0
o5	#^
o0	#+
v0	#x[1]
n-2.0
n2.0
o2	#*
n-4.0
o5	#^
o0	#+
v1	#x[2]
n-3.0
n2.0
o2	#*
n-2.0
o5	#^
v2	#x[3]
n2.0
C1	#c5_c5
o0	#+
o2	#*
n-5.0
o5	#^
v0	#x[1]
n2.0
o16	#-
o5	#^
o0	#+
v2	#x[3]
n-6.0
n2.0
C2	#c6_c6
o54	# sumlist
3	# (n)
o2	#*
n-0.5
o5	#^
o0	#+
v0	#x[1]
n-8.0
n2.0
o2	#*
n-2.0
o5	#^
o0	#+
v1	#x[2]
n-4.0
n2.0
o2	#*
n-3.0
o5	#^
v3	#x[5]
n2.0
C3	#c7_c7
o54	# sumlist
3	# (n)
o16	#-
o5	#^
v0	#x[1]
n2.0
o2	#*
n-2.0
o5	#^
o0	#+
v1	#x[2]
n-2.0
n2.0
o2	#*
o2	#*
n2.0
v0	#x[1]
v1	#x[2]
C4	#c8_c8
o2	#*
n-12.0
o5	#^
o0	#+
v4	#x[9]
n-8.0
n2.0
C5	#c1_c1
n0
C6	#c2_c2
n0
C7	#c3_c3
n0
O0 0	#obj
o0	#+
o54	# sumlist
11	# (n)
o5	#^
v0	#x[1]
n2.0
o5	#^
v1	#x[2]
n2.0
o2	#*
v0	#x[1]
v1	#x[2]
o5	#^
o0	#+
v2	#x[3]
n-10.0
n2.0
o2	#*
n4.0
o5	#^
o0	#+
v5	#x[4]
n-5.0
n2.0
o5	#^
o0	#+
v3	#x[5]
n-3.0
n2.0
o2	#*
n2.0
o5	#^
o0	#+
v6	#x[6]
n-1.0
n2.0
o2	#*
n5.0
o5	#^
v7	#x[7]
n2.0
o2	#*
n7.0
o5	#^
o0	#+
v8	#x[8]
n-11.0
n2.0
o2	#*
n2.0
o5	#^
o0	#+
v4	#x[9]
n-10.0
n2.0
o5	#^
o0	#+
v9	#x[10]
n-7.0
n2.0
n45.0
x10	# initial guess
0 2.0	#x[1]
1 3.0	#x[2]
2 5.0	#x[3]
3 1.0	#x[5]
4 6.0	#x[9]
5 5.0	#x[4]
6 2.0	#x[6]
7 7.0	#x[7]
8 3.0	#x[8]
9 10.0	#x[10]
r	#8 ranges (rhs's)
2 -120.0	#c4_c4
2 -40.0	#c5_c5
2 -30.0	#c6_c6
2 0.0	#c7_c7
2 0.0	#c8_c8
2 -105.0	#c1_c1
2 0.0	#c2_c2
2 -12.0	#c3_c3
b	#10 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
3	#x[5]
3	#x[9]
3	#x[4]
3	#x[6]
3	#x[7]
3	#x[8]
3	#x[10]
k9	#intermediate Jacobian column lengths
8
16
18
20
22
24
26
28
30
J0 4	#c4_c4
0 0
1 0
2 0
5 7.0
J1 4	#c5_c5
0 0
1 -8.0
2 0
5 2.0
J2 4	#c6_c6
0 0
1 0
3 0
6 1
J3 4	#c7_c7
0 0
1 0
3 -14.0
6 6.0
J4 4	#c8_c8
0 3.0
1 -6.0
4 0
9 7.0
J5 4	#c1_c1
0 -4.0
1 -5.0
7 3.0
8 -9.0
J6 4	#c2_c2
0 -10.0
1 8.0
7 17.0
8 -2.0
J7 4	#c3_c3
0 8.0
1 -2.0
4 -5.0
9 2.0
G0 10	#obj
0 -14.0
1 -16.0
2 0
3 0
4 0
5 0
6 0
7 0
8 0
9 0
