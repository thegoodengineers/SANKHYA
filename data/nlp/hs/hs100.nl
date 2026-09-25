g3 1 1 0	# problem hs100
 7 4 1 0 0 	# vars, constraints, objectives, ranges, eqns
 4 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 5 7 5 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 19 7 	# nonzeros in Jacobian, obj. gradient
 10 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1_constr1
o54	# sumlist
3	# (n)
o2	#*
n2.0
o5	#^
v0	#x[1]
n2.0
o2	#*
n3.0
o5	#^
v1	#x[2]
n4.0
o2	#*
n4.0
o5	#^
v3	#x[4]
n2.0
C1	#c2_constr2
o2	#*
n10.0
o5	#^
v2	#x[3]
n2.0
C2	#c3_constr3
o0	#+
o5	#^
v1	#x[2]
n2.0
o2	#*
n6.0
o5	#^
v4	#x[6]
n2.0
C3	#c4_constr4
o54	# sumlist
4	# (n)
o2	#*
n-4.0
o5	#^
v0	#x[1]
n2.0
o16	#-
o5	#^
v1	#x[2]
n2.0
o2	#*
o2	#*
n3.0
v0	#x[1]
v1	#x[2]
o2	#*
n-2.0
o5	#^
v2	#x[3]
n2.0
O0 0	#obj
o54	# sumlist
8	# (n)
o5	#^
o0	#+
v0	#x[1]
n-10.0
n2.0
o2	#*
n5.0
o5	#^
o0	#+
v1	#x[2]
n-12.0
n2.0
o5	#^
v2	#x[3]
n4.0
o2	#*
n3.0
o5	#^
o0	#+
v3	#x[4]
n-11.0
n2.0
o2	#*
n10.0
o5	#^
v5	#x[5]
n6.0
o2	#*
n7.0
o5	#^
v4	#x[6]
n2.0
o5	#^
v6	#x[7]
n4.0
o16	#-
o2	#*
o2	#*
n4.0
v4	#x[6]
v6	#x[7]
x7	# initial guess
0 1.0	#x[1]
1 2.0	#x[2]
2 0.0	#x[3]
3 4.0	#x[4]
4 1.0	#x[6]
5 0.0	#x[5]
6 1.0	#x[7]
r	#4 ranges (rhs's)
1 127.0	#c1_constr1
1 282.0	#c2_constr2
1 196.0	#c3_constr3
2 0.0	#c4_constr4
b	#7 bounds (on variables)
3	#x[1]
3	#x[2]
3	#x[3]
3	#x[4]
3	#x[6]
3	#x[5]
3	#x[7]
k6	#intermediate Jacobian column lengths
4
8
11
13
15
17
J0 5	#c1_constr1
0 0
1 0
2 1
3 0
5 5.0
J1 5	#c2_constr2
0 7.0
1 3.0
2 0
3 1
5 -1
J2 4	#c3_constr3
0 23.0
1 0
4 0
6 -8.0
J3 5	#c4_constr4
0 0
1 0
2 0
4 -5.0
6 11.0
G0 7	#obj
0 0
1 0
2 0
3 0
4 -10.0
5 0
6 -8.0
