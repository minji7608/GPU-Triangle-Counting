# python SUDO CODE for triangle counting algorithm 

import numpy

def dot_product(A,B):
	result = 0
	for i in range(len(A)):
		result += A[i]*B[i]
	return result

A = numpy.array([[0,1,1,0,1,1,1],
				 [1,0,1,1,0,0,1],
				 [1,1,0,1,1,0,1],
				 [0,1,1,0,1,1,0],
				 [1,0,1,1,0,1,1],
				 [1,0,0,1,1,0,1],
				 [1,1,1,0,1,1,0]
				 ])

row_bound = len(A)
col_bound = len(A[0])

delta = 0

for k in range(len(A)):
	# current row, col
	i, j = k, k

	A20_start_row = i+1
	A20_start_col = 0
	A20_end_row = row_bound
	A20_end_col = j-1

	# a12T (row vector)
	a12T_row = i
	a12T_start_col = j+1
	a12T_end_col = col_bound

	# a10T (row vector)
	a10T_row = i
	a10T_start_col = 0
	a10T_end_col = j-1

	A20 = A[A20_start_row:A20_end_row+1, A20_start_col:A20_end_col+1]
	a12T = A[a12T_row, a12T_start_col:a12T_end_col+1]
	a10T = A[a10T_row, a10T_start_col:a10T_end_col+1]

	print("This is an iteration to move vertex [%d][%d]" %(k,k))
	print("A20 is")
	print(A20)
	print("a12T is", a12T)
	print("a10T is", a10T)


	selected_rows = []
	final = 0

	for k in range(len(a12T)):
		if a12T[k] == 1:
			selected_rows += [k]

	print("selected", selected_rows)

	if len(A20) != 0 and selected_rows != []:
		for row in selected_rows:
			final += dot_product(A20[row], a10T)
			print(final)

	delta += final

	# result = numpy.matmul(a12T, A20)
	# print("____________", result)
	# result = numpy.matmul(result, a10T)
	# print("_____________________", result)

	# delta += result

print("delta is", delta)

