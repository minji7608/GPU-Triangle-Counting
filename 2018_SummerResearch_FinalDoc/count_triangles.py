# Triangle counting algorithm implementation on GPU using CUDA Numba in CSR format
from numba import cuda 
import numpy as np
import math
from numba.types import int64
import numba

# device calls
@cuda.jit
def count_triangles(A, delta):

	row_bound = A.shape[0]-1
	col_bound = A.shape[1]-1

	curr_row, curr_col = cuda.threadIdx.x, cuda.threadIdx.y

	# assume square matrix
	if curr_row == curr_col and curr_row < row_bound:

		i, j = curr_row, curr_col

		A20_start_row = i+1
		A20_end_row = row_bound
		A20_start_col = 0
		A20_end_col = j-1

		# a12T (row vector)
		a12T_row = i
		a12T_start_col = j+1
		a12T_end_col = col_bound

		# a10T (row vector)
		a10T_row = i
		a10T_start_col = 0
		a10T_end_col = j-1

		result = 0

		for p in range(a12T_start_col, a12T_end_col+1):
			# select rows of A20
			if A[a12T_row][p] == 1:
				selected_row = p
				for k in range(A20_start_col, A20_end_col+1):
					result += A[selected_row][A20_start_col + k]*A[a10T_row][a10T_start_col + k]

		cuda.atomic.add(delta,0,result)

# host calls

A = np.array([[0,1,1,0],
			  [1,0,1,1],
			  [1,1,0,1],
			  [0,1,1,0],
			 ])
delta = []

A_global_mem = cuda.to_device(A)
delta_global_mem = cuda.device_array(1)

threadsperblock = (16, 16) 
blocks = math.ceil(A.shape[0]/ threadsperblock[0])
blockspergrid = (blocks, blocks)

print("Running count_triangles kernel...")

count_triangles[blockspergrid, threadsperblock](A_global_mem, delta_global_mem)
delta = delta_global_mem.copy_to_host()

print("There are %d triangles in this graph!" %delta[0])

