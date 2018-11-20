# Triangle counting algorithm implementation on GPU using CUDA Numba in CSR format
from numba import cuda 
import numpy as np
import math
from numba.types import uint64
import numba

# device calls
@cuda.jit
def count_triangles_CSR(IA, JA, delta):

	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if i > 0 and i < len(IA)-2:

		a10T_row = i

		num_nnz_curr_row_a10T = IA[a10T_row + 1] - IA[a10T_row]

		a10T_start_col = IA[a10T_row]
		a10T_end_col = a10T_start_col

		# find the last element of a10T
		while JA[a10T_end_col] < i and \
		a10T_end_col < a10T_start_col + num_nnz_curr_row_a10T - 1:
			a10T_end_col += 1

		#prevent overcounting
		if JA[a10T_end_col] > i or \
		a10T_end_col == a10T_start_col + num_nnz_curr_row_a10T:
			a10T_end_col -= 1

		a12T_start_col = a10T_end_col + 1 
		a12T_end_col = a10T_start_col + num_nnz_curr_row_a10T - 1

		num_nnz_a10T = a10T_end_col - a10T_start_col + 1
		num_nnz_a12T = a12T_end_col - a12T_start_col + 1

		# adding the number of triangles for each iteration
		for k in range(num_nnz_a12T):
			a12T_select_location = JA[a12T_start_col+k] - JA[a12T_start_col]
			selected_row = JA[a12T_start_col] + a12T_select_location - (i+1)

			# debugging tool using kernel outputs
			# print("this is for moving vertex...", i)
			# print("______selected i-th row of A: ", selected_row)

			A_row = i + 1 + selected_row

			num_nnz_row_A20 = IA[A_row + 1] - IA[A_row]

			m, n = 0, 0
			while n < num_nnz_a10T and m < num_nnz_row_A20:
				# column index of ONE ele ment
				A20_u_col = IA[A_row] + m
				a10T_u_col = a10T_start_col + n
				if JA[A20_u_col] == JA[a10T_u_col]:
					cuda.atomic.add(delta,0,1)
					m += 1
					n += 1
				elif JA[A20_u_col] > JA[a10T_u_col]:
					n += 1
				else:
					m += 1
		
# host call

# test case 1 (3 triangles)
# A = np.array([[0,1,1,1,0],
#               [1,0,0,1,1],
#               [1,0,0,1,0],
#               [1,1,1,0,1],
#               [0,1,0,1,0]
#              ])

# IA = np.array([0,3,6,8,12,14])
# JA = np.array([1,2,3,
# 			   0,3,4,
# 			   0,3,
# 			   0,1,2,4,
# 			   1,3])

# test case 2 (2 triangles)
# IA = np.array([0,2,5,8,10])
# JA = np.array([1,2,0,2,3,0,1,3,1,2])

#test case 3 (5 triangles)
# IA = np.array([0,4,6,10,12,17,19,22])
# JA = np.array([1,2,4,6,0,2,0,1,3,4,2,4,0,2,3,5,6,4,6,0,4,5])

# test case 4 (13 triangles)
# IA = np.array([0,5,9,14,18,23,27,32])
# JA = np.array([1,2,4,5,6,0,2,3,6,0,1,3,4,6,1,2,4,5,0,2,3,5,6,0,3,4,6,0,1,2,4,5])

d_type = "uint32"

IA_file = "/mnt/large/graph-datasets/p2p-Gnutella31_adj_IA.txt.bin"
IA_open = open(IA_file, "rb")
IA_open.seek(4)
IA = np.fromfile(IA_open, dtype = d_type)

JA_file = "/mnt/large/graph-datasets/p2p-Gnutella31_adj_JA.txt.bin"
JA_open = open(JA_file, "rb")
JA_open.seek(4)
JA = np.fromfile(JA_open, dtype = d_type)

delta = [0]

IA_d = cuda.to_device(IA)
JA_d = cuda.to_device(JA)

delta_d = cuda.device_array(1)

threadsperblock = 32

blocks = math.ceil(IA.shape[0]/ threadsperblock)
blockspergrid = blocks

print("Running count_triangles kernel...")

count_triangles_CSR[blockspergrid, threadsperblock](IA_d, JA_d, delta_d)

delta = delta_d.copy_to_host()

print("Number of triangles: ", delta[0])

