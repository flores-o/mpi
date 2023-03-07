import numpy as np
from mpi4py import MPI

N = 4
A = np.random.rand(N, N)
B = np.random.rand(N, N)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Divide A and B among the processes
A_rows = N // size
B_cols = N // size

# Scatter A and B to all processes
local_A = np.zeros((A_rows, N))
local_B = np.zeros((N, B_cols))

comm.Scatter(A, local_A, root=0)
comm.Scatter(B, local_B, root=0)

# Compute local C = local_A * local_B
local_C = np.dot(local_A, local_B)

# Combine all the local_C matrices using Allreduce
C = np.zeros((N, N))
comm.Allreduce(local_C, C, op=MPI.SUM)

# Print out the result
if rank == 0:
    print(C)
