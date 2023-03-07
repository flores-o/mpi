import numpy as np
from mpi4py import MPI

import os
os.environ['OMPI_MCA_btl_vader_single_copy_mechanism'] = 'none'

# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from rank {rank} of {size}!")

# Define matrix sizes
m = 1000
n = 1000
p = 1000

# Define shard sizes
num_rows_per_shard = m // size
num_cols_per_shard = p // size

# Initialize local shards
A_local = np.zeros((num_rows_per_shard, n))
B_local = np.zeros((n, num_cols_per_shard))

# Generate random matrices on rank 0
if rank == 0:
    A = np.random.rand(m, n)
    B = np.random.rand(n, p)
else:
    A = None
    B = None

# Scatter A and B to all processes
comm.Scatter(A, A_local, root=0)
comm.Scatter(B, B_local, root=0)

# Compute local matrix multiplication
C_local = np.dot(A_local, B_local)

# All-reduce C_local across all processes
C_global = np.zeros((m, p))

print(f"C_local.shape = {C_local.shape}")
print(f"C_global.shape = {C_global.shape}")

comm.Allreduce(C_local, C_global, op=MPI.SUM)

# Verify result on rank 0
if rank == 0:
    C_exact = np.dot(A, B)

    #print(f"{A}")
    #print(f"{B}")

    print("Are the matrices equal?")
    print(np.allclose(C_global, C_exact, rtol=1e-5))

    # assert np.allclose(C_global, C_exact)
    # print("Multiplication successful!")
