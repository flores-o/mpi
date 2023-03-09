from mpi4py import MPI
import random

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Generate random data (distributed across nodes)
data = [random.randint(0, 100) for _ in range(20)]

# Calculate local max
local_max = max(data[rank::size])

# Use all_reduce to calculate global max
global_max = comm.allreduce(local_max, op=MPI.MAX)

# Print the global max
if rank == 0:
    print("The data is:", data)
    print("The maximum value in the data is:", global_max)
