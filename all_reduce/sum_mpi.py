from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Define the input data and the local sum for this node
data = rank + 1
local_sum = data

# Use the MPI allreduce operation to perform a distributed summation across all nodes
global_sum = comm.allreduce(local_sum, op=MPI.SUM)

# Print the results for each node
print("Node", rank, "has data =", data, "and global sum =", global_sum)

# Finalize MPI
MPI.Finalize()
