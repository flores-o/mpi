from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Define the data to be summed (distributed across nodes)
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Calculate local sum
print("rank", rank, "data",data[rank::size])
local_sum = sum(data[rank::size])

# Use all_reduce to calculate global sum
global_sum = comm.allreduce(local_sum, op=MPI.SUM)

# Print the global sum
if rank == 0:
    print("The sum of the data is:", global_sum)
