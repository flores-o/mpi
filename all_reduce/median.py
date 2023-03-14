from mpi4py import MPI
import statistics

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = [1, 2, 3, 4, 5, 6]

# Calculate the median of all numbers in the list
median = statistics.median(data)

# Use MPI allreduce to calculate the median across all processes
result = comm.allreduce(median, op=MPI.SUM)

# Divide the sum of medians by the number of processes to get the final median
result = result / comm.Get_size()

# Print the final result from each process
print(f"Process {rank}: {result}")
