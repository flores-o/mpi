from mpi4py import MPI
import time

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Define the data to be summed (distributed across nodes)
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Serial implementation
start_time = time.time()
serial_sum = sum(data)
serial_time = time.time() - start_time

# Parallel implementation
start_time = time.time()
local_sum = sum(data[rank::size])
global_sum = comm.allreduce(local_sum, op=MPI.SUM)
parallel_time = time.time() - start_time

# Print the results
if rank == 0:
    print("Serial sum: ", serial_sum)
    print("Parallel sum: ", global_sum)
    print("Serial time: ", serial_time)
    print("Parallel time: ", parallel_time)
