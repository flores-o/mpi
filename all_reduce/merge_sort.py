#!/usr/bin/env python
# merge.py
import numpy as np
from mpi4py import MPI

# Initialize MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get the process rank
size = comm.Get_size() # get the number of processes
status = MPI.Status() # status object used for message passing

# Define the size of the array to be sorted
N = 16

# Initialize arrays for unsorted and sorted data, as well as temporary arrays
unsorted = np.zeros(N, dtype="int")
final_sorted = np.zeros(N, dtype="int")
local_array = np.zeros(int(N / size), dtype="int")
local_tmp = np.zeros(int(N / size), dtype="int")
local_remain = np.zeros(2 * int(N / size), dtype="int")

# If the current process is rank 0, generate a random array and print it
if rank == 0:
    unsorted = np.random.randint(low=0,high=N,size=N)
    print (unsorted)

# Scatter the data from rank 0 to all processes
comm.Scatter(unsorted, local_array, root=0)

print(f"Hello, I am rank {rank} and I have the following local array after scatter operation: {local_array}")

# Sort the local arrays
local_array.sort()

# Start the merge process
step = size / 2
while (step >= 1):
    # If the current process rank is between step and 2 * step, send the local array to another process
    if (rank >= step and rank < step * 2):
        comm.Send(local_array, rank - step, tag=0)
    # If the current process rank is less than step, receive an array from another process and merge the arrays
    elif (rank < step):
        # Initialize temporary arrays for merging
        local_tmp = np.zeros(local_array.size, dtype="int")
        local_remain = np.zeros(2 * local_array.size, dtype="int")
        # Receive the array from the other process
        comm.Recv(local_tmp, rank + step, tag=0)
        # Merge the local array and the received array
        i = 0 # local_array counter
        j = 0 # local_tmp counter
        for k in range (0, 2 * local_array.size):
            if (i >= local_array.size):
                local_remain[k] = local_tmp[j]
                j += 1
            elif (j >= local_array.size):
                local_remain[k] = local_array[i]
                i += 1
            elif (local_array[i] > local_tmp[j]):
                local_remain[k] = local_tmp[j]
                j += 1
            else:
                local_remain[k] = local_array[i]
                i += 1


        print(f" ------------------- \n Rank: {rank} \n Step: {step}\nlocal array: {local_array}\nlocal tmp: {local_tmp}\nlocal remain: {local_remain} \n ------------------- \n")

        # Update the local array with the merged result
        local_array = local_remain
    # Reduce the step size for the next iteration
    step = step / 2

# If the current process is rank 0, print the final sorted array
if (rank == 0):
    print (local_array)
