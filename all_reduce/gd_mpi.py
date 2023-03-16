import numpy as np
from mpi4py import MPI

def f_gradient(x):
    return 2 * (x - 2)

def gradient_descent(x, lr, num_iterations):
    for _ in range(num_iterations):
        x -= lr * f_gradient(x)
    return x

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
learning_rate = 0.1
num_iterations = 100
initial_x = np.array([5.0], dtype=np.float64)

if rank == 0:
    print(f"Initial x: {initial_x}")

# Split the gradient descent iterations across processes
iterations_per_process = num_iterations // size
local_x = gradient_descent(initial_x, learning_rate, iterations_per_process)

# Combine the results using Allreduce
global_x = np.zeros(1, dtype=np.float64)

# print(f"Rank {rank}, Local x: {local_x}")
comm.Allreduce(local_x, global_x, op=MPI.SUM)

# Calculate the average x
global_x /= size

if rank == 0:
    print(f"Global minimum x: {global_x}")
