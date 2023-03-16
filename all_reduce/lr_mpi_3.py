import numpy as np
from mpi4py import MPI

def generate_data():
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.randn(100, 1)
    return x, y

def compute_gradients(x, y, theta):
    m = len(y)
    predictions = x.dot(theta)
    gradients = (2/m) * x.T.dot(predictions - y)
    return gradients

def gradient_descent(x, y, theta, lr, num_iterations):
    for _ in range(num_iterations):
        gradients = compute_gradients(x, y, theta)
        theta -= lr * gradients
    return theta

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate data
x, y = generate_data()
x_b = np.c_[np.ones((len(x), 1)), x]

comm.barrier()
if rank == 0:
    print(f"x[:10] {x[:10]}")
    print(f"y[:10] {y[:10]}")
    print(f"x_b[:10] {x_b[:10]}")
comm.barrier()

# Parameters
learning_rate = 0.1
num_iterations = 1000
initial_theta = np.array([[0.0], [0.0]], dtype=np.float64)

if rank == 0:
    print(f"Initial theta: {initial_theta}")

# Split the gradient descent iterations across processes
iterations_per_process = num_iterations // size
local_theta = gradient_descent(x_b, y, initial_theta, learning_rate, iterations_per_process)

# Combine the results using Allreduce
global_theta = np.zeros_like(local_theta)
comm.Allreduce(local_theta, global_theta, op=MPI.SUM)

# Calculate the average theta
global_theta /= size


if rank == 0:
    print(f"Global theta: {global_theta}")
    print(f"x[:10]: {x[:10]}")
    print(f"y[:10]: {y[:10]}")
    print(f"global_theta.T.dot(x_b[:10].T): {global_theta.T.dot(x_b[:10].T)}")
