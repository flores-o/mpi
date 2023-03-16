import numpy as np
from mpi4py import MPI

def generate_data():
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.randn(100, 1)
    return x, y

def compute_stochastic_gradient(x_i, y_i, theta):
    prediction = x_i.dot(theta)
    gradient = (2 * x_i.T * (prediction - y_i)).reshape(-1, 1)
    return gradient

def stochastic_gradient_descent(x, y, theta, lr, num_epochs, batch_size):
    m = len(y)
    for epoch in range(num_epochs):
        shuffled_indices = np.random.permutation(m)
        x_shuffled = x[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, batch_size):
            x_i = x_shuffled[i:i+batch_size]
            y_i = y_shuffled[i:i+batch_size]
            gradients = compute_stochastic_gradient(x_i, y_i, theta)
            theta -= lr * gradients
    return theta

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate data
x, y = generate_data()
x_b = np.c_[np.ones((len(x), 1)), x]

# Parameters
learning_rate = 0.1
num_epochs = 100
batch_size = 1
initial_theta = np.array([[0.0], [0.0]], dtype=np.float64)

if rank == 0:
    print(f"Initial theta: {initial_theta}")

# Split the gradient descent epochs across processes
epochs_per_process = num_epochs // size
local_theta = stochastic_gradient_descent(x_b, y, initial_theta, learning_rate, epochs_per_process, batch_size)

# Combine the results using Allreduce
global_theta = np.zeros_like(local_theta)
comm.Allreduce(local_theta, global_theta, op=MPI.SUM)

# Calculate the average theta
global_theta /= size

if rank == 0:
    print(f"Global theta: {global_theta}")
