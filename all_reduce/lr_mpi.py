import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Define the function for gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        h = X.dot(theta)
        loss = h - y
        gradient = X.T.dot(loss) / m
        theta -= alpha * gradient
    return theta

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the dataset
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]])
y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

# Divide the data into equal parts for each process
n = len(y) // size
X_local = X[rank*n:(rank+1)*n]
y_local = y[rank*n:(rank+1)*n]

# Initialize the parameters
theta = np.zeros(2)
alpha = 0.01
iterations = 1000

# Perform gradient descent on each process
theta_local = gradient_descent(X_local, y_local, theta, alpha, iterations)

# Gather the results to the root process
theta_all = None
if rank == 0:
    theta_all = np.empty([size, 2])
theta_all = comm.gather(theta_local, root=0)

# Plot the results
if rank == 0:
    theta_avg = np.mean(theta_all, axis=0)
    plt.scatter(X[:,1], y)
    plt.plot(X[:,1], X.dot(theta_avg), color='red')
    plt.show()
