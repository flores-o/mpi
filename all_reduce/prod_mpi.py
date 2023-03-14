# draft, not working
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = [1, 2, 3, 4, 5, 6]

# Calculate the product of all numbers in the list
product = 1
for number in data:
    product *= number

# Use MPI allreduce to calculate the product across all processes
result = comm.allreduce(product, op=MPI.PROD)

# Print the final result from each process
print(f"Process {rank}: {result}")
