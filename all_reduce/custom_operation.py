from mpi4py import MPI

# Define the custom operation
def my_max(a, b, datatype):
    """
    Custom operation to compute the maximum of two values
    """
    if datatype == MPI.FLOAT:
        return max(a, b, key=lambda x: float(x))
    elif datatype == MPI.DOUBLE:
        return max(a, b, key=lambda x: float(x))
    elif datatype == MPI.INT:
        return max(a, b, key=lambda x: int(x))
    else:
        raise NotImplementedError("Unsupported datatype")

# Initialize MPI
comm = MPI.COMM_WORLD

# Create a new custom operation
my_op = MPI.Op.Create(my_max, commute=True)

# Define the data to be reduced
data = 2.0 * comm.rank

# Perform the allreduce operation
result = comm.allreduce(data, op=my_op)

# Print the result
print("Process %d result: %f" % (comm.rank, result))

# Free the custom operation
my_op.Free()
