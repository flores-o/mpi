from mpi4py import MPI
import numpy as np

def tree_all_reduce(send_data, recv_data, op=MPI.SUM, root=0):
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # Check if the number of processes is a power of 2
    if not (size & (size - 1) == 0):
        raise ValueError("Number of processes must be a power of 2.")

    # Initialize recv_data with the same data as send_data
    recv_data[:] = send_data

    # Get the number of levels in the tree
    levels = int(np.log2(size))

    for level in range(levels):
        # Compute the partner rank at the current level
        partner = rank ^ (1 << level)

        # Send data to partner rank and receive partner's data
        partner_data = np.empty_like(recv_data)
        MPI.COMM_WORLD.Sendrecv(recv_data, dest=partner, sendtag=level,
                                recvbuf=partner_data, source=partner, recvtag=level)

        # Perform the reduction operation
        if op == MPI.SUM:
            recv_data += partner_data
        elif op == MPI.PROD:
            recv_data *= partner_data
        else:
            raise ValueError("Unsupported reduction operation.")

# Example usage
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create an array with data specific to each process
    send_data = np.array([rank], dtype=np.int32)
    recv_data = np.empty_like(send_data)

    # Perform Tree All Reduce
    tree_all_reduce(send_data, recv_data)

    # Print the results
    print(f"Process {rank}: {recv_data}")
