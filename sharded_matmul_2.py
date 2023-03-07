# code generated with ChatGPT
# run with mpirun -np 2 python sharded_matmul_2.py => forever loop


import numpy as np
from mpi4py import MPI

# Define constants
MATSIZE = 500
NRA = MATSIZE   # number of rows in matrix A
NCA = MATSIZE   # number of columns in matrix A
NCB = MATSIZE   # number of columns in matrix B
MASTER = 0      # taskid of first task
FROM_MASTER = 1 # setting a message type
FROM_WORKER = 2 # setting a message type

# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numtasks = comm.Get_size()

if numtasks < 2:
    print("Need at least two MPI tasks. Quitting...")
    MPI.Abort(MPI_COMM_WORLD, 1)
    exit(1)

numworkers = numtasks - 1

# Define matrices
a = np.zeros((NRA, NCA))
b = np.zeros((NCA, NCB))
c = np.zeros((NRA, NCB))

# Master task
if rank == MASTER:
    print("mpi_mm has started with", numtasks, "tasks.")
    
    # Initialize matrices
    for i in range(NRA):
        for j in range(NCA):
            a[i][j] = i+j
    for i in range(NCA):
        for j in range(NCB):
            b[i][j] = i*j
    
    # Measure start time
    start = MPI.Wtime()
    
    # Send matrix data to the worker tasks
    averow = NRA // numworkers
    extra = NRA % numworkers
    offset = 0
    mtype = FROM_MASTER
    for dest in range(1, numworkers+1):
        rows = averow + 1 if dest <= extra else averow
        comm.send(offset, dest=dest, tag=mtype)
        comm.send(rows, dest=dest, tag=mtype)
        comm.send(a[offset:offset+rows, :], dest=dest, tag=mtype)
        comm.send(b, dest=dest, tag=mtype)
        offset += rows
    
    # Receive results from worker tasks
    mtype = FROM_WORKER
    for i in range(1, numworkers+1):
        source = i
        offset = comm.recv(source=source, tag=mtype)
        rows = comm.recv(source=source, tag=mtype)
        c[offset:offset+rows, :] = comm.recv(source=source, tag=mtype)
    
    # Print results
    """
    print("******************************************************")
    print("Result Matrix:")
    for i in range(NRA):
        print("")
        for j in range(NCB):
            print("%6.2f   " % c[i][j], end="")
    print("\n******************************************************")
    """
    
    # Measure finish time
    finish = MPI.Wtime()
    print("Done in", finish - start, "seconds.")

# Worker task
if rank > MASTER:
    mtype = FROM_MASTER
    offset = comm.recv(source=MASTER, tag=mtype)
    rows = comm.recv(source=MASTER, tag=mtype)
    a = comm.recv(source=MASTER, tag=mtype)
    b = comm.recv(source=MASTER, tag=mtype)

    for k in range(NCB):
        for i in range(rows):
            c[i][k] = 0.0
            for j in range(NCA):
                c[i][k] += a[i][j] * b[j][k]
    
    mtype = FROM_WORKER
    comm.send(offset, dest=MASTER, tag=mtype)
