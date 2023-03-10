from mpi4py import MPI
import numpy as np

SIZE = 8			# Size of matrices
A = np.zeros((SIZE, SIZE), dtype=int)
B = np.zeros((SIZE, SIZE), dtype=int)
C = np.zeros((SIZE, SIZE), dtype=int)

def fill_matrix(m):
    n = 0
    for i in range(SIZE):
        for j in range(SIZE):
            m[i][j] = n
            n += 1

def print_matrix(m):
    for i in range(SIZE):
        print("\n\t| ", end='')
        for j in range(SIZE):
            print("%2d " % m[i][j], end='')
        print("|")

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()	# Who am I?
P = comm.Get_size()	# Number of processes

# Just to use the simple variants of MPI_Gather and MPI_Scatter we
# impose that SIZE is divisible by P. By using the vector versions,
# (MPI_Gatherv and MPI_Scatterv) it is easy to drop this restriction.

if SIZE % P != 0:
    if myrank == 0:
        print("Matrix size not divisible by number of processors")
    MPI.Finalize()
    exit(-1)

from_row = myrank * SIZE // P
to_row = (myrank + 1) * SIZE // P

# Process 0 fills the input matrices and broadcasts them to the rest
# (actually, only the relevant stripe of A is sent to each process)

if myrank == 0:
    fill_matrix(A)
    fill_matrix(B)

comm.Bcast(B, root=0)
comm.Scatter(A, A[from_row:to_row,:], root=0)

print("computing slice %d (from row %d to %d)" % (myrank, from_row, to_row - 1))
for i in range(from_row, to_row):
    for j in range(SIZE):
        C[i][j] = 0
        for k in range(SIZE):
            C[i][j] += A[i][k] * B[k][j]

comm.Gather(C[from_row:to_row,:], C, root=0)

if myrank == 0:
    print("\n\n")
    print_matrix(A)
    print("\n\n\t       * \n")
    print_matrix(B)
    print("\n\n\t       = \n")
    print_matrix(C)
    print("\n\n")

    C_true_label = np.dot(A, B)
    print("Are the matrixes equal?", np.allclose(C, C_true_label, rtol=1e-5))


MPI.Finalize()
