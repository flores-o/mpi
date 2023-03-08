# tutorial: https://www.youtube.com/watch?v=umDORTtqMfo&list=PLxDvEmlm4QvgcMJLy3BiFZZ0J8fLXCuD4&index=8&ab_channel=sentdex

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:

    print(f"sending data from process rank {rank}")
    shared = {'d1': 1, 'd2': 2}
    comm.send(shared, dest=1, tag=1)

    shared2 = {'d1': 10, 'd2': 20}
    comm.send(shared2, dest=1, tag=2)

if rank == 1:

    print(f"receiving data in process rank {rank}")
    receive = comm.recv(source=0, tag=2)
    print(f"receive {receive}")

    receive2 = comm.recv(source=0, tag=1)
    print(f"receive {receive2}")
