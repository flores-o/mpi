# not working

import mpi4py.MPI as MPI
import math

def allreduceRSAG(sendbuf, recvbuf):
    thisProc = MPI.COMM_WORLD.Get_rank()
    nProc = MPI.COMM_WORLD.Get_size()

    dest = 0
    width = 0
    myData = sendbuf.copy()
    temp_sendbuf = np.zeros_like(myData)

    nPhases = int(math.log2(nProc))

    for p in range(nPhases):
        width = int(len(myData) / (2**(p+1)))

        if thisProc % (2**(p+1)) < (2**p):
            dest = thisProc + 2**p
            temp_sendbuf[:width] = myData[width:]  # second half of the message
            MPI.COMM_WORLD.Send(temp_sendbuf[:width], dest=dest, tag=0)
            MPI.COMM_WORLD.Recv(recvbuf[:width], source=dest, tag=1)

            for i in range(width):
                myData[i] += recvbuf[i]  # sum the replay

        else:
            dest = thisProc - 2**p
            MPI.COMM_WORLD.Recv(recvbuf[:width], source=dest, tag=0)
            temp_sendbuf[:width] = myData[:width]  # first half of the message

            for i in range(width):
                myData[i+width] += recvbuf[i]  # sum the replay

            MPI.COMM_WORLD.Send(temp_sendbuf[:width], dest=dest, tag=1)
            # Copy the second half on the first half
            for i in range(width):
                myData[i] = myData[i+width]

    for p in reversed(range(nPhases)):
        width = int(len(myData) / (2**(p+1)))
        if thisProc % (2**(p+1)) < (2**p):
            dest = thisProc + 2**p
            temp_sendbuf[:width] = myData[:width]
            MPI.COMM_WORLD.Send(temp_sendbuf[:width], dest=dest, tag=0)
            MPI.COMM_WORLD.Recv(recvbuf[:width], source=dest, tag=1)

            for i in range(width):
                myData[i+width] = recvbuf[i]

        else:
            dest = thisProc - 2**p
            MPI.COMM_WORLD.Recv(recvbuf[:width], source=dest, tag=0)
            temp_sendbuf[:width] = myData[:width]

            for i in range(width):
                myData[i+width] = myData[i]

            for i in range(width):
                myData[i] = recvbuf[i]

            MPI.COMM_WORLD.Send(temp_sendbuf[:width], dest=dest, tag=1)

    recvbuf[:] = myData[:]

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    thisProc = comm.Get_rank()
    nProc = comm.Get_size()

    # initialize sendbuf
    import numpy as np
    sendbuf = np.array([((i+1)*(thisProc+1)) for i in range(nProc)])
    # initialize recvbuf
    recvbuf = np.zeros(nProc, dtype=np.float64)

    allreduceRSAG(sendbuf, recvbuf)

    # Optionally: write test code
    # (this is not required as we will only test the allreduceRSAG implementation itself.

    MPI.Finalize()
