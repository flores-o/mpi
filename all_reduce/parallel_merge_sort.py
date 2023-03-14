"""
@ mpirun -np 4 python parallel_merge_sort.py 16

Unsorted array: [13, 14, 2, 9, 16, 13, 10, 16, 12, 7, 5, 10, 5, 4, 9, 5]
Sorted array: [2, 4, 5, 5, 5, 7, 9, 9, 10, 10, 12, 13, 13, 14, 16, 16]
Unsorted array: [13, 14, 2, 9, 16, 13, 10, 16, 12, 7, 5, 10, 5, 4, 9, 5]
Sorted array: [2, 4, 5, 5, 5, 7, 9, 9, 10, 10, 12, 13, 13, 14, 16, 16]
"""

from mpi4py import MPI
import random
import sys

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    middle = len(arr) // 2
    left = arr[:middle]
    right = arr[middle:]

    left = merge_sort(left)
    right = merge_sort(right)
    return merge(left, right)

def merge(left, right):
    result = []
    while left and right:
        if left[0] < right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result += left
    result += right
    return result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 2:
        if rank == 0:
            print('Usage: mpirun -np <num_procs> python parallel_merge_sort.py <array_size>')
        sys.exit()

    array_size = int(sys.argv[1])
    random.seed(rank)

    if rank == 0:
        data = [random.randint(1, array_size) for _ in range(array_size)]
        print('Unsorted array:', data)
        chunks = [data[i::size] for i in range(size)]
    else:
        chunks = None

    local_data = comm.scatter(chunks, root=0)
    local_sorted_data = merge_sort(local_data)

    sorted_data = comm.gather(local_sorted_data, root=0)

    if rank == 0:
        final_sorted_data = []
        for chunk in sorted_data:
            final_sorted_data = merge(final_sorted_data, chunk)
        print('Sorted array:', final_sorted_data)

if __name__ == '__main__':
    main()