import numpy as np
from comparison import benchmark

def row_sums(N, A):
   for i in range(N): s = np.sum(A[i, :])

def column_sums(N, A):
   for j in range(N): s = np.sum(A[:, j])

if __name__ == "__main__":
    N = 10000
    A = np.random.rand(N, N)
    A_f = np.asfortranarray(A)

    benchmark(row_sums, N, A_f)
    benchmark(column_sums, N, A_f)