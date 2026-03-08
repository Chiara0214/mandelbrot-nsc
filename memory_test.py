import statistics
import time
import numpy as np

def benchmark(func, *args, n_runs=3):
   """Time func, return median of n_runs."""
   times = []
   for _ in range(n_runs):
      t0 = time.perf_counter()
      result = func(*args)
      times.append(time.perf_counter() - t0)
   median_t = statistics.median(times)
   print(f"Median: {median_t:.4f}s ( min ={min(times):.4f}, max ={max(times):.4f})")
   return median_t , result

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
