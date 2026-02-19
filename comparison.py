import numpy as np

from mandelbrot_naive import compute_mandelbrot as mb_naive
from mandelbrot_numpy import compute_mandelbrot as mb_numpy
import time, statistics

def benchmark(func, * args, n_runs=3) :
    """ Time func, return median of n_runs. """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    median_t = statistics.median(times)
    print (f"Median: {median_t:.4f}s (min={min(times):.4f}, max={max(times):.4f})")

    return median_t, result

if __name__ == "__main__":
    t_naive, M_naive = benchmark(mb_naive, -2, 1, -1.5, 1.5, 1024, 1024, 100)
    t_numpy, M_numpy = benchmark(mb_numpy, -2, 1, -1.5, 1.5, 1024, 1024, 100)

    if np.allclose(M_naive, M_numpy):
        print("Results match!")
    else:
        print("Results differ!")

        diff = np.abs(M_naive - M_numpy)
        print(f"Max difference: {diff.max()}")
        print(f"Different pixels: {(diff > 0).sum()}")