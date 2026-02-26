"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""
import statistics

import numpy as np
import time
from numba import njit

def bench (fn, *args, runs=5) :
    fn (*args) # extra warm -up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

@njit
def mandelbrot_point(c, max_iter=100):
    z_n = 0j

    for n in range(max_iter):
        if z_n.real * z_n.real + z_n.imag * z_n.imag > 4.0:
            return n
        z_n = z_n * z_n + c
    return max_iter

@njit
def compute_mandelbrot_numba(x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, width=1024, height=1024, max_iter=100):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    result = np.zeros((height, width), dtype = np.int32)

    for i in range ( height ):
        for j in range ( width ):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0

            while n < max_iter and \
                z.real * z.real + z.imag * z.imag <= 4.0:
                z = z*z + c ; n += 1
            result [i, j ] = n

    return result

def compute_mandelbrot_hybrid(x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, width=1024, height=1024, max_iter=100):
    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_min, y_max, height)
    n_iterations = np.zeros((width, height), dtype=int)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            c = complex(x, y)
            n_iterations[i, j] = mandelbrot_point(c, max_iter)
    
    return n_iterations

if __name__ == "__main__":

    _ = compute_mandelbrot_numba(-2, 1, -1.5, 1.5, 64 , 64) #warm-up
    start = time.perf_counter()
    result = compute_mandelbrot_numba( -2, 1, -1.5, 1.5, 1024, 1024)
    end = time.perf_counter()

    duration = end - start
    print(f"Time: {duration:.6f} seconds")
    print(f"Result: {result}")

    #Warm up (triggers JIT compilation -- exclude from timing)
    _ = compute_mandelbrot_hybrid( -2, 1, -1.5, 1.5, 64, 64)
    _ = compute_mandelbrot_numba( -2, 1, -1.5, 1.5, 64, 64)
    t_hybrid = bench(compute_mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024)
    t_full = bench(compute_mandelbrot_numba, -2, 1 , -1.5, 1.5, 1024, 1024)
    print(f"Hybrid: { t_hybrid:.3f}s")
    print(f"Fully compiled: { t_full:.3f}s")
    print(f"Ratio: { t_hybrid / t_full:.1f}x")
