import time
import statistics
import dask
import numpy as np
from dask import delayed
from dask.distributed import Client
#from dask.distributed import LocalCluster
from numba import njit


@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """
    Computes the escape iteration for a single complex point

    Parameters
    ----------
    c_real : float
        The real part of the complex coordinate
    c_imag : float
        The imaginary part of the complex coordinate
    max_iter : int
        The maximum number of iterations before assuming the point is in the set

    Returns
    -------
    int
        The number of iterations taken to escape, or max_iter if it didn't escape
    """
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0:
            return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter


@njit(cache=True)
def mandelbrot_chunk(
    row_start: int, 
    row_end: int, 
    N: int, 
    x_min: float, 
    x_max: float, 
    y_min: float, 
    y_max: float, 
    max_iter: int
) -> np.ndarray:
    """
    Computes a horizontal strip (chunk) of the Mandelbrot set

    Parameters
    ----------
    row_start : int
        The starting row index of the chunk
    row_end : int
        The ending row index (exclusive) of the chunk
    N : int
        The total grid resolution (N x N)
    x_min : float
        The minimum real coordinate
    x_max : float
        The maximum real coordinate
    y_min : float
        The minimum imaginary coordinate
    y_max : float
        The maximum imaginary coordinate
    max_iter : int
        The maximum number of iterations

    Returns
    -------
    np.ndarray
        A 2D array of integers containing the iteration counts for the chunk
    """
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out


def mandelbrot_dask(
    N: int, 
    x_min: float, 
    x_max: float, 
    y_min: float, 
    y_max: float, 
    max_iter: int = 100, 
    n_chunks: int = 32
) -> np.ndarray:
    """
    Computes the Mandelbrot set in parallel using Dask delayed tasks

    Parameters
    ----------
    N : int
        The grid resolution (N x N)
    x_min : float
        The minimum real coordinate
    x_max : float
        The maximum real coordinate
    y_min : float
        The minimum imaginary coordinate
    y_max : float
        The maximum imaginary coordinate
    max_iter : int, optional
        The maximum number of iterations, by default 100
    n_chunks : int, optional
        The total number of row chunks to divide the workload into, by default 32

    Returns
    -------
    np.ndarray
        A 2D array representing the full Mandelbrot set
    """
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(row, row_end, N, x_min, x_max, y_min, y_max, max_iter)
        )
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    n_workers = 12
    #cluster = LocalCluster(n_workers, threads_per_worker=1)
    #client = Client(cluster)
    client = Client("tcp://10.92.1.201:8786")

    #mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter) # warm up JIT
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    
    # Serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")


    n_chunks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX,
        # warm up all workers
        Y_MIN, Y_MAX, 10))
    for n in n_chunks:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n)
            times.append(time.perf_counter() - t0)
        t = statistics.median(times)
        print(f"Dask local (n_chunks={n}):{t:.3f}s, LIF: {n_workers * t / t_serial - 1:.2f}")

    client.close() # cluster.close()