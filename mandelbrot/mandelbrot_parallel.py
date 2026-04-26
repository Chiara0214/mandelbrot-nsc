import os
import time
import statistics
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool


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
    Computes a horizontal chunk of the Mandelbrot set

    Parameters
    ----------
    row_start : int
        The starting row index of the chunk
    row_end : int
        The ending row index of the chunk
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


def mandelbrot_serial(
    N: int, 
    x_min: float, 
    x_max: float, 
    y_min: float, 
    y_max: float, 
    max_iter: int = 100
) -> np.ndarray:
    """
    Computes the entire Mandelbrot set sequentially on a single core

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

    Returns
    -------
    np.ndarray
        A 2D array representing the full Mandelbrot set
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args: Tuple[int, int, int, float, float, float, float, int]) -> np.ndarray:
    """
    Helper function to unpack arguments for multiprocessing pool.map

    Parameters
    ----------
    args : tuple
        A tuple containing all arguments required for mandelbrot_chunk

    Returns
    -------
    np.ndarray
        The computed chunk
    """
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    N: int, 
    x_min: float, 
    x_max: float, 
    y_min: float, 
    y_max: float, 
    max_iter: int = 100, 
    n_workers: int = 4, 
    n_chunks: Optional[int] = None, 
    pool = None
) -> np.ndarray:
    """
    Computes the Mandelbrot set in parallel using multiprocessing

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
    n_workers : int, optional
        The number of CPU worker processes to use, by default 4
    n_chunks : int, optional
        The total number of chunks to divide the workload into, by default None (equals n_workers)
    pool : multiprocessing.Pool, optional
        An existing multiprocessing Pool instance, by default None

    Returns
    -------
    np.ndarray
        A 2D array representing the full Mandelbrot set
    """
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
        
    if pool is not None:  # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))
        
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as pool_inst:
        pool_inst.map(_worker, tiny)  # warm-up: load JIT cache in workers
        parts = pool_inst.map(_worker, chunks)
        
    return np.vstack(parts)


if __name__ == '__main__':
    result = mandelbrot_parallel(1024, -2.5, 1.0, -1.25, 1.25, n_workers=4)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[-2.5, 1.0, -1.25, 1.25],
              cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')

    out = Path(__file__).parent.parent / 'outputs' / 'mandelbrot.png'
    out.parent.mkdir(parents=True, exist_ok=True)  # Added to ensure dir exists
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')

    # --- MP2 M3: benchmark (in __main__ block) ---

    N, max_iter = 8192, 100
    n_workers = 12  # adjust to your L04 optimum
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)  # warm up JIT
    
    # Serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")

    # Worker-count sweep
    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
        with Pool(processes=n_workers) as pool_inst:
            pool_inst.map(_worker, chunks)  # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool_inst.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")
    
    # Chunk-count sweep (M2): one Pool per config
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers
        with Pool(processes=n_workers) as pool_inst:
            pool_inst.map(_worker, tiny)  # warm-up: load JIT cache in workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=n_workers, n_chunks=n_chunks, pool=pool_inst)
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")