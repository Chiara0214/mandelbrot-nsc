"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import bench

def mandelbrot_point(C: np.ndarray, max_iter: int) -> np.ndarray:
    """
    Computes the escape iteration for a grid of complex points using vectorized operations

    Parameters
    ----------
    C : np.ndarray
        A 2D numpy array of complex numbers representing the grid coordinates
    max_iter : int
        The maximum number of iterations before assuming the point is in the set

    Returns
    -------
    np.ndarray
        A 2D numpy array of integers containing the iteration count for each point
    """
    Z = np.zeros(C.shape, dtype=complex)
    M = np.zeros(C.shape, dtype=int) # iterations

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask] # updates z for all complex numbers with abs<=2
        M[mask] += 1 # updates iterations for all complex numbers with abs<=2

    return M

def compute_mandelbrot(
    x_min: float = -2.0, 
    x_max: float = 1.0, 
    y_min: float = -1.5, 
    y_max: float = 1.5, 
    width: int = 1024, 
    height: int = 1024, 
    max_iter: int = 100
) -> np.ndarray:
    """
    Generates the Mandelbrot set for a specified spatial region and grid resolution

    Parameters
    ----------
    x_min : float, optional
        The minimum real coordinate, by default -2.0
    x_max : float, optional
        The maximum real coordinate, by default 1.0
    y_min : float, optional
        The minimum imaginary coordinate, by default -1.5
    y_max : float, optional
        The maximum imaginary coordinate, by default 1.5
    width : int, optional
        The number of grid points along the real axis, by default 1024
    height : int, optional
        The number of grid points along the imaginary axis, by default 1024
    max_iter : int, optional
        The maximum number of iterations to compute, by default 100

    Returns
    -------
    np.ndarray
        A 2D array of iteration counts corresponding to each pixel in the grid
    """
    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x_values, y_values)
    C = X + 1j * Y

    print(f"Shape: {C.shape}") # (height, width)
    print(f"Type: {C.dtype}") 

    n_iterations = mandelbrot_point(C, max_iter)
    
    return n_iterations

if __name__ == "__main__":

    n_iterations = compute_mandelbrot()

    plt.imshow(n_iterations, extent=(-2, 1, -1.5, 1.5), cmap='twilight', origin='lower')
    plt.colorbar()
    plt.title('Mandelbrot')
    plt.show()

    # plot of runtime comparisons with different grid sizes
    grid_sizes = [256, 512, 1024, 2048, 4096]
    times = []

    for size in grid_sizes:
        t = bench(compute_mandelbrot, -2, 1, -1.5, 1.5, size, size, 100)
        times.append(t)
    
    plt.plot(grid_sizes, times)
    plt.xlabel('Grid size (NxN)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime plot')
    plt.show()