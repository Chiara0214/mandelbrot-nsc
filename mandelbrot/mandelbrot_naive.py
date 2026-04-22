"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""
from tracemalloc import start
import numpy as np
import time
import matplotlib.pyplot as plt

#os.environ['LINE_PROFILE'] = '1'

def mandelbrot_point(c: complex, max_iter: int = 100) -> int:
    """
    Compute the escape iteration count for a single complex point

    Parameters
    ----------
    c : complex
        Complex coordinate to test
    max_iter : int, optional
        Maximum number of iterations to perform

    Returns
    -------
    int
        Iteration count at which the trajectory escapes, or max_iter if it stays bounded within the iteration limit
    """
    z_n = 0j

    for n in range(max_iter):
        z_n = z_n**2 + c
        if abs(z_n) > 2:
            return n

    return max_iter

#@line_profiler.profile
def compute_mandelbrot(
    x_min: float = -2.0,
    x_max: float = 1.0,
    y_min: float = -1.5,
    y_max: float = 1.5,
    width: int = 1024,
    height: int = 1024,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Compute Mandelbrot escape counts on a rectangular grid

    Parameters
    ----------
    x_min : float
        Minimum real value in the sampled region
    x_max : float
        Maximum real value in the sampled region
    y_min : float
        Minimum imaginary value in the sampled region
    y_max : float
        Maximum imaginary value in the sampled region
    width : int
        Number of grid points along the real axis
    height : int
        Number of grid points along the imaginary axis
    max_iter : int, optional
        Maximum number of iterations used for each point

    Returns
    -------
    np.ndarray
        Two-dimensional array of escape iteration counts for the sampled grid points
    """

    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_min, y_max, height)
    n_iterations = np.zeros((width, height), dtype=int)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            c = complex(x, y)
            n_iterations[i, j] = mandelbrot_point(c, max_iter)
    
    return n_iterations

if __name__ == "__main__":

    #testing the function, returns 100 = max_iter
    print(mandelbrot_point(0))

    start = time.time()

    n_iterations = compute_mandelbrot()

    elapsed = time.time() - start
    print(f"Computation took {elapsed:.3f} seconds")

    plt.imshow(n_iterations, extent=(-2, 1, -1.5, 1.5), cmap='twilight', origin='lower')
    plt.colorbar()
    plt.title('Mandelbrot')
    plt.show()