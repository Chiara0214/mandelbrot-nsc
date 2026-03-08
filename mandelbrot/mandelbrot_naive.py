"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""
import os
from tracemalloc import start
import line_profiler
import numpy as np
import time
import matplotlib.pyplot as plt

#os.environ['LINE_PROFILE'] = '1'

def mandelbrot_point(c, max_iter=100):
    """
    Function that takes a complex number c as input and returns the number of iterations

    Parameters
    ----------
    c : complex
        Input value

    Returns
    -------
    int
        Output value
    """
    z_n = 0
    max_iter = 100

    for n in range(max_iter):
        z_n = z_n**2 + c
        if abs(z_n) > 2:
            return n

    return max_iter

#@line_profiler.profile
def compute_mandelbrot(x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, width=1024, height=1024, max_iter=100):
    """
    Function that returns a list with the number of iterations for each point given a region and a resolution

    Parameters
    ----------
    x_min : float
    x_max : float
    y_min : float
    y_max : float
    width : int
    height : int

    Returns
    -------
    list of int
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