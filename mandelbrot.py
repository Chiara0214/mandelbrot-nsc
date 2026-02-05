"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""
from tracemalloc import start
import numpy as np
import time
import matplotlib.pyplot as plt

def mandelbrot_point(c):
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

def compute_mandelbrot(x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, width=1024, height=1024):
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
    n_iterations = []

    for x in x_values:
        for y in y_values:
            c = complex(x, y)
            n_iterations.append(mandelbrot_point(c))
    
    return n_iterations

if __name__ == "__main__":

    #testing the function, returns 100 = max_iter
    print(mandelbrot_point(0))

    start = time.time()

    n_iterations = compute_mandelbrot()

    elapsed = time.time() - start
    print(f"Computation took {elapsed:.3f} seconds")


        