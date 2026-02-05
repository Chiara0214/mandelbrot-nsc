"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""
import numpy as np
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

def compute_mandelbrot():
    """
    Function that returns a list with the number of iterations for each point

    Returns
    -------
    list of int
        Output value
    """
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5

    x_values = np.linspace(x_min, x_max, 100)
    y_values = np.linspace(y_min, y_max, 100)
    n_iterations = []

    for x in x_values:
        for y in y_values:
            c = complex(x, y)
            n_iterations.append(mandelbrot_point(c))
    
    return n_iterations

if __name__ == "__main__":

    #testing the function, returns 100 = max_iter
    print(mandelbrot_point(0))

    n_iterations = compute_mandelbrot()

        