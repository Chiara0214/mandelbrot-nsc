"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_point(C, max_iter):
    """
    Function that takes np.meshgrid C of complex numbers as input and returns meshgrid M with number of iterations

    Parameters
    ----------
    C : np.meshgrid

    Returns
    -------
    M : np.meshgrid
    """
    Z = np.zeros(C.shape, dtype=complex)
    M = np.zeros(C.shape, dtype=int) #iterations

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask] #updates z for all complex numbers with abs<=2
        M[mask] += 1 #updates iterations for all complex numbers with abs<=2

    return M

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
    X, Y = np.meshgrid(x_values, y_values)
    C = X + 1j*Y

    print (f"Shape: {C.shape }") # (1024, 1024)
    print (f"Type: {C.dtype }") 

    n_iterations = mandelbrot_point(C, max_iter)
    
    return n_iterations

if __name__ == "__main__":

    n_iterations = compute_mandelbrot()

    plt.imshow(n_iterations, extent=(-2, 1, -1.5, 1.5), cmap='twilight', origin='lower')
    plt.colorbar()
    plt.title('Mandelbrot')
    plt.show()
    plt.savefig('mandelbrot.png')     