"""
Mandelbrot Set Generator
Author : Chiara Caselli
Course : Numerical Scientific Computing 2026
"""

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

if __name__ == "__main__":

    #testing the function, returns 100 = max_iter
    print(mandelbrot_point(0))
        