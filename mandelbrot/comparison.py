from mandelbrot_naive import compute_mandelbrot as mb_naive
from mandelbrot_numpy import compute_mandelbrot as mb_numpy  
from mandelbrot_numba import compute_mandelbrot_numba as mb_numba
from utils import bench

if __name__ == "__main__":

    width, height = 1024, 1024
    args = (-2, 1, -1.5, 1.5, width, height, 100)
    t_naive = bench(mb_naive, *args)
    t_numpy = bench(mb_numpy, *args)
    t_numba = bench(mb_numba, *args)
    print (f"Naive: { t_naive:.3f}s")
    print (f"NumPy: { t_numpy:.3f}s ({ t_naive / t_numpy:.1f}x)")
    print (f"Numba: { t_numba:.3f}s ({ t_naive / t_numba:.1f}x)")