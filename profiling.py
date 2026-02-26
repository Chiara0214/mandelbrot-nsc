from mandelbrot_naive import compute_mandelbrot as mb_naive
from mandelbrot_numpy import compute_mandelbrot as mb_numpy

import cProfile , pstats

if __name__ == "__main__":

    cProfile.run('mb_naive(-2, 1, -1.5, 1.5, 512, 512)', 'naive_profile.prof')
    cProfile.run('mb_numpy(-2, 1, -1.5, 1.5, 512, 512)', 'numpy_profile.prof')

    for name in ('naive_profile.prof', 'numpy_profile.prof'):
        stats = pstats.Stats(name)
        stats.sort_stats('cumulative')
        stats.print_stats(10)