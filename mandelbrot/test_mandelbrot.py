import pytest
import numpy as np

from mandelbrot_parallel import mandelbrot_pixel, mandelbrot_serial, mandelbrot_parallel

#Test 1: verify that individual pixels calculate the correct escape iteration for known cases
KNOWN_CASES = [
    (0.0, 0.0, 100, 100),   #origin, never escapes, returns max_iter
    (5.0, 0.0, 100, 1),     #far outside, escapes on iteration 1
    (-2.5, 0.0, 100, 1),    #left tip, escapes on iteration 1
]

@pytest.mark.parametrize("c_real, c_imag, max_iter, expected", KNOWN_CASES)
def test_pixel(c_real, c_imag, max_iter, expected):
    result = mandelbrot_pixel(c_real, c_imag, max_iter)
    assert result == expected

#Test 2: result is always in [0, max_iter]
#Checks 200 random points with real and imaginary parts between -3 and 3
def test_result_in_range():
    max_iter = 100
    reals = np.random.uniform(-3, 3, 200)
    imags = np.random.uniform(-3, 3, 200)
    
    for r, i in zip(reals, imags):
        res = mandelbrot_pixel(r, i, max_iter)
        assert 0 <= res <= max_iter

#Test 3: test that the parallel implementation matches serial on a small grid
def test_parallel_match_serial():
    N = 32  #small grid
    max_iter = 50
    
    serial_result = mandelbrot_serial(N, -2.0, 1.0, -1.5, 1.5, max_iter)
    parallel_result = mandelbrot_parallel(N, -2.0, 1.0, -1.5, 1.5, max_iter, n_workers=2)
    
    np.testing.assert_array_equal(parallel_result, serial_result)