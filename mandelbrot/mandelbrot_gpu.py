import pyopencl as cl
import numpy as np
import time, matplotlib.pyplot as plt
import time
import statistics
import matplotlib.pyplot as plt

KERNEL_SRC_F32 = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

KERNEL_SRC_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0, zi = 0.0;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""
    
def run_mandelbrot_gpu(kernel, ctx, queue, N, dtype_func):
    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    kernel(queue, (N, N), None, image_dev, dtype_func(X_MIN), dtype_func(X_MAX), 
           dtype_func(Y_MIN), dtype_func(Y_MAX), np.int32(N), np.int32(MAX_ITER))
    queue.finish()

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    return image

def timed(fn, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        queue.finish()  # queue.finish() waits for GPU
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

if __name__ == "__main__":
        
    pl_id = 0
    dev_id = 0
    platform = cl.get_platforms()
    my_devices = [platform[pl_id].get_devices()[dev_id]]
    ctx = cl.Context(devices=my_devices)
    queue = cl.CommandQueue(ctx)

    N, MAX_ITER = 1024, 1024
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    result_host = np.zeros((N, N), dtype=np.int32)
    result_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result_host.nbytes)

    # Float32
    prog_f32 = cl.Program(ctx, KERNEL_SRC_F32).build()
    kernel_f32 = cl.Kernel(prog_f32, "mandelbrot")

    img32 = run_mandelbrot_gpu(kernel_f32, ctx, queue, N, np.float32)

    bench_f32 = timed(lambda: kernel_f32(queue, (N, N), None, result_dev, 
                                            np.float32(X_MIN), np.float32(X_MAX), 
                                            np.float32(Y_MIN), np.float32(Y_MAX), 
                                            np.int32(N), np.int32(MAX_ITER)))
    
    print(f"GPU {N}x{N} f32: {bench_f32*1e3:.3f} ms")
    plt.imshow(img32, cmap='hot', origin='lower'); plt.axis('off')
    plt.savefig("mandelbrot_gpu_f32.png", dpi=150, bbox_inches='tight')


    # Float64
    # Check fp64 support
    bench_f64 = None
    dev = ctx.devices[0]
    if "cl_khr_fp64" not in dev.extensions:
        print("No native fp64 -- Apple Silicon: emulated, expect large slowdown")
    else:
        prog_f64 = cl.Program(ctx, KERNEL_SRC_F64).build()
        kernel_f64 = cl.Kernel(prog_f64, "mandelbrot_f64")

        img64 = run_mandelbrot_gpu(kernel_f64, ctx, queue, N, np.float64)

        bench_f64 = timed(lambda: kernel_f64(queue, (N, N), None, result_dev, 
                                                      np.float64(X_MIN), np.float64(X_MAX), 
                                                      np.float64(Y_MIN), np.float64(Y_MAX), 
                                                      np.int32(N), np.int32(MAX_ITER)))
        
        print(f"GPU {N}x{N} f64: {bench_f64*1e3:.3f} ms")
        plt.imshow(img64, cmap='hot', origin='lower'); plt.axis('off')
        plt.savefig("mandelbrot_gpu_f64.png", dpi=150, bbox_inches='tight')

    # Bar chart
    results = {
            "naive Python": 6.797,
            "NumPy": 1.439,
            "Numba f32": 0.065,
            "Numba f64": 0.063,
            "multiprocessing": 0.013,
            "Dask local": 0.077,
            "Dask cluster": 0.084,
            "GPU f32": bench_f32,
            "GPU f64": bench_f64
    }

    names, times = zip(*results.items())

    plt.figure(figsize=(10, 6))
    plt.bar(names, times, log=True)
    plt.ylabel("seconds (log scale)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("benchmark_mp3.png", dpi=150)