from multiprocessing import Pool
import random, time, os
from functools import reduce

def monte_carlo_chunk(num_samples): 
    """Estimate pi contributions for num_samples random points"""
    inside = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside

def test_granularity(total_work, chunk_size, n_proc):
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks
    t0 = time.perf_counter()
    if n_proc == 1:
        results = [monte_carlo_chunk(s) for s in tasks]
    else:
        with Pool(processes=n_proc) as pool:
            results = pool.map(monte_carlo_chunk, tasks)
    return time.perf_counter() - t0, 4 * sum(results) / total_work

def subtract_seven(x):
    return x - 7

if __name__ =='__main__':
    total_work = 1_000_000
    n_proc = os.cpu_count() // 2
    chunk_sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    print(f"{'L':>12} | {'serial(s)':>12} | {'parallel(s)':>12}")

    for L in chunk_sizes:
        t_ser, _ = test_granularity(total_work, L, n_proc=1)
        t_par, pi = test_granularity(total_work, L, n_proc=n_proc)
        print(f"{L:12d} | {t_ser:12.4f} | {t_par:12.4f} pi={pi:.4f}")

    N = 1_000_000
    data = [random.randint(10, 100) for _ in range(N)]

    t0 = time.perf_counter()
    result_ser = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 1, map(subtract_seven, data)))
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    
    with Pool() as pool:
        mapped = pool.map(subtract_seven, data)

    result_par = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 1, mapped))
    t_parallel = time.perf_counter() - t0

    print(f"Serial:{t_serial:.4f}s result={result_ser}")
    print(f"Parallel:{t_parallel:.4f}s result={result_par}")
    print(f"Speedup:{t_serial/ t_parallel:.2f}")
    print(f"Efficiency:{t_serial/ (t_parallel * n_proc):.2f}")