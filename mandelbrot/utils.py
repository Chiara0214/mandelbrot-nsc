import statistics
import time

def bench(fn, *args, runs=5) :
    fn(*args) # warm-up
    times = []
    for _ in range(runs) :
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)