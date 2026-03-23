import dask, random, time, statistics
from dask import delayed
from dask.distributed import Client, LocalCluster

def monte_carlo_chunk(n_samples):
    inside = 0
    for _ in range(n_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside

if __name__ == '__main__':
    total, n_chunks = 1_000_000, 8
    samples = total // n_chunks 

    # Serial baseline
    t0 = time.perf_counter()
    results = [monte_carlo_chunk(samples) for
    _ in range(n_chunks)]
    t_serial = time.perf_counter() - t0
    print(f"Serial: {t_serial:.3f}s pi={4*sum(results)/total:.4f}")

    # Dask delayed -- task graph is built, not executed yet
    tasks = [delayed(monte_carlo_chunk)(samples) for
    _ in range(n_chunks)]
    t0 = time.perf_counter()
    results = dask.compute(*tasks)
    t_dask = time.perf_counter() - t0
    print(f"Dask: {t_dask:.3f}s pi={4*sum(results)/total:.4f}")

    # Visualise (requires: conda install python-graphviz)
    #dask.visualize(*tasks, filename='task_graph.png')

    # Create local cluster; start with max workers -- scale() adjusts without restarting
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dashboard: {client.dashboard_link}")
    # --> open the printed URL in your browser

    # Rerun E1 tasks; LocalCluster scheduler takes over
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    results = dask.compute(*tasks)

    # Vary n_workers: scale() resizes without restarting the scheduler
    # (recreating LocalCluster while the browser is open breaks the dashboard)
    cluster.scale(4); client.wait_for_workers(4) # now 4 workers
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    results = dask.compute(*tasks)
    
    client.close()
    cluster.close()