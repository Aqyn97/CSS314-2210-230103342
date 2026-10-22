"""Task 2: parallel array summation and Amdahl's Law scaling."""

import multiprocessing as mp
import os
import platform
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

STUDENT_ID = 230103342
ARRAY_SIZE = 20_000_000
PARALLEL_FRACTION = 0.95


def sum_chunk(args):
    """Sum one disjoint portion of the same read-only memory-mapped array."""
    path, start, stop = args
    values = np.memmap(path, dtype=np.float64, mode="r", shape=(ARRAY_SIZE,))
    return float(np.sum(values[start:stop], dtype=np.float64))


def parallel_sum(path, workers):
    """Include worker creation, data processing, and result collection in time."""
    chunks = [
        (path, ARRAY_SIZE * i // workers, ARRAY_SIZE * (i + 1) // workers)
        for i in range(workers)
    ]
    start_time = time.perf_counter()
    if workers == 1:
        results = [sum_chunk(chunks[0])]
    else:
        # Processes achieve CPU parallelism without Python's GIL limitation.
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(sum_chunk, chunks))
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return sum(results), elapsed_ms


def amdahl_bound(workers):
    return 1.0 / ((1.0 - PARALLEL_FRACTION) + PARALLEL_FRACTION / workers)


def main():
    logical_cores = os.cpu_count() or 1
    worker_counts = list(dict.fromkeys([1, 2, 4, logical_cores, logical_cores * 2]))
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Logical cores detected: {logical_cores}")
    print(f"Array size: {ARRAY_SIZE:,} float64 elements")
    print(f"Random seed: {STUDENT_ID}")
    print("Parallel model: multiprocessing (processes)")
    print("Timing includes process startup, summation, and aggregation")
    print("Amdahl parallel fraction (P): 0.95")
    print()
    print(f"{'Cores':>7} | {'Wall Time (ms)':>14} | {'Speedup':>9} | {'Amdahl Upper Bound':>18}")
    print("-" * 59)

    # A disk-backed mmap avoids copying 160 MB into every worker process.
    with tempfile.TemporaryDirectory(prefix="css314_task2_") as temp_dir:
        path = os.path.join(temp_dir, "values.bin")
        values = np.memmap(path, dtype=np.float64, mode="w+", shape=(ARRAY_SIZE,))
        rng = np.random.default_rng(STUDENT_ID)
        # Fill in batches to avoid an additional 160 MB temporary array.
        batch = 1_000_000
        for start in range(0, ARRAY_SIZE, batch):
            stop = min(start + batch, ARRAY_SIZE)
            values[start:stop] = rng.random(stop - start, dtype=np.float64)
        values.flush()
        del values

        baseline_ms = None
        baseline_sum = None
        for workers in worker_counts:
            result, wall_ms = parallel_sum(path, workers)
            if baseline_ms is None:
                baseline_ms = wall_ms
                baseline_sum = result
            # Different summation orders have normal floating-point rounding.
            if not np.isclose(result, baseline_sum, rtol=1e-12, atol=1e-6):
                raise ValueError(f"Incorrect sum for {workers} workers: {result}")
            speedup = baseline_ms / wall_ms
            print(f"{workers:7d} | {wall_ms:14.3f} | {speedup:9.3f} | {amdahl_bound(workers):18.3f}", flush=True)
        print()
        print(f"Baseline sum: {baseline_sum:.10f}")
        print("All worker-count sums verified against the 1-worker result.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
