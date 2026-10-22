"""CSS314 Task 5: measure serial/parallel vector-operation crossover.

The NumPy arithmetic operations release Python's GIL for contiguous numeric
arrays, so ThreadPoolExecutor can execute different slices concurrently.
Pool creation, data allocation and correctness checks are outside timing.
Task submission, synchronization and the arithmetic are inside timing.
"""

import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103342
SIZES = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
WORKERS = os.cpu_count() or 1
REPEATS = 7
A = 1.75
B = -0.25


def operation(x, y):
    """Compute y[i] = A*x[i]+B, without allocating temporary arrays."""
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def slices(n, workers):
    """Give each worker a non-overlapping, near-equal contiguous slice."""
    return [(n * i // workers, n * (i + 1) // workers)
            for i in range(workers)]


def benchmark(n, executor):
    rng = np.random.default_rng(STUDENT_ID + n)
    x = rng.random(n, dtype=np.float64)
    serial_y = np.empty_like(x)
    parallel_y = np.empty_like(x)
    partitions = slices(n, WORKERS)

    def serial():
        operation(x, serial_y)

    def parallel():
        futures = [executor.submit(operation, x[start:end], parallel_y[start:end])
                   for start, end in partitions]
        for future in futures:
            future.result()

    # Warm up the NumPy execution paths and the already-created thread pool.
    serial()
    parallel()
    if not np.array_equal(serial_y, parallel_y):
        raise AssertionError(f"Mismatch between serial and parallel outputs at N={n}")

    serial_times = []
    parallel_times = []
    # Alternate order to reduce systematic measurement bias.
    for trial in range(REPEATS):
        for label, function in (("serial", serial), ("parallel", parallel)) if trial % 2 == 0 else (("parallel", parallel), ("serial", serial)):
            start = time.perf_counter_ns()
            function()
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            (serial_times if label == "serial" else parallel_times).append(elapsed_ms)

    if not np.array_equal(serial_y, parallel_y):
        raise AssertionError(f"Post-benchmark result mismatch at N={n}")
    return statistics.median(serial_times), statistics.median(parallel_times)


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"System: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical cores / parallel workers: {WORKERS}")
    print(f"Operation: y[i] = {A} * x[i] + ({B})")
    print(f"Benchmark: {REPEATS} timed repeats per mode; median wall time")
    print("Thread pool reused across measurements; dispatch and synchronization timed")
    print("All serial and parallel results verified for equality")
    print()
    print(f"{'N':>12} | {'Serial (ms)':>13} | {'Parallel (ms)':>13} | {'Speedup':>9} | {'Faster':>8}")
    print("-" * 70)

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for n in SIZES:
            serial_ms, parallel_ms = benchmark(n, executor)
            faster = "Parallel" if parallel_ms < serial_ms else "Serial" if parallel_ms > serial_ms else "Tie"
            results.append((n, serial_ms, parallel_ms, faster))
            print(f"{n:12,d} | {serial_ms:13.6f} | {parallel_ms:13.6f} | {serial_ms / parallel_ms:9.3f} | {faster:>8}")

    print("\nObserved slower-to-faster crossover interval(s):")
    transitions = [(previous[0], current[0])
                   for previous, current in zip(results, results[1:])
                   if previous[3] == "Serial" and current[3] == "Parallel"]
    if transitions:
        for lower, upper in transitions:
            print(f"  {lower:,} < N <= {upper:,} (within tested sizes)")
        print("The precise crossover N is not resolved by these discrete test sizes.")
    else:
        print("  NONE observed among the six tested sizes.")
        print("  No exact crossover point can be inferred from these measurements.")

    print("\nNote: wall times vary with machine load; rerun to assess stability.")


if __name__ == "__main__":
    main()
