"""CSS314 Task 4: static vs dynamic scheduling under irregular workload."""

import os
import platform
import queue
import threading
import time

STUDENT_ID = "230103342"
TASK_COUNT = 1_000
WORKERS = 4
DYNAMIC_CHUNK_SIZE = 5
SLEEP_FACTOR_SECONDS = 0.0001


def process_item(i: int) -> None:
    """The required non-uniform workload (sleep releases Python's GIL)."""
    time.sleep(SLEEP_FACTOR_SECONDS * (i % 20))


def benchmark(strategy: str):
    completed_at = [0.0] * WORKERS
    processed_counts = [0] * WORKERS
    errors = []
    start_gate = threading.Barrier(WORKERS + 1)
    work_queue = queue.Queue()

    if strategy == "dynamic":
        for start in range(0, TASK_COUNT, DYNAMIC_CHUNK_SIZE):
            work_queue.put(range(start, min(start + DYNAMIC_CHUNK_SIZE, TASK_COUNT)))

    def worker(worker_id: int):
        try:
            start_gate.wait()
            if strategy == "static":
                block_size = (TASK_COUNT + WORKERS - 1) // WORKERS
                begin = worker_id * block_size
                finish = min(begin + block_size, TASK_COUNT)
                for i in range(begin, finish):
                    process_item(i)
                    processed_counts[worker_id] += 1
            else:
                while True:
                    try:
                        chunk = work_queue.get_nowait()
                    except queue.Empty:
                        break
                    for i in chunk:
                        process_item(i)
                        processed_counts[worker_id] += 1
            completed_at[worker_id] = time.perf_counter()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(j,)) for j in range(WORKERS)]
    for thread in threads:
        thread.start()

    # Threads are ready at this common gate before timing starts.
    start = time.perf_counter()
    start_gate.wait()
    for thread in threads:
        thread.join()
    end = time.perf_counter()

    if errors:
        raise RuntimeError(f"Worker failed: {errors[0]}")
    if sum(processed_counts) != TASK_COUNT:
        raise RuntimeError(f"Expected {TASK_COUNT} tasks, processed {sum(processed_counts)}")

    return {
        "wall_ms": (end - start) * 1_000,
        "first_ms": (min(completed_at) - start) * 1_000,
        "last_ms": (max(completed_at) - start) * 1_000,
        "delta_ms": (max(completed_at) - min(completed_at)) * 1_000,
        "counts": processed_counts,
    }


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Items: {TASK_COUNT} (0 through {TASK_COUNT - 1})")
    print(f"Workers: {WORKERS} threads")
    print("Workload: sleep(0.0001 * (i % 20))")
    print(f"Dynamic queue chunk size: {DYNAMIC_CHUNK_SIZE}")
    print()
    static = benchmark("static")
    dynamic = benchmark("dynamic")
    print(f"{'Strategy':<24} {'Wall (ms)':>12} {'First (ms)':>13} {'Last (ms)':>12} {'Finish delta (ms)':>19}")
    print("-" * 86)
    for name, result in (("Static (250 per worker)", static), ("Dynamic queue (chunk=5)", dynamic)):
        print(f"{name:<24} {result['wall_ms']:>12.3f} {result['first_ms']:>13.3f} {result['last_ms']:>12.3f} {result['delta_ms']:>19.3f}")
    print()
    print(f"Static tasks per worker:  {static['counts']}")
    print(f"Dynamic tasks per worker: {dynamic['counts']}")
    print(f"Wall time ratio (Static / Dynamic): {static['wall_ms'] / dynamic['wall_ms']:.3f}x")
    print("Validation: all 1,000 items processed once per strategy.")
    print("Note: The workload repeats every 20 items; 250-item static blocks split cycles unevenly.")
    print("Scheduling and OS noise may make dynamic scheduling slower despite load balancing.")


if __name__ == "__main__":
    main()
