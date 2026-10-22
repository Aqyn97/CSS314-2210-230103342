import os
import platform
import subprocess
import time

import numpy as np

STUDENT_ID = 230103342
ARRAY_SIZE = 20_000_000


def cpu_model():
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"Apple Silicon ({result.stdout.strip()})"
    return platform.processor() or platform.machine()


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Array size: {ARRAY_SIZE:,}")
    print("Data type: float64")
    print(f"Random seed: {STUDENT_ID}")
    print("Execution: serial (single thread)")

    rng = np.random.default_rng(STUDENT_ID)
    values = rng.random(ARRAY_SIZE, dtype=np.float64)

    start = time.perf_counter()
    total = 0.0
    mean = 0.0
    m2 = 0.0

    # Welford's online algorithm, executed in one Python thread.
    for count, value in enumerate(values, start=1):
        x = float(value)
        total += x
        delta = x - mean
        mean += delta / count
        m2 += delta * (x - mean)

    std_dev = (m2 / ARRAY_SIZE) ** 0.5  # Population standard deviation
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Sum: {total:.10f}")
    print(f"Standard deviation: {std_dev:.10f}")
    print(f"Wall-clock time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
