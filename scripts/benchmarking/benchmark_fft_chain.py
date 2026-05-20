"""Micro-benchmark for the FFT chain in RS: ifft2(fft2(U) * fft2(H)).

Outputs a CSV with median timings across sizes.
"""

import csv
import statistics
import time
from pathlib import Path

from diffractio import um, np
from diffractio.scalar_fields_XY import kernelRS
from scipy.fft import fft2, ifft2


DEFAULT_CSV_PATH = Path("outputs/benchmarking/fft_chain_benchmark.csv")


def benchmark_fft_chain(sizes, repeats=5, warmup=1, csv_path=DEFAULT_CSV_PATH):
    rows = []
    wavelength = 0.5 * um
    z = 500 * um
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        x0 = np.linspace(-100 * um, 100 * um, size)
        y0 = np.linspace(-100 * um, 100 * um, size)
        X, Y = np.meshgrid(x0, y0)

        U = (np.random.rand(size, size) + 1j * np.random.rand(size, size)).astype(complex)
        H = kernelRS(X, Y, wavelength, z, n=1.0, kind="z")

        for _ in range(warmup):
            ifft2(fft2(U) * fft2(H))

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            ifft2(fft2(U) * fft2(H))
            times.append(time.perf_counter() - start)

        rows.append({
            "size": size,
            "elapsed_min_s": min(times),
            "elapsed_median_s": statistics.median(times),
            "elapsed_max_s": max(times),
            "repeats": repeats,
            "warmup": warmup,
        })

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    benchmark_fft_chain(sizes=[256, 512, 1024, 2048], repeats=5, warmup=1)
