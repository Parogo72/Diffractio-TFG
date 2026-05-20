import csv
import statistics
import time
from pathlib import Path

from tfg_profiler.outputs import sanitize_name


def resolve_profile_item(item, size_value=None):
    """Resolve a profile item to (label, func, args, kwargs)."""
    if isinstance(item, dict):
        label = item["label"]
        if "setup" in item and callable(item["setup"]):
            resolved = item["setup"](size_value)
            func = resolved["func"]
            args = resolved.get("args", ())
            kwargs = resolved.get("kwargs", {})
        else:
            func = item["func"]
            args = item.get("args", ())
            kwargs = item.get("kwargs", {})
    else:
        label, func, args, kwargs = item
    return label, func, args, kwargs


def benchmark_suite(
    profiles,
    repeats=5,
    warmup=1,
    csv_path="benchmark_results.csv",
    size_value=None,
    write_csv=True,
):
    """Benchmark a list of profiles and optionally write a CSV report."""
    rows = []
    for item in profiles:
        label = item["label"] if isinstance(item, dict) and "label" in item else str(item)
        try:
            for _ in range(warmup):
                label, func, args, kwargs = resolve_profile_item(item, size_value=size_value)
                func(*args, **kwargs)

            elapsed_list = []
            for _ in range(repeats):
                label, func, args, kwargs = resolve_profile_item(item, size_value=size_value)
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                elapsed_list.append(elapsed)
        except Exception as exc:
            print(f"\n=== Benchmark FAILED: {label} (size={size_value}) ===")
            print(f"Reason: {type(exc).__name__}: {exc}")
            continue

        median_elapsed = statistics.median(elapsed_list)
        min_elapsed = min(elapsed_list)
        max_elapsed = max(elapsed_list)

        print(f"\n=== Benchmark: {label} ===")
        print(f"Repeats: {repeats} (warmup {warmup})")
        print(f"Elapsed min/med/max: {min_elapsed:.6f} / {median_elapsed:.6f} / {max_elapsed:.6f} s")

        rows.append({
            "label": label,
            "size": size_value,
            "repeats": repeats,
            "warmup": warmup,
            "elapsed_min_s": min_elapsed,
            "elapsed_median_s": median_elapsed,
            "elapsed_max_s": max_elapsed,
        })

    if write_csv and rows:
        target = Path(csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nBenchmark results written to: {target}")
    return rows


def benchmark_sweep(
    profiles,
    size_sweep,
    repeats=5,
    warmup=1,
    csv_path="benchmark_sweep_results.csv",
):
    """Run benchmark_suite for each size in a sweep and write a CSV report."""
    rows = []
    for size_value in size_sweep:
        sweep_rows = benchmark_suite(
            profiles,
            repeats=repeats,
            warmup=warmup,
            csv_path=csv_path,
            size_value=size_value,
            write_csv=False,
        )
        rows.extend(sweep_rows)

    if rows:
        target = Path(csv_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSweep results written to: {target}")
    return rows


def write_combined_csv(rows, csv_path="benchmark_all.csv"):
    """Write combined benchmark rows to a CSV file."""
    if not rows:
        return
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCombined results written to: {target}")


def plot_scaling(rows, output_dir="plots", file_suffix=None):
    """Plot scaling curves per label and save PNGs into output_dir."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available; skipping plots.")
        return

    if not rows:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_label = {}
    for row in rows:
        label = row.get("label")
        size = row.get("size")
        median = row.get("elapsed_median_s")
        if size is None or median is None:
            continue
        by_label.setdefault(label, []).append((size, median))

    for label, points in by_label.items():
        points.sort(key=lambda x: x[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.title(label)
        plt.xlabel("Size")
        plt.ylabel("Median time (s)")
        plt.grid(True, alpha=0.3)

        stem = sanitize_name(label)
        if file_suffix:
            stem = f"{stem}_{file_suffix}"

        plt.savefig(out / f"{stem}.png", dpi=150)
        plt.close()
