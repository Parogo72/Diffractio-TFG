#!/usr/bin/env python3
"""
Plot CPU vs GPU elapsed time by input size for each family and report crossover points.

Usage:
    python scripts/plotting/plot_crossover_by_size.py \
    --cpu-csv outputs/benchmarking/benchmark_all_YYYYMMDD_HHMMSS.csv \
    --gpu-csv outputs/benchmarking/benchmark_all_diffractio_gpu2_YYYYMMDD_HHMMSS.csv \
    --out outputs/plots/cpu_gpu_crossover_by_size.png
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def size_rows(rows):
    return [r for r in rows if r.get("size") not in ("", None)]


def family_from_label(label: str) -> str:
    match = re.match(r"^RS(?:_[A-Za-z0-9_]+)?\s+([A-Za-z0-9_]+)", label)
    return match.group(1) if match else label


def build_size_time_map(rows):
    data = {}
    for row in size_rows(rows):
        family = family_from_label(row["label"])
        size = int(float(row["size"]))
        elapsed = float(row["elapsed_median_s"])
        data.setdefault(family, {})[size] = elapsed
    return data


def first_gpu_better_size(common_sizes, cpu_times, gpu_times):
    for size in common_sizes:
        if gpu_times[size] < cpu_times[size]:
            return size
    return None


def plot_family_curves(cpu_map, gpu_map, out_png: Path):
    families = sorted([family for family in cpu_map if family in gpu_map])
    if not families:
        raise RuntimeError("No common families with sweep sizes found between CPU and GPU CSV files.")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    axes_flat = axes.flatten()
    crossover_report = []

    for idx, family in enumerate(families[:4]):
        axis = axes_flat[idx]
        cpu_times = cpu_map[family]
        gpu_times = gpu_map[family]
        common_sizes = sorted([size for size in cpu_times if size in gpu_times])
        if not common_sizes:
            axis.set_title(f"{family} (no common sizes)")
            axis.axis("off")
            continue

        cpu_y = [cpu_times[size] for size in common_sizes]
        gpu_y = [gpu_times[size] for size in common_sizes]

        axis.plot(common_sizes, cpu_y, marker="o", linewidth=1.8, label="CPU", color="#1d3557")
        axis.plot(common_sizes, gpu_y, marker="s", linewidth=1.8, label="GPU2", color="#e63946")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(f"{family}: elapsed median by size")
        axis.set_xlabel("Input size")
        axis.set_ylabel("Elapsed median (s)")

        crossover = first_gpu_better_size(common_sizes, cpu_times, gpu_times)
        if crossover is not None:
            axis.axvline(crossover, color="#2a9d8f", linestyle="--", linewidth=1.1)
            axis.text(
                crossover,
                min(min(cpu_y), min(gpu_y)),
                f"GPU better @ {crossover}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=8,
                color="#2a9d8f",
            )

        axis.legend(loc="best")
        crossover_report.append((family, crossover))

    for idx in range(len(families[:4]), 4):
        axes_flat[idx].axis("off")

    fig.suptitle("CPU vs GPU2 Crossover by Family", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return crossover_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-csv", required=True, help="Path to CPU benchmark CSV")
    parser.add_argument("--gpu-csv", required=True, help="Path to GPU benchmark CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    cpu_rows = read_rows(Path(args.cpu_csv))
    gpu_rows = read_rows(Path(args.gpu_csv))

    cpu_map = build_size_time_map(cpu_rows)
    gpu_map = build_size_time_map(gpu_rows)

    report = plot_family_curves(cpu_map, gpu_map, Path(args.out))
    print(f"Saved plot: {args.out}")
    for family, crossover in report:
        if crossover is None:
            print(f"{family}: GPU2 does not beat CPU in common measured sizes")
        else:
            print(f"{family}: first measured size where GPU2 beats CPU -> {crossover}")


if __name__ == "__main__":
    main()
