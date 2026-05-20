#!/usr/bin/env python3
"""
Generate CPU vs GPU comparison plots from two benchmark CSV files.

Usage:
    python scripts/plotting/plot_speedup_by_family.py \
    --cpu-csv outputs/benchmarking/benchmark_all_20260223_175043.csv \
    --gpu-csv outputs/benchmarking/benchmark_all_gpu_20260309_113027.csv \
    --out outputs/plots/cpu_vs_gpu_speedup_by_family_custom.png
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def family_from_label(label: str) -> str:
    m = re.match(r"^RS(?:_[A-Za-z0-9_]+)?\s+([A-Za-z0-9_]+)", label)
    return m.group(1) if m else label


def suite_rows(rows):
    return [r for r in rows if r.get("size") in ("", None)]


def size_rows(rows):
    return [r for r in rows if r.get("size") not in ("", None)]


def build_family_map(rows):
    fmap = {}
    for r in rows:
        fam = family_from_label(r["label"])
        fmap[fam] = float(r["elapsed_median_s"])
    return fmap


def build_case_map(rows):
    cmap = {}
    for r in rows:
        case = family_from_label(r["label"])
        cmap[case] = {
            "label": r["label"],
            "elapsed_median_s": float(r["elapsed_median_s"]),
        }
    return cmap


def build_case_size_map(rows):
    csmap = {}
    for r in rows:
        fam = family_from_label(r["label"])
        size = str(r["size"]).strip()
        key = (fam, size)
        csmap[key] = float(r["elapsed_median_s"])
    return csmap


def _derived_output(out_png: Path, suffix: str) -> Path:
    return out_png.with_name(f"{out_png.stem}{suffix}{out_png.suffix}")


def plot_speedup_by_family(families, speedups, out_png: Path):
    plt.figure(figsize=(10, 5.5))
    colors = ["#2a9d8f" if s >= 1.0 else "#e76f51" for s in speedups]
    bars = plt.bar(families, speedups, color=colors, edgecolor="black", linewidth=0.6)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Speedup (CPU/GPU)")
    plt.xlabel("Family")
    plt.title("CPU vs GPU Speedup by Family")
    plt.grid(axis="y", alpha=0.25)

    for b, s in zip(bars, speedups):
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{s:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_speedup_by_case(case_labels, case_speedups, out_png: Path):
    order = sorted(range(len(case_speedups)), key=lambda i: case_speedups[i], reverse=True)
    labels_sorted = [case_labels[i] for i in order]
    speedups_sorted = [case_speedups[i] for i in order]

    height = max(6.0, 0.36 * len(labels_sorted))
    plt.figure(figsize=(10.5, height))
    colors = ["#2a9d8f" if s >= 1.0 else "#e76f51" for s in speedups_sorted]
    bars = plt.barh(labels_sorted, speedups_sorted, color=colors, edgecolor="black", linewidth=0.5)
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Speedup (CPU/GPU)")
    plt.ylabel("Case (family, size)")
    plt.title("CPU vs GPU Speedup by Case")
    plt.grid(axis="x", alpha=0.25)

    xmax = max(speedups_sorted) if speedups_sorted else 1.0
    pad = 0.01 * xmax
    for b, s in zip(bars, speedups_sorted):
        plt.text(
            b.get_width() + pad,
            b.get_y() + b.get_height() / 2.0,
            f"{s:.2f}x",
            ha="left",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_scatter_loglog(cpu_times, gpu_times, labels, out_png: Path):
    plt.figure(figsize=(7.5, 6))
    plt.scatter(cpu_times, gpu_times, s=64, color="#457b9d", edgecolors="black", linewidths=0.5)

    lo = min(min(cpu_times), min(gpu_times))
    hi = max(max(cpu_times), max(gpu_times))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray", label="y=x")

    for x, y, label in zip(cpu_times, gpu_times, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("CPU elapsed_median_s")
    plt.ylabel("GPU elapsed_median_s")
    plt.title("CPU vs GPU Scatter (Log-Log)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-csv", required=True, help="Path to CPU benchmark CSV")
    parser.add_argument("--gpu-csv", required=True, help="Path to GPU benchmark CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    cpu_csv = Path(args.cpu_csv)
    gpu_csv = Path(args.gpu_csv)
    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cpu_rows_all = read_rows(cpu_csv)
    gpu_rows_all = read_rows(gpu_csv)

    cpu_rows = suite_rows(cpu_rows_all)
    gpu_rows = suite_rows(gpu_rows_all)

    cpu_map = build_family_map(cpu_rows)
    gpu_map = build_family_map(gpu_rows)

    families = sorted([f for f in cpu_map if f in gpu_map])
    if not families:
        raise RuntimeError("No common families found between CPU and GPU CSVs.")

    speedups = [cpu_map[f] / gpu_map[f] for f in families]

    plot_speedup_by_family(families, speedups, out_png)

    cpu_case_size_map = build_case_size_map(size_rows(cpu_rows_all))
    gpu_case_size_map = build_case_size_map(size_rows(gpu_rows_all))
    case_keys = sorted([key for key in cpu_case_size_map if key in gpu_case_size_map], key=lambda k: (k[0], int(k[1])))
    case_labels = [f"{fam} (size={size})" for fam, size in case_keys]
    case_cpu_times = [cpu_case_size_map[key] for key in case_keys]
    case_gpu_times = [gpu_case_size_map[key] for key in case_keys]
    case_speedups = [cpu / gpu for cpu, gpu in zip(case_cpu_times, case_gpu_times)]

    case_out = _derived_output(out_png, "_speedup_by_case")
    scatter_out = _derived_output(out_png, "_scatter_loglog")

    plot_speedup_by_case(case_labels, case_speedups, case_out)
    plot_scatter_loglog(case_cpu_times, case_gpu_times, case_labels, scatter_out)

    print(f"Saved plot: {out_png}")
    print(f"Saved plot: {case_out}")
    print(f"Saved plot: {scatter_out}")
    for fam in families:
        print(
            f"{fam}: CPU={cpu_map[fam]:.6f}s "
            f"GPU={gpu_map[fam]:.6f}s "
            f"speedup={cpu_map[fam] / gpu_map[fam]:.4f}x"
        )


if __name__ == "__main__":
    main()
