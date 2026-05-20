import cProfile
import io
import pstats
import time
import tracemalloc
from pathlib import Path

from tfg_profiler.outputs import sanitize_name

from .telemetry import get_rss_bytes


def profile_callable(label, func, *args, report_path=None, **kwargs):
    """Profile a callable and print time, memory, and cProfile statistics."""
    profiler = cProfile.Profile()
    tracemalloc.start()
    rss_before = get_rss_bytes()

    start = time.perf_counter()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    elapsed = time.perf_counter() - start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = get_rss_bytes()

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats("cumulative").print_stats(30)
    stats_text = stats_stream.getvalue()

    print(f"\n=== Profiling: {label} ===")
    print(f"Elapsed: {elapsed:.6f} s")
    print(f"Python alloc current: {current / (1024 ** 2):.2f} MiB")
    print(f"Python alloc peak: {peak / (1024 ** 2):.2f} MiB")
    if rss_before is not None and rss_after is not None:
        delta_rss = (rss_after - rss_before) / (1024 ** 2)
        print(f"Process RSS delta: {delta_rss:.2f} MiB")
    print(stats_text)

    if report_path:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8") as handle:
            handle.write(f"label: {label}\n")
            handle.write(f"elapsed_s: {elapsed:.6f}\n")
            handle.write(f"python_current_mib: {current / (1024 ** 2):.4f}\n")
            handle.write(f"python_peak_mib: {peak / (1024 ** 2):.4f}\n")
            if rss_before is not None and rss_after is not None:
                delta_rss = (rss_after - rss_before) / (1024 ** 2)
                handle.write(f"rss_delta_mib: {delta_rss:.4f}\n")
            handle.write("\n[cprofile]\n")
            handle.write(stats_text)
        print(f"Profile report written to: {report}")

    return result


def profile_suite(profiles, output_dir=None, run_id=None):
    """Run profile_callable for multiple profile items."""
    results = {}
    for item in profiles:
        if isinstance(item, dict):
            label = item["label"]
            func = item["func"]
            args = item.get("args", ())
            kwargs = item.get("kwargs", {})
        else:
            label, func, args, kwargs = item

        report_path = None
        if output_dir and run_id:
            report_path = Path(output_dir) / f"profile_{sanitize_name(label)}_{run_id}.txt"

        results[label] = profile_callable(label, func, *args, report_path=report_path, **kwargs)
    return results
