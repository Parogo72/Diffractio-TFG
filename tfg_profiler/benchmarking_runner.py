from tfg_profiler.benchmarking import benchmark_suite, benchmark_sweep, plot_scaling, write_combined_csv
from tfg_profiler.config import BenchmarkConfig
from tfg_profiler.outputs import build_output_layout, sanitize_name
from tfg_profiler.runner_cases import build_profile_sets


def _backend_file_tag(package_name):
    if package_name == "diffractio":
        return ""
    return f"_{sanitize_name(package_name)}"


def run_benchmarking(config=None, outputs=None, package_name="diffractio"):
    """Run benchmark/sweep/plot stages for a selected diffractio-compatible backend."""
    config = config or BenchmarkConfig().with_defaults()
    outputs = outputs or build_output_layout(config.output_base_dir)
    _, benchmark_profiles, sweep_profiles = build_profile_sets(package_name=package_name)
    file_tag = _backend_file_tag(package_name)
    plot_suffix = outputs.run_id if not file_tag else f"{file_tag[1:]}_{outputs.run_id}"

    bench_rows = []
    if config.enable_benchmark:
        bench_rows = benchmark_suite(
            benchmark_profiles,
            repeats=config.benchmark_repeats,
            warmup=config.benchmark_warmup,
            csv_path=outputs.benchmarking_dir / f"benchmark_suite{file_tag}_{outputs.run_id}.csv",
        )

    sweep_rows_all = []
    if config.enable_size_sweep:
        sweep_specs = [
            (0, "x", config.size_sweep_x),
            (1, "xz", config.size_sweep_xz),
            (2, "xy", config.size_sweep_xy),
            (3, "xyz", config.size_sweep_xyz),
        ]
        for profile_idx, suffix, sizes in sweep_specs:
            sweep_rows_all.extend(
                benchmark_sweep(
                    [sweep_profiles[profile_idx]],
                    sizes,
                    repeats=config.benchmark_repeats,
                    warmup=config.benchmark_warmup,
                    csv_path=outputs.sweeps_dir / f"benchmark_sweep_{suffix}{file_tag}_{outputs.run_id}.csv",
                )
            )

    combined_rows = bench_rows + sweep_rows_all
    if combined_rows:
        write_combined_csv(combined_rows, csv_path=outputs.benchmarking_dir / f"benchmark_all{file_tag}_{outputs.run_id}.csv")

    if config.enable_plots and sweep_rows_all:
        plot_scaling(
            sweep_rows_all,
            output_dir=outputs.plots_dir,
            file_suffix=plot_suffix,
        )

def main(config=None, package_name="diffractio"):
    """Entry point for benchmark stage."""
    run_benchmarking(config=config, package_name=package_name)


if __name__ == "__main__":
    main()
