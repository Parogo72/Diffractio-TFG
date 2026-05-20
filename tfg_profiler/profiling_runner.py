from tfg_profiler.config import BenchmarkConfig
from tfg_profiler.outputs import build_output_layout
from tfg_profiler.profiling import profile_suite
from tfg_profiler.runner_cases import build_profile_sets


def run_profiling(config=None, outputs=None, package_name="diffractio"):
    """Run only profiling and write timestamped profiling reports."""
    config = config or BenchmarkConfig().with_defaults()
    outputs = outputs or build_output_layout(config.output_base_dir)
    profiles, _, _ = build_profile_sets(package_name=package_name)

    if config.enable_profiling:
        profile_suite(profiles, output_dir=outputs.profiling_dir, run_id=outputs.run_id)


def main(config=None, package_name="diffractio"):
    """Entry point for profiling stage."""
    run_profiling(config=config, package_name=package_name)


if __name__ == "__main__":
    main()
