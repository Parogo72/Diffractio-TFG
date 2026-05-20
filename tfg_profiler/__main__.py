import argparse

from tfg_profiler.benchmarking_runner import run_benchmarking
from tfg_profiler.profile_output_checker import run_profile_output_check
from tfg_profiler.profiling_runner import run_profiling

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m tfg_profiler",
        description="Run profiling, benchmarking, or compare outputs between two backends.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="profiling",
        help="profiling | benchmarking | compare.",
    )
    parser.add_argument(
        "backend",
        nargs="?",
        default="diffractio",
        help="Backend package.",
    )
    parser.add_argument(
        "other_backend",
        nargs="?",
        default=None,
        help="Package B for compare mode.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    target = args.target.lower()

    if target == "profiling":
        run_profiling(package_name=args.backend)
        return
    if target == "benchmarking":
        run_benchmarking(package_name=args.backend)
        return
    if target == "compare":
        if not args.other_backend:
            raise SystemExit("Compare mode requires two backends: python -m tfg_profiler compare <pkg_a> <pkg_b>")
        run_profile_output_check(package_a=args.backend, package_b=args.other_backend)
        return

    raise SystemExit(
        "Invalid arguments. Use profiling, benchmarking, or compare <pkg_a> <pkg_b>."
    )


if __name__ == "__main__":
    main()