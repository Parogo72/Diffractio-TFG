import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class OutputLayout:
    """Filesystem layout for profiler outputs in a single run."""

    base_dir: Path
    run_id: str
    profiling_dir: Path
    benchmarking_dir: Path
    sweeps_dir: Path
    plots_dir: Path


def sanitize_name(value: str) -> str:
    """Convert a label into a filesystem-friendly token."""
    normalized = value.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "item"


def build_output_layout(base_dir: str = "outputs", run_id: str = None) -> OutputLayout:
    """Create output directories and return a structured layout."""
    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(base_dir)
    profiling_dir = base_path / "profiling"
    benchmarking_dir = base_path / "benchmarking"
    sweeps_dir = benchmarking_dir / "sweeps"
    plots_dir = base_path / "plots"

    profiling_dir.mkdir(parents=True, exist_ok=True)
    benchmarking_dir.mkdir(parents=True, exist_ok=True)
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return OutputLayout(
        base_dir=base_path,
        run_id=resolved_run_id,
        profiling_dir=profiling_dir,
        benchmarking_dir=benchmarking_dir,
        sweeps_dir=sweeps_dir,
        plots_dir=plots_dir,
    )
