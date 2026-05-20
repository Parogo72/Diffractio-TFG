from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for profiling, benchmarking, and sweep runs."""

    enable_profiling: bool = True
    enable_benchmark: bool = True
    enable_size_sweep: bool = True
    enable_plots: bool = True
    benchmark_repeats: int = 3
    benchmark_warmup: int = 1
    output_base_dir: str = "outputs"
    size_sweep_x: Optional[List[int]] = None
    size_sweep_xz: Optional[List[int]] = None
    size_sweep_xy: Optional[List[int]] = None
    size_sweep_xyz: Optional[List[int]] = None

    def with_defaults(self):
        """Fill missing sweep sizes with defaults and return self."""
        if self.size_sweep_x is None:
            self.size_sweep_x = [
                256,
                384,
                512,
                768,
                1024,
                1536,
                2048,
                3072,
                4096,
                6144,
                8192,
                12288,
                16384,
                24576,
                32768,
                36864,
                40960,
                49152,
                57344,
                65536,
            ]
        if self.size_sweep_xz is None:
            self.size_sweep_xz = [
                128,
                192,
                256,
                384,
                512,
                640,
                768,
                896,
                1024,
                1280,
                1536,
                2048,
                2560,
                3072,
                3584,
                4096,
                4608,
                5120,
                6144,
            ]
        if self.size_sweep_xy is None:
            self.size_sweep_xy = [
                128,
                192,
                256,
                384,
                512,
                640,
                768,
                896,
                1024,
                1280,
                1536,
                2048,
                2560,
                3072,
                3584,
                4096,
                4608,
                5120,
                6144,
            ]
        if self.size_sweep_xyz is None:
            self.size_sweep_xyz = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 832, 896, 960, 1024]
        return self
