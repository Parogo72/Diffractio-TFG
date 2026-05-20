## 1. Profiling general de RS

Version CPU:

```bash
python -m tfg_profiler profiling diffractio
```

```bash
python -m tfg_profiler profiling <package>
```

## 2. Profiling interno de RS

Profiling interno programatico:

```bash
python scripts/profiling/line_profile_internal.py
```

Profiling con `line_profiler` sobre RS:

```bash
python -m line_profiler -v scripts/profiling/line_profile_runner.py
```

## 3. Comparacion de resultados entre versiones

```bash
python -m tfg_profiler compare diffractio1 diffractio2
```

## 4. Benchmark

Benchmark version CPU:

```bash
python -m tfg_profiler benchmarking diffractio
```

```bash
python -m tfg_profiler benchmarking <package>
```

## 5. Plotting de resultados

Usa los CSV generados en `outputs/benchmarking/`:

```bash
python scripts/plotting/plot_speedup_by_family.py --cpu-csv outputs/benchmarking/benchmark_all_YYYYMMDD_HHMMSS.csv --gpu-csv outputs/benchmarking/benchmark_all_gpu_YYYYMMDD_HHMMSS.csv --out outputs/plots/cpu_vs_gpu_speedup_by_family_custom.png
```

Grafica de cruce por tamano (curvas CPU y GPU2 en la misma figura):

```bash
python scripts/plotting/plot_crossover_by_size.py --cpu-csv outputs/benchmarking/benchmark_all_YYYYMMDD_HHMMSS.csv --gpu-csv outputs/benchmarking/benchmark_all_diffractio_gpu2_YYYYMMDD_HHMMSS.csv --out outputs/plots/cpu_gpu_crossover_by_size.png
```