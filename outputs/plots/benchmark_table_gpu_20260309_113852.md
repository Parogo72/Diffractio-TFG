# Benchmark Summary (GPU)

## XYZ Fixed Profile

| Metodo | Mediana (s) | Speedup vs RS |
|---|---:|---:|
| RS XYZ | 0.401797 | 1.00x |
| RS_batched XYZ (batch_size=8) | 0.258681 | 1.55x |

## XYZ Sweep

| Size | RS mediana (s) | RS_batched mediana (s) | Speedup (RS/RS_batched) |
|---:|---:|---:|---:|
| 128 | 0.098766 | 0.028183 | 3.50x |
| 256 | 0.399556 | 0.245297 | 1.63x |
| 512 | 2.927140 | 5.250649 | 0.56x |