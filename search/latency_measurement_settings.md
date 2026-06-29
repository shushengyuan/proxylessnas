# Latency Measurement Settings

Date: 2026-05-19

This document records the current CPU/GPU latency measurement settings used in this repo and the measured values that should be referenced later. The goal is to keep all architecture comparisons under the same measurement protocol.

## Final CPU Protocol

Use this CPU setting when reporting CPU latency for the current comparison.

- Environment: `conda activate new_env`
- Script: `search/measure_cpu_time_from_net_config.py`
- Input: batch size `1`
- Crop size: `192`
- Warmup: `10`
- Samples: `100`
- CPU threads: `32`
- Execution: serial, do not run other CPU latency jobs in parallel
- Output: JSON file with all per-sample latency values

This protocol was chosen because it reproduces the older CPU target latency scale more closely. In particular, the previous CPU target architecture was reported around `63.91 ms`; under this setting it measures `62.49 ms`.

Command template:

```bash
source /home/intern/anaconda3/etc/profile.d/conda.sh
conda activate new_env
PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
python search/measure_cpu_time_from_net_config.py <net.config> \
  --samples 100 \
  --warmup 10 \
  --cpu-threads 32 \
  --crop-size 192 \
  --output <output.json>
```

CPU results:

| Architecture | net.config | Mean ms | P50 ms | P90 ms | JSON |
|---|---:|---:|---:|---:|---|
| Previous CPU target | `/home/intern/proxylessnas/search/cpu_net.config` | `62.489` | `61.362` | `67.494` | `/home/intern/proxylessnas/search/cpu_net.cpu_latency_threads32_crop192_100samples_serial.json` |
| Current CPU candidate | `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8/learned_net/net.config` | `63.144` | `62.518` | `66.001` | `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8/learned_net/cpu_latency_threads32_crop192_100samples_serial.json` |
| GPU-direction candidate, CPU measured | `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2/learned_net/net.config` | `87.943` | `86.769` | `94.291` | `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2/learned_net/cpu_latency_threads32_crop192_100samples_serial.json` |
| 2023 MBConv reference, CPU measured | `/home/intern/proxylessnas/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/learned_net/net.config` | `90.177` | `89.684` | `94.534` | `/home/intern/proxylessnas/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/learned_net/cpu_latency_threads32_crop192_100samples_serial.json` |

Notes:

- Do not use the strict single-thread CPU measurements for the final table unless explicitly reporting an ablation of measurement settings.
- The single-thread exploratory protocol was `crop_size=256`, `cpu_threads=1`, `warmup=10`, `samples=400`; it is more controlled but does not match the older CPU latency scale.

## Final GPU Protocol

Use this GPU setting when reporting GPU latency for the current comparison.

- Environment: `conda activate new_env`
- Script: `search/measure_gpu_latency_from_net_config.py`
- Device: `cuda:0`
- GPU: `NVIDIA GeForce RTX 4090`
- Input: batch size `1`
- Crop size: `256`
- Warmup: `100`
- Samples: `400`
- CUDA synchronization: enabled, `--sync-cuda`
- cuDNN benchmark: enabled, `--cudnn-benchmark`
- Execution: serial, do not run parallel GPU benchmarks
- Output: JSON file with all per-sample latency values

This protocol is preferred for final reporting because synchronized CUDA timing is more defensible than the old search-log timing style. The old search-log style did not call `torch.cuda.synchronize()` and can be strongly affected by long-tail samples.

Command template:

```bash
source /home/intern/anaconda3/etc/profile.d/conda.sh
conda activate new_env
PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
python search/measure_gpu_latency_from_net_config.py <net.config> \
  --device cuda:0 \
  --crop-size 256 \
  --samples 400 \
  --warmup 100 \
  --sync-cuda \
  --cudnn-benchmark \
  --output <output.json>
```

Raw GPU results under the final protocol on this RTX 4090:

| Architecture | net.config | Mean ± Std ms | P50 ms | P90 ms | JSON |
|---|---:|---:|---:|---:|---|
| 2023 MBConv reference | `/home/intern/proxylessnas/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/learned_net/net.config` | `8.878 ± 3.716` | `7.882` | `13.691` | `/home/intern/proxylessnas/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/learned_net/gpu_latency_cuda0_crop256_400samples_sync_cudnnbench_warm100.json` |
| Current GPU candidate | `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2/learned_net/net.config` | `8.539 ± 1.293` | `8.244` | `9.019` | `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2/learned_net/gpu_latency_cuda0_crop256_400samples_sync_cudnnbench_warm100.json` |

Recommended value to fill for the current GPU candidate:

- If the table uses the old 2023 anchor where the 2023 MBConv reference is `4.954 ms`, use the anchor-normalized value: `4.765 ± 0.722 ms`.
- Current-machine raw RTX 4090 measurement is only a reference: `8.539 ± 1.293 ms`.
- The 2023 anchor row should remain `4.954 ms` in that table, not the raw RTX 4090 value `8.878 ± 3.716 ms`.

Normalization formula:

```text
scale              = old_2023_anchor / raw_2023_mean
                   = 4.954 / 8.878405094146729
                   = 0.5579831002829581

normalized_current_mean = raw_current_mean * scale
                        = 8.539266586303711 * 0.5579831002829581
                        = 4.7647664439684165 ms

normalized_current_std  = raw_current_std * scale
                        = 1.2931876586977296 * 0.5579831002829581
                        = 0.721576859047819 ms
```

## Legacy GPU Protocol For Reference

The original `RunManager.net_latency(l_type='gpu')` style used during search is:

- Input: batch size `1`
- Crop size: usually `256`
- Fast mode: `warmup=5`, `samples=10`
- Slow mode: `warmup=50`, `samples=100`
- CUDA synchronization: disabled
- Timing source: Python `time.time()`

This protocol is useful only when trying to match old search logs such as `Latency-gpu: 4.954ms`. It should not be used as the primary final reporting protocol unless the paper/table explicitly follows the old search-log latency definition.

Example legacy command:

```bash
source /home/intern/anaconda3/etc/profile.d/conda.sh
conda activate new_env
PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
python search/measure_gpu_latency_from_net_config.py <net.config> \
  --device cuda:0 \
  --crop-size 256 \
  --samples 10 \
  --warmup 5 \
  --cudnn-benchmark \
  --output <output.json>
```

Measured legacy-style result for the current GPU candidate:

| Architecture | Setting | Mean ms | P50 ms | P90 ms | JSON |
|---|---:|---:|---:|---:|---|
| Current GPU candidate | `warmup=50`, `samples=400`, no CUDA sync, `cudnn.benchmark=True` | `9.307` | `8.239` | `14.263` | `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2/learned_net/gpu_latency_cuda0_crop256_400samples_legacy_cudnnbench.json` |

## Reporting Recommendation

For the current tables:

- CPU latency: report the `mean_ms` from the final CPU protocol.
- GPU latency in a table anchored to the old 2023 reference: keep the 2023 architecture as `4.954 ms`, and report the current GPU candidate as `4.765 ± 0.722 ms`.
- GPU latency on this RTX 4090 without anchoring: report the raw measured value `8.539 ± 1.293 ms` for `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2/learned_net/net.config`.
- Keep the raw JSON output files with per-sample latency values for auditability.
