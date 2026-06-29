# GPU latency rerun on cuda:1, warmup=0, anchor=5.474 ms

Generated: 2026-05-22T15:22:13

## Protocol

- Device: `cuda:1` (`NVIDIA GeForce RTX 4090`)
- Input crop: `256`
- Samples: `400`
- Warmup: `0`
- CUDA synchronize timing: `True`
- `torch.backends.cudnn.benchmark`: `True`
- Stable rule: raw sample `<= 20.0 ms`; cold-start outliers are excluded only for stable summary, while all 400 samples are preserved in the JSON/CSV files.
- Anchor: previous CPU-target architecture stable latency is mapped to `5.474 ms`.
- Anchor measured stable mean on cuda:1: `7.675128 ms`.
- Scale: `0.713212817115`.

## Results

| Architecture | Stable mapped latency | All-400 mapped latency | Raw stable latency | Outlier indices |
|---|---:|---:|---:|---|
| Previous CPU-target anchor | 5.474000 ± 0.351319 ms | 6.552204 ± 21.539957 ms | 7.675128 ± 0.492586 ms | [0] |
| Current CPU candidate | 6.151726 ± 0.397026 ms | 7.245780 ± 21.857306 ms | 8.625372 ± 0.556673 ms | [0] |

## Output files

- Previous mapped JSON: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/previous_cpu_target.gpu_latency_cuda1_crop256_400samples_warm0_anchor5p474_rerun_seconds.json`
- Current mapped JSON: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/current_cpu_candidate.gpu_latency_cuda1_crop256_400samples_warm0_anchor5p474_rerun_seconds.json`
- Summary CSV: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/gpu_latency_cuda1_warm0_anchor5p474_rerun_summary.csv`
- All 400 samples CSV: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/gpu_latency_cuda1_warm0_anchor5p474_rerun_all_400_samples.csv`
- Raw previous rerun JSON: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/previous_cpu_target.gpu_latency_cuda1_crop256_400samples_sync_cudnnbench_warm0_rerun.json`
- Raw current rerun JSON: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/current_cpu_candidate.gpu_latency_cuda1_crop256_400samples_sync_cudnnbench_warm0_rerun.json`

## Reproduce

Run the latency script on an idle GPU, then map with the same anchor rule:

```bash
CUDA_VISIBLE_DEVICES=1 /home/intern/anaconda3/envs/new_env/bin/python /home/intern/proxylessnas/search/measure_gpu_latency_from_net_config.py \
  --net-config /home/intern/proxylessnas/search/cpu_net.config \
  --device cuda:0 --crop-size 256 --samples 400 --warmup 0 --sync-cuda --cudnn-benchmark \
  --out /home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/previous_cpu_target.gpu_latency_cuda1_crop256_400samples_sync_cudnnbench_warm0_rerun.json

CUDA_VISIBLE_DEVICES=1 /home/intern/anaconda3/envs/new_env/bin/python /home/intern/proxylessnas/search/measure_gpu_latency_from_net_config.py \
  --net-config /home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8/learned_net/net.config \
  --device cuda:0 --crop-size 256 --samples 400 --warmup 0 --sync-cuda --cudnn-benchmark \
  --out /home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/current_cpu_candidate.gpu_latency_cuda1_crop256_400samples_sync_cudnnbench_warm0_rerun.json
```

Mapping formula:

```text
scale = 5.474 / mean(previous raw stable samples in ms)
mapped_sample_seconds = raw_sample_ms * scale / 1000
```
