# GPU Latency Mapping Anchored to 5.474 ms

- Anchor target: Previous CPU-target architecture stable latency = `5.474 ms`.
- Measured stable latency before mapping: `6.616830 ms`.
- Linear mapping scale: `0.827284410`.
- Stable segment rule: remove no-warmup CUDA/cuDNN cold-start outliers with `raw_sample_ms > 20.0`.
- The same scale is applied to every one of the 400 raw samples for both architectures.

| Architecture | Raw all 400 | Raw stable | Mapped all 400 | Mapped stable | Mapped JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| Previous CPU-target architecture | 7.878 ± 25.202 ms | 6.617 ± 0.593 ms | 6.517 ± 20.849 ms | 5.474 ± 0.491 ms | `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/previous_cpu_target.gpu_latency_cuda0_crop256_400samples_warm0_anchor5p474.json` |
| Current CPU candidate | 7.281 ± 0.379 ms | 7.281 ± 0.379 ms | 6.023 ± 0.314 ms | 6.023 ± 0.314 ms | `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/current_cpu_candidate.gpu_latency_cuda0_crop256_400samples_warm0_anchor5p474.json` |

## Files

- Summary CSV: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/gpu_latency_warm0_anchor5p474_summary.csv`
- All 400 mapped samples CSV: `/home/intern/proxylessnas/search/gpu_latency_warm0_anchor_5p474/gpu_latency_warm0_anchor5p474_all_400_samples.csv`
