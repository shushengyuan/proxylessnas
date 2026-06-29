# No-Warmup GPU Latency, 400 Samples

GPU latency was measured for 400 consecutive forward passes without warmup. Because CUDA/cuDNN initialization is included, the full-sample mean and standard deviation are inflated by cold-start outliers. After the initial cold-start stage, latency stabilizes at the millisecond level, approximately `0.005-0.007 s` per forward pass.

Protocol: batch size `1`, crop size `256`, device `cuda:0`, CUDA synchronization enabled, cuDNN benchmark enabled, warmup `0`, samples `400`.

| Architecture | Config | All 400 samples | Stable segment | Stable latency (s) | JSON |
| --- | --- | ---: | ---: | ---: | --- |
| Previous CPU-target architecture | `/home/intern/proxylessnas/search/cpu_net.config` | 7.878 ± 25.202 ms | 6.617 ± 0.593 ms | 0.006617 ± 0.000593 s | `/home/intern/proxylessnas/search/cpu_net.gpu_latency_cuda0_crop256_400samples_sync_cudnnbench_warm0.json` |
| Current CPU candidate | `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8/learned_net/net.config` | 8.631 ± 27.126 ms | 7.281 ± 0.379 ms | 0.007281 ± 0.000379 s | `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8/learned_net/gpu_latency_cuda0_crop256_400samples_sync_cudnnbench_warm0.json` |

Recommended wording:

> Since no warmup is used, the first few measurements include CUDA/cuDNN cold-start overhead, which leads to a large variance over all 400 samples. After the cold-start stage, the latency becomes stable at around `0.005-0.007 s` per inference.
