# Edge CPU/GPU/FLOPs Config Latency Results

- PyTorch CPU follows the previous final CPU protocol: crop 192, 32 threads, warmup 10, samples 100, serial run.
- PyTorch GPU follows the previous final GPU protocol: crop 256, cuda:0 RTX 4090, cuDNN benchmark on, CUDA sync on, warmup 100, samples 400.
- ONNX/TensorRT follows the previous export protocol: crop 256, warmup 100, samples 400, ONNX Runtime CPU/CUDA and TensorRT FP32/FP16.
- Values are mean ± std in ms.

## edge_cpu

- Config: `/home/intern/proxylessnas/search/edge_cpu.config`
- ONNX model: `/home/intern/proxylessnas/search/edge_latency/edge_cpu/onnx_trt_latency/model_fixed_b1.onnx`

| Backend | Latency ms | Samples | Crop | Device | JSON |
| --- | ---: | ---: | ---: | --- | --- |
| `pytorch_cpu` | 101.447 ± 16.965 | 100 | 192 | `cpu` | `/home/intern/proxylessnas/search/edge_latency/edge_cpu/pytorch_cpu_threads32_crop192_100samples_warm10.json` |
| `pytorch_gpu` | 7.265 ± 0.358 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_cpu/pytorch_gpu_cuda0_crop256_400samples_sync_cudnnbench_warm100.json` |
| `onnx_cpu` | 90.687 ± 26.565 | 400 | 256 | `cpu` | `/home/intern/proxylessnas/search/edge_latency/edge_cpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `onnx_cuda` | 2.266 ± 0.060 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_cpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `tensorrt_fp32` | 1.072 ± 0.054 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_cpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `tensorrt_fp16` | 0.788 ± 0.064 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_cpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |

## edge_gpu

- Config: `/home/intern/proxylessnas/search/edge_gpu.config`
- ONNX model: `/home/intern/proxylessnas/search/edge_latency/edge_gpu/onnx_trt_latency/model_fixed_b1.onnx`

| Backend | Latency ms | Samples | Crop | Device | JSON |
| --- | ---: | ---: | ---: | --- | --- |
| `pytorch_cpu` | 174.273 ± 23.101 | 100 | 192 | `cpu` | `/home/intern/proxylessnas/search/edge_latency/edge_gpu/pytorch_cpu_threads32_crop192_100samples_warm10.json` |
| `pytorch_gpu` | 6.676 ± 0.290 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_gpu/pytorch_gpu_cuda0_crop256_400samples_sync_cudnnbench_warm100.json` |
| `onnx_cpu` | 150.088 ± 43.964 | 400 | 256 | `cpu` | `/home/intern/proxylessnas/search/edge_latency/edge_gpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `onnx_cuda` | 2.476 ± 0.290 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_gpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `tensorrt_fp32` | 1.528 ± 0.019 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_gpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `tensorrt_fp16` | 1.069 ± 0.015 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/edge_gpu/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |

## flpos

- Config: `/home/intern/proxylessnas/search/logs/flpos.config`
- ONNX model: `/home/intern/proxylessnas/search/edge_latency/flpos/onnx_trt_latency/model_fixed_b1.onnx`

| Backend | Latency ms | Samples | Crop | Device | JSON |
| --- | ---: | ---: | ---: | --- | --- |
| `pytorch_cpu` | 210.607 ± 7.278 | 100 | 192 | `cpu` | `/home/intern/proxylessnas/search/edge_latency/flpos/pytorch_cpu_threads32_crop192_100samples_warm10.json` |
| `pytorch_gpu` | 7.033 ± 0.132 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/flpos/pytorch_gpu_cuda0_crop256_400samples_sync_cudnnbench_warm100.json` |
| `onnx_cpu` | 149.414 ± 34.161 | 400 | 256 | `cpu` | `/home/intern/proxylessnas/search/edge_latency/flpos/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `onnx_cuda` | 4.498 ± 0.018 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/flpos/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `tensorrt_fp32` | 1.885 ± 0.068 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/flpos/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |
| `tensorrt_fp16` | 1.044 ± 0.020 | 400 | 256 | `cuda:0` | `/home/intern/proxylessnas/search/edge_latency/flpos/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json` |

## Compact Table

| Config | PyTorch CPU | PyTorch GPU | ONNX CPU | ONNX CUDA | TRT FP32 | TRT FP16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `edge_cpu` | 101.447 ± 16.965 | 7.265 ± 0.358 | 90.687 ± 26.565 | 2.266 ± 0.060 | 1.072 ± 0.054 | 0.788 ± 0.064 |
| `edge_gpu` | 174.273 ± 23.101 | 6.676 ± 0.290 | 150.088 ± 43.964 | 2.476 ± 0.290 | 1.528 ± 0.019 | 1.069 ± 0.015 |
| `flpos` | 210.607 ± 7.278 | 7.033 ± 0.132 | 149.414 ± 34.161 | 4.498 ± 0.018 | 1.885 ± 0.068 | 1.044 ± 0.020 |

## Reproduce

```bash
PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
/home/intern/anaconda3/envs/new_env/bin/python search/measure_gpu_latency_from_net_config.py <config> \
  --device cuda:0 --crop-size 256 --samples 400 --warmup 100 --sync-cuda --cudnn-benchmark --output <json>

OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
/home/intern/anaconda3/envs/new_env/bin/python search/measure_cpu_time_from_net_config.py <config> \
  --samples 100 --warmup 10 --cpu-threads 32 --crop-size 192 --output <json>

PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
/home/intern/anaconda3/envs/new_env/bin/python search/measure_onnx_trt_latency_from_net_config.py <config> \
  --device cuda:0 --crop-size 256 --samples 400 --warmup 100 --output-dir <out_dir> --trt-cache-dir <out_dir>/trt_cache
```

- CSV summary: `/home/intern/proxylessnas/search/edge_latency/edge_latency_pytorch_onnx_tensorrt_summary.csv`

## ONNX CUDA IO-Binding/CUDA-Graph Check

The default `onnx_cuda` rows use ordinary `session.run` with CPU NumPy input/output. The following rows use ONNX Runtime CUDA IO binding plus CUDA Graph, keeping tensors on GPU and reducing ORT launch overhead. These are diagnostic/optimized ONNX CUDA numbers; TensorRT remains the preferred deployment backend.

| Config | ONNX CUDA default | ONNX CUDA IO-binding + CUDA Graph | JSON |
| --- | ---: | ---: | --- |
| `edge_cpu` | 2.266 ± 0.060 | 1.536 ± 0.029 | `/home/intern/proxylessnas/search/edge_latency/edge_cpu/onnx_trt_latency/onnx_cuda_iobinding_cudagraph_crop256_400samples.json` |
| `edge_gpu` | 2.476 ± 0.290 | 1.854 ± 0.024 | `/home/intern/proxylessnas/search/edge_latency/edge_gpu/onnx_trt_latency/onnx_cuda_iobinding_cudagraph_crop256_400samples.json` |
| `flpos` | 4.498 ± 0.018 | 3.798 ± 0.104 | `/home/intern/proxylessnas/search/edge_latency/flpos/onnx_trt_latency/onnx_cuda_iobinding_cudagraph_crop256_400samples.json` |
