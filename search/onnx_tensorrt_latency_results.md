# ONNX and TensorRT Latency Results

Date: 2026-05-19

This document records the ONNXRuntime and TensorRT latency measurements for two `net.config` architectures.

## Measurement Protocol

- Environment: `conda activate new_env`
- GPU: `NVIDIA GeForce RTX 4090`
- Device: `cuda:0`
- Input: fixed zero tensor, batch size `1`
- Input shape: `1 x 3 x 256 x 256`
- Crop size: `256`
- Warmup: `100`
- Samples: `400`
- ONNX backend: ONNXRuntime
- TensorRT backend: ONNXRuntime `TensorrtExecutionProvider`
- Report format: `mean ± std`, unit `ms`
- Raw samples: saved in JSON files listed below

The `±` value is the standard deviation (`std_ms`), not the variance. Variance is stored separately as `var_ms2` in the JSON files.

## Tested Architectures

| Name | net.config | Checkpoint |
|---|---|---|
| GPU candidate retrain 10 | `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/net.config` | `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/checkpoint/model_best.pth.tar` |
| CPU candidate | `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8/learned_net/net.config` | `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/checkpoint/model_best.pth.tar` |

For the CPU candidate, the retrain directory copy of `net.config` and the requested `learned_net/net.config` are identical.

## Results

Main GPU backend results:

| Architecture | ONNX CUDA | TensorRT FP32 | TensorRT FP16 |
|---|---:|---:|---:|
| `ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10` | `2.831 ± 0.096` | `1.243 ± 0.057` | `1.033 ± 0.960` |
| `ablate_cpu_combo_b2Ghost5x5_b4Shufflee8` | `2.926 ± 0.081` | `1.089 ± 0.044` | `0.742 ± 0.023` |

ONNX CPU results:

| Architecture | ONNX CPU |
|---|---:|
| `ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10` | `418.284 ± 73.367` |
| `ablate_cpu_combo_b2Ghost5x5_b4Shufflee8` | `315.341 ± 57.910` |

Detailed statistics:

| Architecture | Backend | Mean ms | Std ms | Var ms^2 | P50 ms | P90 ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10` | ONNX CPU | `418.284` | `73.367` | `5382.681835` | `407.805` | `521.428` | `240.422` | `674.831` |
| `ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10` | ONNX CUDA | `2.831` | `0.096` | `0.009200` | `2.825` | `2.940` | `2.684` | `3.318` |
| `ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10` | TensorRT FP32 | `1.243` | `0.057` | `0.003238` | `1.232` | `1.275` | `1.188` | `1.841` |
| `ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10` | TensorRT FP16 | `1.033` | `0.960` | `0.921071` | `0.909` | `0.973` | `0.845` | `13.582` |
| `ablate_cpu_combo_b2Ghost5x5_b4Shufflee8` | ONNX CPU | `315.341` | `57.910` | `3353.617291` | `316.224` | `389.632` | `141.563` | `497.213` |
| `ablate_cpu_combo_b2Ghost5x5_b4Shufflee8` | ONNX CUDA | `2.926` | `0.081` | `0.006524` | `2.891` | `3.034` | `2.843` | `3.500` |
| `ablate_cpu_combo_b2Ghost5x5_b4Shufflee8` | TensorRT FP32 rerun | `1.089` | `0.044` | `0.001894` | `1.084` | `1.096` | `1.067` | `1.821` |
| `ablate_cpu_combo_b2Ghost5x5_b4Shufflee8` | TensorRT FP16 rerun | `0.742` | `0.023` | `0.000528` | `0.739` | `0.764` | `0.713` | `0.954` |

Note: TensorRT FP16 for `ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10` has one large max outlier, so its standard deviation is inflated. For `ablate_cpu_combo_b2Ghost5x5_b4Shufflee8`, the first TensorRT FP32 run had two outliers (`16.806 ms` and `25.179 ms`) and reported `1.327 ± 1.437 ms`; a TensorRT-only rerun removed that artifact and is used as the final TensorRT value.

## Output Files

GPU candidate retrain 10:

- JSON: `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/onnx_trt_latency/onnx_trt_latency_crop256_400samples.json`
- ONNX: `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/onnx_trt_latency/model_fixed_b1.onnx`
- TensorRT cache: `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/onnx_trt_latency/trt_cache`

CPU candidate:

- JSON: `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/onnx_trt_latency_from_learned_net/onnx_trt_latency_crop256_400samples.json`
- ONNX: `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/onnx_trt_latency_from_learned_net/model_fixed_b1.onnx`
- TensorRT cache: `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/onnx_trt_latency_from_learned_net/trt_cache`
- TensorRT rerun JSON: `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/onnx_trt_latency_from_learned_net_rerun_trt_only/onnx_trt_latency_crop256_400samples.json`
- TensorRT rerun ONNX: `/home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/onnx_trt_latency_from_learned_net_rerun_trt_only/model_fixed_b1.onnx`

## Reproduction

Script used:

```bash
/home/intern/proxylessnas/search/measure_onnx_trt_latency_from_net_config.py
```

The script exports the model to ONNX and benchmarks:

- `CPUExecutionProvider`
- `CUDAExecutionProvider`
- `TensorrtExecutionProvider` with FP32
- `TensorrtExecutionProvider` with FP16

Dependencies installed in `new_env`:

```bash
python -m pip install onnx onnxruntime-gpu tensorrt-cu12
```

Verify providers:

```bash
source /home/intern/anaconda3/etc/profile.d/conda.sh
conda activate new_env
python - <<'PY'
import onnxruntime as ort
import tensorrt as trt
print(ort.get_available_providers())
print(trt.__version__)
PY
```

Expected providers include:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Reproduce GPU candidate retrain 10:

```bash
source /home/intern/anaconda3/etc/profile.d/conda.sh
conda activate new_env
PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
python search/measure_onnx_trt_latency_from_net_config.py \
  /home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/net.config \
  --checkpoint /home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/checkpoint/model_best.pth.tar \
  --device cuda:0 \
  --crop-size 256 \
  --warmup 100 \
  --samples 400 \
  --output-dir /home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/onnx_trt_latency
```

Reproduce CPU candidate:

```bash
source /home/intern/anaconda3/etc/profile.d/conda.sh
conda activate new_env
PYTHONPATH=/home/intern/proxylessnas:/home/intern/proxylessnas/search \
python search/measure_onnx_trt_latency_from_net_config.py \
  /home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8/learned_net/net.config \
  --checkpoint /home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/checkpoint/model_best.pth.tar \
  --device cuda:0 \
  --crop-size 256 \
  --warmup 100 \
  --samples 400 \
  --output-dir /home/intern/proxylessnas/search/logs/ablate_cpu_combo_b2Ghost5x5_b4Shufflee8_Retrain_2/onnx_trt_latency_from_learned_net
```

## Notes

- The first TensorRT run builds an engine/timing cache, which can take tens of seconds to minutes. This build time is not included in the measured samples.
- ONNXRuntime TensorRT EP may print tactic cache warnings. These warnings appeared during engine build but did not stop execution.
- Reusing an existing TensorRT cache can change startup time, but not the reported per-sample latency measurement after warmup.
- Keep the JSON files if exact per-sample auditability is needed.
