  
# HNA-NAS: Hardware-aware Nested Architecture Search for Infrared Small Target Segmentation


## News
Here is the English translation, perfect for your GitHub README's News section:

- [🔥 Latest Release] We propose HNA-NAS (Hardware-aware Nested Architecture Neural Architecture Search)—a hardware-optimized neural architecture search method for infrared small target segmentation!

- [✨ Core Framework] We introduce Meta-NA, a brand new upgraded nested architecture, and design a unique two-stage NAS algorithm to progressively achieve architecture simplification and cell optimization.

- [🚀 Ultra-Low Search Cost] By adopting a binarized parameter update strategy, we significantly reduce memory requirements. The entire architecture search process can be completed using only a single RTX 3090 GPU!

- [⚡ Extreme Lightweight & Efficiency] We incorporate typical computational costs (Params, FLOPs, and Latency) as hard constraints. Compared to existing state-of-the-art (SOTA) methods, HNA-NAS achieves comparable high-accuracy segmentation performance with much fewer parameters and faster inference speeds.

- [💻 Extensive Edge Deployment] Perfectly adapted for edge devices with limited computing capabilities! This project provides detailed deployment experiments and support across various edge computing hardware, including Nvidia-AGX, Xavier, Ascend 310B, and 310P. The source code is fully open-sourced!

## Performance Summary of HNA-NAS

Based on experiments across three benchmark IRST datasets, **HNA-NAS** maintains comparable segmentation accuracy to previous nested architectures while drastically reducing computational overhead and inference time. 

**Key Achievements:**
*   **Highly Lightweight:** Reduces parameters by ~77% and FLOPs by ~90%.
*   **Versatile Acceleration:** Delivers significant speedups across all tested platforms, including standard hardware and resource-constrained edge devices.

### 1. Computational Cost & Inference Latency
| Efficiency Metric | HNA-NAS | Previous Nested Architectures |
| :--- | :--- | :--- |
| **Parameters** | 1.051M | 4.696M |
| **FLOPs** | 1.328G | 14.05G |
| **Latency (GPU)** | 4.954ms | 26.98ms |
| **Latency (CPU)** | 63.91ms | 168.2ms |
| **Latency (Edge GPU)** | 54.47ms | 195.5ms |
| **Latency (Edge CPU)** | 920.7ms | 4061ms |

### 2. Segmentation Performance vs. SOTA Methods
*(Note: **IoU** = Intersection over Union, **Pd** = Probability of Detection, **Fa** = False Alarm rate. CPU/GPU metrics indicate inference latency in milliseconds).*

| Method | IoU | CPU | GPU | Pd | Fa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| UIUNet [32] | 63.37 | 523.5 | 21.65 | 91.07 | 28.46 |
| MSHNet [51] | 67.76 | 64.34 | 9.632 | 94.22 | 10.85 |
| PConv [52] | 66.81 | 69.06 | 10.06 | 92.52 | 11.01 |
| NAS-UNet [20] | 61.63 | 270.5 | 10.76 | 87.41 | 9.413 |
| BiX-NAS [53] | 62.72 | 144.4 | 11.26 | 87.75 | 8.046 |
| ProxylessNAS [19] | 65.73 | 326.7 | 7.571 | 93.19 | 8.046 |
| **HNA-NAS** (*CPU*) | 66.99 | 63.91 | 5.474 | 89.45 | 8.502 |
| **HNA-NAS** (*GPU*) | 68.16 | 91.60 | 4.954 | 91.15 | 8.502 |

## On-Device Model Deployment Demo (Inference)
<table>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/c6abea94-90fd-4b67-ab22-da5355dde30c" width="100%" controls></video><br>
      <b>CPU model on 310b</b>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/39121fa6-f988-4467-97fc-70ad33c6b5d9" width="100%" controls></video><br>
      <b>GPU model on 310b</b>
    </td>
  </tr>
  
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/851f7956-2dd3-4d48-adb9-344dcb3d5844" width="100%" controls></video><br>
      <b>CPU model on 310p</b>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/b7023986-68a1-4e24-ac78-225c9f0e1dbf" width="100%" controls></video><br>
      <b>GPU model on 310p</b>
    </td>
  </tr>

  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/6f7ea2e7-16cc-4925-97ec-3522daac877e" width="100%" controls></video><br>
      <b>CPU model on agx</b>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/dafc86a4-9817-4b7c-9b27-4421efd44998" width="100%" controls></video><br>
      <b>GPU model on agx</b>
    </td>
  </tr>
</table>









