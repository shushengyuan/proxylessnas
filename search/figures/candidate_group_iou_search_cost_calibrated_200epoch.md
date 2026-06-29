# Candidate Group IoU vs Search Cost, Calibrated by 200 Epochs

- Anchor: `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10`.
- Anchor best IoU within 200 epochs: `0.6630`.
- Anchor final/best IoU: `0.6855`.
- Mapping factor: `0.685500 / 0.663000 = 1.033937`.
- Mapped IoU formula: `candidate_200epoch_best_iou * mapping_factor`.
- Search cost is parsed from `logs/train_console.txt` by summing logged average batch time for detected epochs and multiplying by GPU count.
- The plotted right axis uses warmup + search GPU-hours; search-only GPU-hours are also saved in CSV.

## Plotted Points

| Candidate group | 200ep IoU | mapped IoU | logged epochs | search cost GPU-h | cost source |
| --- | ---: | ---: | ---: | ---: | --- |
| `Res_Group_Spa_MBConv` | 0.4560 | 0.4715 | 1800 | 4.3318 | `0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00` |
| `whole` | 0.6630 | 0.6855 | 1600 | 17.9035 | `0,1_NUAA-SIRST_Super_all_whole_all_27_04_2026_00_04_38` |

## Reproduce

```bash
/home/intern/anaconda3/envs/new_env/bin/python /home/intern/proxylessnas/search/plot_candidate_group_iou_cost_calibrated.py --logs-root /home/intern/proxylessnas/search/logs --output-dir /home/intern/proxylessnas/search/figures
```
