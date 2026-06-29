# Candidate Group IoU vs Search Cost

- Logs root: `/home/intern/proxylessnas/search/logs`
- Target hardware filter: `gpu`
- Include ablation dirs: `False`
- IoU: best value in `logs/valid_console.txt` from `val_mean_IOU` / `Validate_IoU`.
- Search cost: sum of final logged epoch average batch time times batch count, then multiplied by GPU count to get GPU-hours.
- Note: the cost is log-estimated train-loop GPU-hours; validation and checkpoint overhead are not included.

## Plotted Points

| Candidate group | Ops | IoU | Search cost (GPU-h) | Wall hours | Source run |
| --- | ---: | ---: | ---: | ---: | --- |
| `Res_Group_Spa_MBConv` | 12 | 0.6600 | 4.3318 | 2.1659 | `0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00` |
| `whole_all` | 18 | 0.6540 | 17.9035 | 8.9517 | `0,1_NUAA-SIRST_Super_all_whole_all_27_04_2026_00_04_38` |

## Not Plotted

- `whole`: no valid IoU log, no train-time log, search_epochs <= 0.

## Reproduce

```bash
/home/intern/anaconda3/envs/new_env/bin/python /home/intern/proxylessnas/search/plot_candidate_group_iou_cost.py --logs-root /home/intern/proxylessnas/search/logs --output-dir /home/intern/proxylessnas/search/figures --target-hardware gpu
```
