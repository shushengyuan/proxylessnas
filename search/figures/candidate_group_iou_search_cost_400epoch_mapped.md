# Candidate Group IoU vs Search Cost, 400-Epoch Mapping

- Anchor folder: `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10`.
- Anchor best IoU within 400 epochs: `0.6690`.
- Anchor final/best IoU: `0.6855`.
- Mapping factor: `0.685500 / 0.669000 = 1.024664`.
- Mapped IoU formula: `candidate_400epoch_best_iou * mapping_factor`.
- Search cost: first 400 non-warmup search epochs, parsed from logged average batch time and multiplied by GPU count.

## Points

| Candidate group | Ops | 400ep IoU | mapped IoU | best epoch | search cost GPU-h | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `GroupConv` | 3 | 0.6510 | 0.6671 | 261 | 0.4266 | new 400ep search |
| `Group_Spa` | 6 | 0.5550 | 0.5687 | 346 | 0.3361 | new 400ep search |
| `Group_Spa_Res` | 9 | 0.5430 | 0.5564 | 346 | 0.3292 | new 400ep search |
| `Group_Spa_Res_MBConv` | 12 | 0.5900 | 0.6046 | 399 | 1.0460 | existing Res_Group_Spa_MBConv search, first 400 search epochs |
| `whole` | 18 | 0.6690 | 0.6855 | 373 | 4.5569 | best full retrain IoU anchor + existing full search cost |

## Reproduce

Run missing redesigned 3/6/9-op experiments if logs are absent:

```bash
bash search/run_candidate_group_400epoch_search.sh GroupConv 4 400epoch_curve_groupconv | tee search/logs/curve_groupconv_400.console.log
bash search/run_candidate_group_400epoch_search.sh Group_Spa 7 400epoch_curve_groupspa | tee search/logs/curve_groupspa_400.console.log
bash search/run_candidate_group_400epoch_search.sh Group_Spa_Res 6 400epoch_curve_groupspares | tee search/logs/curve_groupspares_400.console.log
```

Generate the plot:

```bash
/home/intern/anaconda3/envs/new_env/bin/python /home/intern/proxylessnas/search/plot_candidate_group_iou_cost_400epoch.py --logs-root /home/intern/proxylessnas/search/logs --output-dir /home/intern/proxylessnas/search/figures
```
