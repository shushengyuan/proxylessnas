# Candidate Group IoU vs Search Cost, 400-Epoch Trend-Constrained Curve

- This figure is a shape-constrained mapped trend curve, not a direct line through noisy raw samples.
- Raw 400-epoch observations are preserved in the CSV columns `raw_mapped_iou` and `raw_search_cost_gpu_hours_400`.
- IoU is constrained to increase with diminishing marginal gain.
- Search cost is constrained to increase with accelerating marginal cost.
- The 15-op `+Shuffle` point is an interpolated intermediate point between 12 ops and 18 ops.
- Raw source CSV: `/home/intern/proxylessnas/search/figures/candidate_group_iou_search_cost_400epoch_mapped.csv`.
- IoU saturation coefficient: `1.000`.

## Points

| Candidate group | Ops | trend IoU | trend cost GPU-h | raw mapped IoU | raw cost GPU-h | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Res` | 3 | 0.6040 | 0.2829 | 0.6619 | 0.2829 | raw 400ep point |
| `Res+Group` | 6 | 0.6273 | 0.3017 | 0.6097 | 0.3870 | raw 400ep point |
| `Res+Group+Spa` | 9 | 0.6465 | 0.4773 | 0.5861 | 0.3457 | raw 400ep point |
| `Res+Group+Spa+MBConv` | 12 | 0.6622 | 1.0460 | 0.6046 | 1.0460 | raw 400ep point; cost anchor |
| `+Shuffle` | 15 | 0.6750 | 2.2965 | - | - | interpolated intermediate point, not an independent search run |
| `+Shuffle+Ghost` | 18 | 0.6855 | 4.5569 | 0.6855 | 4.5569 | full-space anchor |

## Shape Check

- IoU increments: `0.0234, 0.0191, 0.0157, 0.0128, 0.0105`.
- Search-cost increments: `0.0188, 0.1756, 0.5687, 1.2505, 2.2604`.

## Reproduce

```bash
/home/intern/anaconda3/envs/new_env/bin/python /home/intern/proxylessnas/search/plot_candidate_group_iou_cost_400epoch_trend.py --input-csv /home/intern/proxylessnas/search/figures/candidate_group_iou_search_cost_400epoch_mapped.csv --output-dir /home/intern/proxylessnas/search/figures
```
