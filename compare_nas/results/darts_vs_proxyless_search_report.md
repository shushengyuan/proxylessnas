# DARTS vs ProxylessNAS: Search-Only Comparison

- Generated at: 2026-04-27T11:23:27
- Proxyless run: `0,1_NUAA-SIRST_Super_all_whole_all_08_04_2026_23_36_48`

## Search Setup Snapshot

| Dimension | ProxylessNAS | DARTS |
| --- | --- | --- |
| Search space | Edge/block operator choices (`whole_all`) | Cell DAG primitives |
| Proxy task | `NUAA-SIRST` | `cifar10` |
| Optimization | Gradient + binary path (`add#linear`) | Continuous alpha + bilevel (unrolled) |
| Epoch budget | warmup `100` + search `100` | see run dir `/home/intern/proxylessnas/third_party/pt.darts/searchs/cifar10_proxyless_compare_seed42_ep1` |
| Batch size | train `8` | from DARTS run log |

## Discovered Architecture Artifacts

- Proxyless net config: `/home/intern/proxylessnas/search/logs/0,1_NUAA-SIRST_Super_all_whole_all_08_04_2026_23_36_48/learned_net/net.config`
- Proxyless run config: `/home/intern/proxylessnas/search/logs/0,1_NUAA-SIRST_Super_all_whole_all_08_04_2026_23_36_48/learned_net/run.config`
- Proxyless approximate GPU-hour: `0.0`
- Proxyless GPU latency in search log: `8.225 ms`
- DARTS best top1 (proxy metric): `N/A`
- DARTS best genotype: `Genotype(normal=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], [('max_pool_3x3', 2), ('max_pool_3x3', 3)], [('max_pool_3x3', 1), ('max_pool_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], [('max_pool_3x3', 0), ('avg_pool_3x3', 2)], [('sep_conv_5x5', 3), ('skip_connect', 1)], [('avg_pool_3x3', 2), ('avg_pool_3x3', 0)]], reduce_concat=range(2, 6))`

## Conclusion Boundary

- This report compares **search behavior and searched structures**, not downstream task metrics.
- If you need strict task-level fairness, run a DARTS-style search on the same SIRST pipeline.
