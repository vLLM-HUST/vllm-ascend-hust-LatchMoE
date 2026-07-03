# Figure Data Manifest

| Figure/data file | Role | Source | Status |
|---|---|---|---|
| `benchmark/artifacts/reports/week2_smoke_20260630/week2_smoke_summary.csv` | Week2 smoke evidence table. | `collect_evidence.py` over six smoke unit artifacts. | Smoke-only, not paper result. |
| `benchmark/artifacts/reports/week2_smoke_20260630/week2_smoke_ttft.svg` | Initial TTFT plot. | Smoke summary CSV. | Planning figure. |
| `benchmark/artifacts/reports/week2_smoke_20260630/week2_smoke_tpot.svg` | Initial TPOT plot. | Smoke summary CSV. | Planning figure. |
| `benchmark/artifacts/reports/week2_smoke_20260630/week2_smoke_throughput.svg` | Initial throughput plot. | Smoke summary CSV. | Planning figure. |
| `benchmark/artifacts/reports/week2_smoke_20260630/week2_smoke_weights.svg` | Initial weight-memory plot. | Smoke summary CSV. | Planning figure. |
| `benchmark/artifacts/reports/week2_smoke_20260630/week2_smoke_kv_cache.svg` | Initial KV-cache plot. | Smoke summary CSV. | Planning figure. |
| `benchmark/artifacts/reports/week2_smoke_20260630/week2_smoke_h2d.svg` | Initial H2D profile plot. | Smoke summary CSV and profile JSONL. | Planning figure. |

Final paper figures must be regenerated from repeated full-workload raw
artifacts. The current SVG files exist because the `vllm-hust-dev` environment
does not include matplotlib; final plots should use Matplotlib/Seaborn in a
dedicated plotting environment.
