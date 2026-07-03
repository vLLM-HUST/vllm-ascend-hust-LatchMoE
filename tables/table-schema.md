# Table Schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| T1: Benchmark feasibility | Show success/failure under each case. | Cases x workloads. | status, failure reason, graph capture completed. | `unit_result.json`, `server.log`. | Experiment runner. |
| T2: End-to-end serving | Main performance comparison. | Successful cases x workloads x repetitions. | TTFT, TPOT, throughput, request success. | `benchmark.json`. | Experiment runner. |
| T3: Memory cost | Explain HBM and CPU memory tradeoff. | Cases x workloads. | weights GB, available KV GiB, slot-bank GiB, host-store GiB. | `server.log`, `moe_profile.jsonl`. | Experiment runner. |
| T4: Offload overhead | Explain expert movement cost. | SEW cases x workloads. | H2D bytes, stage ms, active experts, wave count. | `moe_profile.jsonl`. | Experiment runner. |
| T5: Ablation | Attribute mechanism value. | Full SEW and ablation variants. | TTFT, TPOT, H2D bytes, failures. | Full raw run dirs. | Experiment runner. |

Aggregation rule: paper tables use repeated full workloads, not smoke runs.
Report median and tail metrics from per-request records when available.
