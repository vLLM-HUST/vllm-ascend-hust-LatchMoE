# Result-Heavy Stage Gates

These gates protect the experiment layer from drifting away from the paper
story. A gate is closed only when the required artifacts are present and the
allowed claims are updated accordingly.

## Gate D0: Experiment Protocol Locked

Required artifacts:

- `plan/experiment-protocol.md`
- `benchmark/configs/sew_offload_v1.yaml`
- `benchmark/scenarios/sew_offload_scenarios.json`

Lock conditions:

- Dataset, workload buckets, seeds, model path, hardware, software stack, and
  log schema are recorded.
- Main setting is explicitly scoped as single-card, memory-constrained MoE
  offloading, not cloud serving throughput.
- Baseline fairness boundaries separate native prefetch, legacy layered, SEW
  eager, and SEW capture-on into separate server processes.
- The Ascend-UVA-like threat model is tracked as E0 with a 14 GiB runtime
  probe and separate U1-U4 gates before any CUDA-UVA-equivalence claim.

Status: open until the full-workload repeat rule is satisfied.

## Gate D1: Method-Experiment Traceability

Required artifact:

- `plan/review/method-experiment-traceability.md`

Lock conditions:

- Every contribution maps to at least one method module and one experiment.
- Every claim has a table or figure target.
- Any unsupported contribution is either removed from Introduction or marked as
  a limitation/future-work boundary.

Status: structurally complete; evidence status remains partial.

## Gate D2: Table/Figure Data Contract

Required artifacts:

- `tables/table-schema.md`
- `figures/data-manifest.md`
- raw run directories containing `unit_result.json`, `server.log`,
  `benchmark.json` when successful, and `moe_profile.jsonl` when profiling is
  enabled.

Lock conditions:

- Every final table has an aggregation rule.
- Every final figure names its raw data source.
- Smoke-only artifacts are clearly labeled and excluded from final paper
  claims.

Status: open.

## Gate D3: Main, Efficiency, Ablation, Robustness Results

Required result families:

- E0 Ascend-UVA-like feasibility as a pre-serving threat-model gate.
- E1 end-to-end performance.
- E2 memory feasibility and graph compatibility.
- E3 B2 prefill overflow.
- E4 mechanism ablation.
- E5 slot/budget sensitivity.
- E6 correctness and lifecycle safety.
- E7 concurrency robustness as secondary evidence.

Lock conditions:

- Each successful case has at least three repeated full-workload runs unless
  the table explicitly reports a failure case.
- Each failure case records a classified failure reason.
- Each result family has raw logs, aggregation output, table update, figure
  script, and prose note.

Status: open.

## Gate D4: Result Chapter Decontamination

Lock conditions:

- No planning notes remain inside Results prose.
- No mock or smoke values are described as final results.
- Prose uses measured values only from final data contracts.
- Phrases such as "experiment purpose", "table position", "discussion prompt",
  and temporary TODOs are removed from manuscript-facing sections.

Status: not started.

## Gate D5: Peer Review Pass

Required artifact:

- `plan/review/results-peer-review.md`

Lock conditions:

- A reviewer-style pass checks baseline fairness, missing metrics, unsupported
  claims, reproducibility, and figure readability.
- Any CRITICAL finding blocks submission.

Status: not started.
