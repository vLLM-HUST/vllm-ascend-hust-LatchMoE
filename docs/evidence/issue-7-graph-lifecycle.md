# Issue 7 Graph Replay and Slot-Lifecycle Evidence

This record closes the graph replay, slot lifecycle, and H2D evidence contract
from [Issue #7](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/7).
The raw bundle is checked in at
[`bundles/issue-7-graph-lifecycle-743045d.tar.gz`](bundles/issue-7-graph-lifecycle-743045d.tar.gz).

## Scope and provenance

- Runtime evidence SHA: `743045dab5931092cb391d6489502e93382162b2`
- Parent SHA: `21ec4601ef9729bc47241ac3ff064828aef6736b`
- vLLM: `ad7125a431e176d4161099480a66f0169609a690`
- vLLM-Ascend seam: `4806367eeeb7d62b32078ae90cd929cc06d825fe`
- Model/config: Qwen3-30B-A3B BF16, config SHA-256
  `2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297`
- Workload manifest SHA-256:
  `3d30e76f651eadc3b2104ae550071efbd94717d8e773fceaff8707ea4c501c08`
- Hardware: physical NPU5, one Ascend 910B2, TP1
- Shape: `max_num_seqs=1`, client concurrency 1, prefix cache disabled
- Offload: 14 GiB, 12 managed layers, 32 fixed slots, default `multi_wave`
- Graph: PIECEWISE ACLGraph, `vllm::moe_offload_stage` splitting op

Every unit manifest reports `runtime_paths_dirty=false` and includes the full
command, selected environment, dependency commits, runtime-bundle identity,
model/workload hashes, and device. No unit used `--enforce-eager`, an eager
fallback, a reused server, or prefix-cache hits.

## Ordered gates

| Stage | Independent start | Requests | Tokens | TTFT p50 (ms) | TPOT p50 (ms/token) | Throughput (token/s) | HBM peak / final |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke | yes | 1/1 | 8 | 788.88 | 48.44 | 6.79 | 90% / 5% |
| short gate | yes | 11/11 | 1,408 | 572.61 | 54.84 | 16.79 | 91% / 5% |
| repeat 1 | yes | 11/11 | 1,408 | 573.64 | 55.33 | 16.52 | 91% / 5% |
| repeat 2 | yes | 11/11 | 1,408 | 537.70 | 55.93 | 16.50 | 91% / 5% |
| repeat 3 | yes | 11/11 | 1,408 | 573.11 | 53.56 | 17.08 | 91% / 5% |

For the three formal starts, mean TTFT p50 is 561.48 ms (population SD
16.82 ms), mean TPOT p50 is 54.94 ms/token (SD 1.01), and mean output
throughput is 16.70 token/s (SD 0.27). These are narrow qualification
measurements, not a broad or matched performance claim.

## Correctness and lifecycle gates

All five stages passed the fail-closed verifier:

- 12/12 managed layers had identical slot-bank and logical-map data pointers
  at capture lock and replay validation.
- Each recorded compute handle retained the same slot owner/generation until
  its completion event was synchronized.
- Every H2D decode miss installed the consumer dependency and published its
  logical mapping only after readiness.
- Every multi-wave active set stayed within the 32-slot capacity; the profile
  reported no late after-compute prefetch issue.
- Server logs contain explicit Graph capture and ACLGraph replay and contain no
  forbidden capture, stream, OOM, address, generation, or stale-H2D marker.
- Every manager stop produced `release_ack.json`; physical HBM returned to 5%
  in the post-release sample.

The short gate is the token oracle for repeat 1, and repeat 1 is the oracle for
repeats 2 and 3. Each comparison matched 11/11 request IDs and 1,408/1,408
output token IDs exactly. Thus the three formal starts matched 33/33 requests
and 4,224/4,224 tokens.

## Timing breakdown

The values below are aggregate profile totals for each 11-request formal
start. Decode sampling is expanded using its recorded sample rate. Graph
replay issue time is sampled CPU-side call/dispatch time around
`NPUGraph.replay()` without an NPU synchronize; it is not device-kernel time.

| Metric | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|
| H2D bytes | 347,533,737,984 | 347,533,737,984 | 347,533,737,984 |
| H2D copy enqueue (ms) | 2,983.07 | 2,944.86 | 2,906.10 |
| waiting/event (ms) | 887.91 | 888.51 | 905.81 |
| slot mapping update (ms) | 10,447.24 | 9,594.98 | 10,350.99 |
| wave MLP compute (ms) | 369.85 | 358.17 | 351.13 |
| wave stage issue (ms) | 1,059.55 | 1,026.95 | 1,031.56 |
| wave stage wait (ms) | 85.12 | 84.72 | 84.49 |
| sampled Graph replay issue, expanded (ms) | 7,425.52 | 7,686.52 | 6,902.08 |

Transfer/compute overlap is evidenced by wave scheduling: future-wave H2D is
issued before current-wave compute, stage wait is separately measured, and
`prefetch_after_compute_issues=0`. The physical-device HBM JSONL and runtime
memory ledger are both retained. The ledger reports 3,623,878,656 bytes of
persistent slots plus 603,979,776 bytes of two-buffer prefill staging (3.94
GiB total NPU slot/stage storage); all 12 original expert-weight banks were
released.

## Portable raw bundle

- Bundle SHA-256:
  `e6ffed05c0eec0a79174f0270f48455cb6c6247726f63e1a865a96ba3102c5fa`
- Compressed size: 572 KiB
- Contents: five complete unit directories, external release ACKs, and the
  generated JSON/CSV summary. Each unit contains manifest, result, server and
  client logs, launcher lifecycle, benchmark token arrays, profile JSONL, NPU
  samples, local release ACK, verifier report, and pass/fail marker.

Fresh-checkout verification:

```bash
mkdir /tmp/issue7-evidence
tar -xzf docs/evidence/bundles/issue-7-graph-lifecycle-743045d.tar.gz \
  -C /tmp/issue7-evidence
python benchmark/scripts/verify_issue7_graph_unit.py \
  --unit-dir /tmp/issue7-evidence/repeat-1/sew-offload-ascend-v1-20260812T100922Z/sew_14gb_autoslots/mixed_chat \
  --minimum-requests 11
```

This extraction/verification path was executed successfully. Artifact-level
SHA-256 values are stored in every `graph_correctness.json`.

## Legacy branch disposition

The commit-by-commit runtime/control/experiment/obsolete classification is in
[`../issue7_legacy_175_audit.md`](../issue7_legacy_175_audit.md). No unique
runtime hunk remained to migrate; control policy was extracted to
`intellistream/ascend-moe-control@8dadc76`; experiment-only and obsolete
material was retained only by source SHA. The remote legacy branch
`feature/latchmoe-issue-175-control-plane` is absent from `git ls-remote`, so
the requested ref deletion has already taken effect.

## Deliberate claim boundary

The former README capture-off table is removed because it lacked the repeated
raw evidence contract and is excluded from this graph-only qualification. The
bundle establishes the narrow graph replay and lifecycle claim. It does not
establish performance breadth across models, capacities, or concurrency.
