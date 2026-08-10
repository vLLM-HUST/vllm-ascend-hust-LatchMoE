# Legacy Issue #175 Branch Audit

This document records the disposition of the legacy mixed branch before its
branch ref is deleted.

## Scope

- Source branch: `feature/latchmoe-issue-175-control-plane`
- Source tip: `ec53fbc77fa0182fd28369a1ec44de853caf1df0`
- Merge base: `64583fa48288ee1e47fde4bf27363eec050d0e10`
- Audited target: LatchMoE `main@b08f26e29f96f9c93d69666336e06425d6738264`
- Control target: `intellistream/ascend-moe-control`
- Control extraction: `8dadc76fec0a190e9e61428e0cff5c164e8c69cc`
- Audit date: 2026-08-10

The source branch has three commits not reachable from LatchMoE `main`. It is
also 19 commits behind the audited target, so merging the branch as a whole
would reintroduce control-policy code and overwrite newer runtime and test
work.

## Classification Rules

- `runtime mechanism`: address-stable slots, Host Store/H2D, graph lifecycle,
  wave prefill, or the policy-neutral placement-plan adapter. Migrate only when
  current LatchMoE `main` does not already contain the behavior.
- `control policy`: Static, frequency-aware, route-transition-aware, EPLB
  comparisons, or policy experiment orchestration. This belongs in
  `intellistream/ascend-moe-control`.
- `experiment-only`: workload construction, historical comparison tools,
  result collection, and correctness helpers tied to the old Issue #175
  experiment contract. Record the source revision; do not mix it into the
  runtime migration.
- `obsolete`: code replaced by the graph-only harness, managed launcher,
  current runtime, or stricter lifecycle tests.

## Outcome

No additional runtime hunk needs to be migrated from the legacy branch.

| Category | Disposition |
|---|---|
| Runtime mechanism | Already present in current LatchMoE `main`; no cherry-pick |
| Control policy | Extracted and refined in Ascend MoE Control at `8dadc76` |
| Experiment-only | Retained as historical Issue #175 provenance by the source SHAs below; not promoted into the graph-only runtime path |
| Obsolete | Replaced by current graph-only validation, managed launch, and lifecycle tests |

## Commit `e50b022`: `sync-c0dab100`

Classification: mixed commit. It must not be cherry-picked.

### Control policy

The following content implements or evaluates Static, frequency-aware, and
route-transition-aware placement decisions:

- `benchmark/configs/issue175_control_plane.yaml`
- `benchmark/scripts/build_issue175_routes.py`
- `benchmark/scripts/run_issue175_complete.sh`
- `docs/issue175_execution.md`
- `tests/test_control_plane_policy.py`
- `tests/test_issue175_routes.py`
- policy-related hunks in `benchmark/scenarios/sew_offload_scenarios.json`,
  `benchmark/scripts/collect_evidence.py`, `benchmark/scripts/run_suite.py`,
  `benchmark/scripts/sew_bench.py`, and `tests/test_benchmark_design.py`
- `vllm_moe_offload_ascend/moe_offload/policy.py`
- policy exports and integration hunks in `moe_offload/__init__.py`,
  `config.py`, `runtime.py`, `slot_bank.py`, `moe_offload_stage_op.py`, and
  `patch_fused_moe.py`

Disposition: do not migrate these files to LatchMoE. The policy contract and
Static/LRU/frequency/transition implementations were extracted and tightened
in `intellistream/ascend-moe-control@8dadc76`. In particular, the extracted
Static policy requires an explicit finalized warm-up bucket instead of the
legacy implicit EWMA freeze.

### Runtime mechanism

The commit also carried runtime support used by its embedded controller. The
potentially reusable mechanisms were checked independently:

- `slot_readiness_for_experts` is present in current `runtime.py`, originally
  added on the main line by `8d5e46c`.
- B2 phase handling, exact pair waves, transfer readiness, and buffer-release
  event handling are present on the main line from `3a6c6d1` and subsequent
  Issue #4 runtime hardening.
- Address fingerprint, owner/generation, in-flight compute, H2D readiness, and
  fail-closed lifecycle checks are covered by the current Issue #4 runtime and
  tests from `aa5b77c` and later fixes.
- The policy-neutral external boundary is the validated placement-plan adapter
  from `c240e6e`, rather than the embedded policy executor in this commit.

Disposition: no runtime hunk is missing from current `main`.

### Experiment-only and obsolete

- `tools/compare_smoke_outputs.py` and `tests/test_correctness_oracle.py` are an
  old output-comparison helper. Issue #7 requires a new oracle aligned with the
  managed graph-only artifact contract, so this version is provenance, not the
  accepted oracle.
- `tools/collect_moe_trace.py` and `tests/test_collect_moe_trace.py` are tied to
  the old trace-only/eager-capable workflow. The current graph-only trace entry
  point is `benchmark/scripts/collect_moe_trace.py`.
- `benchmark/scripts/run_openai_manifest.py` and the old runner/collector hunks
  are experiment-only. The current service path is managed by
  `third_party/vllm-hust-dev-hub` and records release acknowledgement.
- The large removal from `tests/test_prefill_stage_runtime.py` is obsolete and
  must not be replayed because it would drop newer lifecycle coverage.
- The runtime ACLGraph monkey-patch counters are obsolete as an authority for
  Issue #7. Graph evidence must come from the pinned seam and the fail-closed
  artifact verifier, with native instrumentation added at the authoritative
  graph boundary when needed.

## Commit `bcd1f2c`: `sync-c0-dependencies`

Classification: experiment-only plus superseded configuration.

- The shared Qwen3 model path is already present in the current canonical
  config.
- `tools/sharegpt_manifest.py` is already represented by the maintained
  `benchmark/scripts/sharegpt_manifest.py` implementation.
- The removed eager/capture-off cases and old on-demand comparison wording are
  superseded by the graph-only preflight in `a78195b`.

Disposition: no migration required.

## Commit `ec53fbc`: `merge-main-b2-with-c0-control`

Classification: runtime reconciliation plus historical eager-baseline support.

The reusable B2/runtime changes are all present in current `main`, including:

- mixed-phase recognition and `_current_forward_phase`;
- small-route CPU counting and exact pair-wave overflow handling;
- device pair planning and device scatter descriptors;
- per-wave H2D estimation and optional profile detail emission;
- prefill stage-buffer release events;
- current B2 stage/compute/scatter profiling fields.

The on-demand eager staging path also exists in the code history, but it is
excluded from the Issue #7 graph-only acceptance path and must not be used as a
passing comparison.

Disposition: no cherry-pick required. Replaying this reconciliation commit
would overwrite newer seam compatibility and lifecycle fixes.

## Branch Deletion Decision

The branch can be deleted after this audit is merged and the current test suite
passes because:

1. no unique runtime mechanism remains to migrate;
2. control-policy ownership is established in Ascend MoE Control;
3. experiment-only material is explicitly tied to its original source commits
   and is not evidence for the current graph-only claims; and
4. obsolete changes have named replacements on current `main`.

Deleting the branch removes only the legacy branch ref. It does not revert any
current LatchMoE or Ascend MoE Control commit.
