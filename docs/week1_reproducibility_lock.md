# Week 1 Claim and Reproducibility Lock

Date: 2026-06-30

User constraint: do not execute or record the vLLM-Ascend commit. This document
therefore intentionally omits any vLLM-Ascend commit field.

## Frozen Claim

System name: SEW-Offload.

Paper target: English full paper for a CCF-A-level systems venue.

Thesis:

> SEW-Offload makes dynamic MoE expert offloading compatible with Ascend
> ACLGraph replay by virtualizing logical experts into compute-protected,
> fixed-address physical slots and moving dynamic staging decisions outside the
> graph replay boundary.

Non-goal for Week 1: produce final paper numbers. Week-1 smoke results only
validate that each configuration launches, writes artifacts, and exposes the
expected logs/profile files.

## Clean Smoke Config Set

Config file: `demo/week1_smoke_config.json`

Cases:

| Case | Role | Purpose |
|---|---|---|
| `no_offload_capacity_probe` | baseline | Capacity probe without expert offload. Failure/OOM can still be useful evidence. |
| `native_prefetch_14gb` | baseline | Native prefetch offload baseline, separated from SEW env. |
| `legacy_layered_14gb` | baseline | Plugin AutoConfig legacy/layered path without SEW dataplane. |
| `sew_14gb_slots32` | main | Graph-compatible SEW fixed-slot dataplane at 14 GiB. |

Smoke command:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python tools/run_annual_demo_suite.py \
  --config demo/week1_smoke_config.json \
  --output-root demo_runs/week1_smoke \
  --python /root/miniconda3/envs/vllm-hust-dev/bin/python
```

Run a single case with:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python tools/run_annual_demo_suite.py \
  --config demo/week1_smoke_config.json \
  --output-root demo_runs/week1_smoke \
  --case sew_14gb_slots32 \
  --python /root/miniconda3/envs/vllm-hust-dev/bin/python
```

## Environment Snapshot

Conda:

| Field | Value |
|---|---|
| Conda binary | `/root/miniconda3/bin/conda` |
| Environment | `vllm-hust-dev` |
| Python | `3.11.15` |
| Python executable | `/root/miniconda3/envs/vllm-hust-dev/bin/python` |

Python packages:

| Package | Value |
|---|---|
| vLLM | `0.20.1.post1.dev363+gec4847981` |
| vLLM module path | `/root/vllm-hust/vllm/__init__.py` |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0` |
| Local plugin package | `vllm-moe-offload-ascend==0.1.0` |

Ascend stack:

| Component | Value |
|---|---|
| CANN OPP | `8.5.1` |
| Driver | `25.2.1` |
| ATB | `8.5.1.B080` |
| npu-smi | `25.2.1` |

Hardware selection:

| Field | Value |
|---|---|
| Selected smoke NPU | `ASCEND_RT_VISIBLE_DEVICES=4` |
| Observed device type | `910B2` |
| Rationale | NPU 4 had no running process in the initial `npu-smi info` snapshot and only about 3.4 GiB HBM in use. |

Model and dataset:

| Field | Value |
|---|---|
| Model | `/data/shared_models/modelscope_cache/Qwen/Qwen3-30B-A3B` |
| Model size on disk | `57G` |
| Dataset | `/data/shared_datasets/ShareGPT_V3_unfiltered_cleaned_split.json` |
| Dataset size on disk | `642M` |

Plugin repository:

| Field | Value |
|---|---|
| Repository | `/root/vllm-moe-offload-ascend` |
| Branch | `main` |
| HEAD | `7c8a4fa5ba01ca98c23a4c472320aae957f682ac` |
| Worktree | Dirty; see `git status --short --untracked-files=all` before interpreting any result as reproducible. |

## Verification

Unit test command:

```bash
/root/miniconda3/bin/conda run -n vllm-hust-dev python -m pytest tests -q
```

Latest observed result:

```text
103 passed, 3 warnings in 16.76s
```

Actual smoke execution summary: `docs/week1_execution_summary.md`.

## Smoke Artifact Checklist

For each case, confirm:

1. `case_manifest.json` exists and records server command, benchmark command,
   and selected `VLLM_ASCEND_MOE_*` environment variables.
2. `server.log` exists, even for expected failure/OOM cases.
3. `benchmark.json` exists for successful serving cases.
4. `case_result.json` and `summary.md` exist for successful serving cases.
5. `moe_profile.jsonl` exists for SEW and legacy/layered cases when profile
   hooks are reached.
6. Any failure is recorded as a launch/runtime outcome, not silently ignored.
