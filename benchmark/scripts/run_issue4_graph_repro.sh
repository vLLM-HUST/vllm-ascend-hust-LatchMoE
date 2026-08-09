#!/usr/bin/env bash
set -euo pipefail

runtime_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
seam_root=""
model=""
device="0"
output_root="${runtime_root}/benchmark/artifacts/issue4"
original_args=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seam-root) seam_root="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [[ -z "$seam_root" || -z "$model" ]]; then
    printf 'usage: %s --seam-root PATH --model PATH [--device ID] [--output-root PATH]\n' "$0" >&2
    exit 2
fi

run_id="latchmoe_issue4_graph_npu${device}_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${output_root}/${run_id}"
mkdir -p "$run_dir"
printf '%q ' "$0" "${original_args[@]}" > "${run_dir}/command.txt"
printf '\n' >> "${run_dir}/command.txt"

export ASCEND_RT_VISIBLE_DEVICES="$device"
export PYTHONPATH="${seam_root}:${runtime_root}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE=1
export VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM=1
export VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD=1
export VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1
export VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL=0
export VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1
export VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=0
export VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS=16
export VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0
export VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES=1
export VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS=""
export VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
export VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1
export VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD=0
export SEW_OFFLOAD_PROBE=1
export SEW_SEAM_PROBE=1

status="failed"
trap 'python "${runtime_root}/benchmark/scripts/collect_issue4_manifest.py" --run-dir "$run_dir" --runtime-root "$runtime_root" --seam-root "$seam_root" --device "$device" --status "$status" --command-file "${run_dir}/command.txt"' EXIT

python "${runtime_root}/benchmark/scripts/collect_issue4_manifest.py" \
    --run-dir "$run_dir" \
    --runtime-root "$runtime_root" \
    --seam-root "$seam_root" \
    --device "$device" \
    --status started \
    --command-file "${run_dir}/command.txt"
npu-smi info > "${run_dir}/npu-smi-before.txt" 2>&1 || true

smoke_command=(
    python "${runtime_root}/benchmark/scripts/run_fixed_slot_smoke.py"
    --config "${runtime_root}/benchmark/configs/sew_offload_v1.yaml"
    --mode fixed_slot_sync
    --model "$model"
    --output-dir "$run_dir"
    --inline-prompt "Explain why fixed-address expert slots permit graph replay."
    --inline-max-output-tokens 4
    --max-model-len 64
    --max-num-seqs 1
    --max-num-batched-tokens 2
    --kv-cache-memory-mb 256
    --gpu-memory-utilization 0.4
    --disable-ascend-norm-quant-fusion
    --no-enforce-eager
    --num-slots 16
    --release-original-expert-weights
    --layered-runtime
    --fanout-threshold 0
)
printf '%q ' "${smoke_command[@]}" > "${run_dir}/smoke_command.txt"
printf '\n' >> "${run_dir}/smoke_command.txt"
"${smoke_command[@]}" 2>&1 | tee "${run_dir}/console.log"

python "${runtime_root}/benchmark/scripts/verify_issue4_graph_artifacts.py" --run-dir "$run_dir"
npu-smi info > "${run_dir}/npu-smi-after.txt" 2>&1 || true
status="passed"
