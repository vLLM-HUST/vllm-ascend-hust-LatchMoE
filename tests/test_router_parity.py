from __future__ import annotations

import json

import torch

from vllm_moe_offload_ascend.moe_offload.router_parity import (
    compare_router_artifacts,
    record_router_snapshot,
)


def _record(*, layer: int, ids, weights, logits=None):
    return {
        "layer_id": layer,
        "topk_ids": ids,
        "topk_weights": weights,
        "router_logits": logits if logits is not None else [[0.0, 0.0]],
    }


def test_router_parity_accepts_exact_ids_and_weights():
    native = [_record(layer=3, ids=[[1, 2]], weights=[[0.75, 0.25]])]
    seam = [_record(layer=3, ids=[[1, 2]], weights=[[0.75, 0.25]])]

    assert compare_router_artifacts(native, seam)["status"] == "passed"


def test_router_parity_reports_first_topk_token_and_position():
    native = [_record(layer=7, ids=[[1, 2], [3, 4]], weights=[[0.7, 0.3], [0.6, 0.4]])]
    seam = [_record(layer=7, ids=[[1, 2], [8, 4]], weights=[[0.7, 0.3], [0.6, 0.4]])]

    report = compare_router_artifacts(native, seam)

    assert report["status"] == "failed"
    assert report["layer_id"] == 7
    assert report["mismatch"] == {
        "field": "topk_ids",
        "flat_index": 2,
        "native": 3,
        "seam": 8,
        "token": 1,
        "expert_position": 0,
    }


def test_router_parity_uses_configured_floating_tolerance():
    native = [_record(layer=1, ids=[[1]], weights=[[0.5000]])]
    seam = [_record(layer=1, ids=[[1]], weights=[[0.5004]])]

    assert compare_router_artifacts(native, seam, atol=0.001)["status"] == "passed"
    assert compare_router_artifacts(native, seam, atol=0.0001)["status"] == "failed"


def test_router_parity_rejects_missing_snapshots():
    assert compare_router_artifacts([], [_record(layer=1, ids=[[1]], weights=[[1.0]])]) == {
        "status": "failed",
        "reason": "no_native_records",
    }
    assert compare_router_artifacts([_record(layer=1, ids=[[1]], weights=[[1.0]])], []) == {
        "status": "failed",
        "reason": "no_seam_records",
    }


def test_router_snapshot_is_bounded_and_ignores_non_tensor_values(tmp_path, monkeypatch):
    path = tmp_path / "router.jsonl"
    monkeypatch.setenv("VLLM_ASCEND_MOE_ROUTER_PARITY_PATH", str(path))
    monkeypatch.setenv("VLLM_ASCEND_MOE_ROUTER_PARITY_MAX_TOKENS", "1")

    record_router_snapshot(
        role="seam",
        layer_id=1,
        router_logits=torch.ones((2, 3)),
        topk_ids=torch.ones((2, 1), dtype=torch.int32),
        topk_weights=torch.ones((2, 1)),
    )
    record_router_snapshot(
        role="seam",
        layer_id=1,
        router_logits="not-a-tensor",
        topk_ids=torch.ones((1, 1), dtype=torch.int32),
        topk_weights=torch.ones((1, 1)),
    )
    assert not path.exists()

    record_router_snapshot(
        role="seam",
        layer_id=1,
        router_logits=torch.ones((1, 3)),
        topk_ids=torch.ones((1, 1), dtype=torch.int32),
        topk_weights=torch.ones((1, 1)),
    )
    assert json.loads(path.read_text()) ["layer_id"] == 1
