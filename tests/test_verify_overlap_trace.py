import json

from benchmark.scripts.verify_overlap_trace import verify_matched_overlap


def _trace(*, multistream, wrapper=False, hard_sync=False, profile_teardown=False):
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "Overlap Analysis"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 0, "args": {"name": "Communication"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 2, "args": {"name": "Computing"}},
        {"ph": "M", "name": "process_name", "pid": 2, "tid": 0, "args": {"name": "Ascend Hardware"}},
        {"ph": "X", "name": "Computing", "pid": 1, "tid": 2, "ts": 100, "dur": 80},
        {"ph": "X", "name": "vllm::moe_mlp_shared", "pid": 3, "tid": 3, "ts": 100, "dur": 100},
        {"ph": "X", "name": "acl_memcpy_host_to_device", "pid": 3, "tid": 3, "ts": 120, "dur": 20},
        {
            "ph": "X",
            "name": "MEMCPY_ASYNC",
            "pid": 2,
            "tid": 46,
            "ts": 125,
            "dur": 30,
            "args": {"Task Type": "PCIE_DMA_SQE"},
        },
        {"ph": "X", "name": "aclnnGroupedMatmul", "pid": 2, "tid": 46, "ts": 130, "dur": 45},
    ]
    if multistream:
        events.extend(
            [
                {"ph": "X", "name": "aclnnMatmul", "pid": 2, "tid": 43, "ts": 120, "dur": 30},
                {"ph": "X", "name": "aclnnMatmul", "pid": 2, "tid": 43, "ts": 160, "dur": 20},
            ]
        )
    else:
        events.extend(
            [
                {"ph": "X", "name": "aclnnMatmul", "pid": 2, "tid": 46, "ts": 100, "dur": 20},
                {"ph": "X", "name": "aclnnMatmul", "pid": 2, "tid": 46, "ts": 175, "dur": 20},
            ]
        )
    if hard_sync:
        events.append({"ph": "X", "name": "aclrtSynchronizeDevice", "pid": 2, "tid": 46, "ts": 250, "dur": 1})
    if profile_teardown:
        events.append({"ph": "X", "name": "PROFILING_DISABLE", "pid": 2, "tid": 49, "ts": 300, "dur": 1})
    return {"traceEvents": events} if wrapper else events


def _summary(multistream):
    return {
        "status": "ok",
        "graph_compatible_offload": True,
        "ascend_additional_config": {"multistream_overlap_shared_expert": multistream},
    }


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifacts(tmp_path, *, hard_sync=False):
    overlap_trace = tmp_path / "overlap_trace.json"
    control_trace = tmp_path / "control_trace.json"
    _write(overlap_trace, _trace(multistream=True, wrapper=True, hard_sync=hard_sync))
    _write(control_trace, _trace(multistream=False))
    overlap_summary = tmp_path / "overlap_summary.json"
    control_summary = tmp_path / "control_summary.json"
    _write(overlap_summary, _summary(True))
    _write(control_summary, _summary(False))
    overlap_output = tmp_path / "overlap_outputs.jsonl"
    control_output = tmp_path / "control_outputs.jsonl"
    overlap_output.write_text('{"text":"same"}\n', encoding="utf-8")
    control_output.write_text('{"text":"same"}\n', encoding="utf-8")
    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        json.dumps(
            {
                "name": "decode_fixed_slot_stage",
                "payload": {"h2d_bytes": 4096, "consumer_dependency_installed": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    overlap_log = tmp_path / "overlap.log"
    control_log = tmp_path / "control.log"
    overlap_log.write_text("Multistream overlap shared expert is enabled.\n", encoding="utf-8")
    control_log.write_text("PIECEWISE graph replay enabled\n", encoding="utf-8")
    return {
        "overlap_trace": overlap_trace,
        "control_trace": control_trace,
        "overlap_summary": overlap_summary,
        "control_summary": control_summary,
        "overlap_output": overlap_output,
        "control_output": control_output,
        "overlap_profile": profile,
        "overlap_log": overlap_log,
        "control_log": control_log,
    }


def test_verifier_accepts_incremental_raw_hardware_overlap(tmp_path):
    report = verify_matched_overlap(**_artifacts(tmp_path), min_incremental_overlap_us=10)

    assert report["status"] == "passed"
    assert report["incremental_shared_routed_compute_overlap_us"] == 35
    assert report["incremental_shared_h2d_overlap_us"] == 25
    assert all(check["passed"] for check in report["checks"])


def test_verifier_rejects_device_wide_sync(tmp_path):
    report = verify_matched_overlap(**_artifacts(tmp_path, hard_sync=True))

    assert report["status"] == "failed"
    sync_check = next(
        check for check in report["checks"] if check["name"] == "no_runtime_device_wide_sync"
    )
    assert sync_check["passed"] is False


def test_verifier_allows_profile_teardown_device_sync(tmp_path):
    artifacts = _artifacts(tmp_path)
    _write(
        artifacts["overlap_trace"],
        _trace(multistream=True, hard_sync=True, profile_teardown=True),
    )
    _write(
        artifacts["control_trace"],
        _trace(multistream=False, hard_sync=True, profile_teardown=True),
    )

    report = verify_matched_overlap(**artifacts)

    assert report["status"] == "passed"
    assert report["treatment"]["profile_teardown_device_sync_events"] == [
        "aclrtSynchronizeDevice"
    ]


def test_verifier_reports_unclassified_overlap_analysis_as_diagnostic(tmp_path):
    artifacts = _artifacts(tmp_path)
    _write(artifacts["overlap_trace"], _trace(multistream=True))

    report = verify_matched_overlap(**artifacts)

    assert report["status"] == "passed"
    assert report["diagnostics"]["overlap_analysis_intervals_present"] == {
        "treatment": False,
        "control": False,
    }
