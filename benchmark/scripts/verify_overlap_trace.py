# SPDX-License-Identifier: Apache-2.0
"""Verify a matched Ascend shared-expert overlap experiment.

Ascend's profiler emits an ``Overlap Analysis`` process in addition to raw
hardware-stream events.  CANN 9.0.1 does not consistently classify PCIe H2D
DMA in the analysis process, so the acceptance path uses raw hardware events:
the backend's ``vllm::moe_mlp_shared`` marker identifies the independent shared
stream, ``MEMCPY_ASYNC`` PCIe tasks identify staged H2D, and
``GroupedMatmul`` identifies routed MLP work.  ``Overlap Analysis`` remains a
diagnostic only.

The profiler has used both a top-level event array and a Chrome-style
``{"traceEvents": [...]}`` object.  Both encodings are supported.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_HARD_DEVICE_SYNC = re.compile(
    r"(?:aclrt)?SynchronizeDevice|cudaDeviceSynchronize", re.IGNORECASE
)
_EAGER_FALLBACK = re.compile(
    r"(?:eager fallback|fall(?:ing)? back to eager|fallback_to_eager)",
    re.IGNORECASE,
)
_GROUPED_MATMUL = re.compile(r"GroupedMatmul", re.IGNORECASE)
_MATMUL = re.compile(r"Matmul", re.IGNORECASE)
_SHARED_MARKER = "vllm::moe_mlp_shared"
_H2D_MARKER = "acl_memcpy_host_to_device"
_PROFILE_DISABLE = "PROFILING_DISABLE"


class VerificationError(ValueError):
    """An artifact cannot support an overlap conclusion."""


@dataclass(frozen=True)
class Interval:
    start_us: float
    end_us: float

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _event_id(value: Any) -> str:
    return str(value)


def load_trace_events(path: str | Path) -> list[dict[str, Any]]:
    """Load either Ascend's event-array or the Chrome trace wrapper format."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read trace {source}: {exc}") from exc
    if isinstance(payload, Mapping):
        payload = payload.get("traceEvents")
    if not isinstance(payload, list):
        raise VerificationError(f"trace {source} is neither an event array nor traceEvents")
    events = [event for event in payload if isinstance(event, dict)]
    if not events:
        raise VerificationError(f"trace {source} has no events")
    return events


def _process_names(events: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        args = event.get("args")
        if isinstance(args, Mapping) and isinstance(args.get("name"), str):
            names[_event_id(event.get("pid"))] = args["name"]
    return names


def _thread_names(events: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    names: dict[tuple[str, str], str] = {}
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "thread_name":
            continue
        args = event.get("args")
        if isinstance(args, Mapping) and isinstance(args.get("name"), str):
            names[(_event_id(event.get("pid")), _event_id(event.get("tid")))] = args["name"]
    return names


def _merged(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(intervals, key=lambda item: (item.start_us, item.end_us))
    result: list[Interval] = []
    for current in ordered:
        if current.end_us <= current.start_us:
            continue
        if result and current.start_us <= result[-1].end_us:
            previous = result[-1]
            result[-1] = Interval(previous.start_us, max(previous.end_us, current.end_us))
        else:
            result.append(current)
    return result


def _intersection_us(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    total = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i].start_us, right[j].start_us)
        end = min(left[i].end_us, right[j].end_us)
        if end > start:
            total += end - start
        if left[i].end_us <= right[j].end_us:
            i += 1
        else:
            j += 1
    return total


def _intervals(
    events: Iterable[Mapping[str, Any]], *, pid: str, tids: set[str]
) -> list[Interval]:
    values: list[Interval] = []
    for event in events:
        if event.get("ph") != "X" or _event_id(event.get("pid")) != pid:
            continue
        if _event_id(event.get("tid")) not in tids:
            continue
        start = _number(event.get("ts"))
        duration = _number(event.get("dur"))
        if start is not None and duration is not None and duration > 0:
            values.append(Interval(start, start + duration))
    return _merged(values)


def _event_interval(event: Mapping[str, Any]) -> Interval | None:
    if event.get("ph") != "X":
        return None
    start = _number(event.get("ts"))
    duration = _number(event.get("dur"))
    if start is None or duration is None or duration <= 0:
        return None
    return Interval(start, start + duration)


def _intersects_any(interval: Interval, windows: Sequence[Interval]) -> bool:
    return _intersection_us([interval], windows) > 0


def _hardware_overlap_facts(
    events: Sequence[Mapping[str, Any]],
    *,
    hardware_pids: set[str],
) -> dict[str, Any]:
    """Identify shared/routed/H2D hardware intervals without CANN labels."""

    shared_markers = _merged(
        interval
        for event in events
        if str(event.get("name", "")) == _SHARED_MARKER
        for interval in [_event_interval(event)]
        if interval is not None
    )
    h2d_launches = _merged(
        interval
        for event in events
        if str(event.get("name", "")) == _H2D_MARKER
        for interval in [_event_interval(event)]
        if interval is not None
    )
    hardware_events: list[tuple[Interval, str, str, str]] = []
    for event in events:
        if _event_id(event.get("pid")) not in hardware_pids:
            continue
        interval = _event_interval(event)
        if interval is None:
            continue
        args = event.get("args")
        task_type = ""
        if isinstance(args, Mapping):
            task_type = str(args.get("Task Type", ""))
        hardware_events.append(
            (
                interval,
                str(event.get("name", "")),
                _event_id(event.get("tid")),
                task_type,
            )
        )

    routed_streams = {
        stream
        for _, name, stream, _ in hardware_events
        if _GROUPED_MATMUL.search(name)
    }
    candidate_counts: dict[str, int] = {}
    candidate_intervals: dict[str, list[Interval]] = {}
    for interval, name, stream, _ in hardware_events:
        if stream in routed_streams:
            continue
        if not _MATMUL.search(name) or _GROUPED_MATMUL.search(name):
            continue
        if not _intersects_any(interval, shared_markers):
            continue
        candidate_counts[stream] = candidate_counts.get(stream, 0) + 1
        candidate_intervals.setdefault(stream, []).append(interval)

    # External shared experts issue two projections per marker.  Requiring a
    # substantial fraction makes an unrelated short-lived MatMul stream fail
    # closed instead of being mistaken for the shared lane.
    minimum_shared_projections = max(2, (2 * len(shared_markers) + 2) // 3)
    shared_streams = {
        stream
        for stream, count in candidate_counts.items()
        if count >= minimum_shared_projections
    }
    shared_compute = _merged(
        interval
        for stream in shared_streams
        for interval in candidate_intervals[stream]
    )
    routed_compute = _merged(
        interval
        for interval, name, _, _ in hardware_events
        if _GROUPED_MATMUL.search(name)
    )

    def is_matched_h2d(interval: Interval) -> bool:
        # A traced host-to-device launch and its PCIe DMA task use the same
        # clock domain but the task may be queued shortly after the host call.
        return any(
            launch.start_us - 50.0 <= interval.start_us <= launch.end_us + 1_000.0
            for launch in h2d_launches
        )

    h2d_dma = _merged(
        interval
        for interval, name, _, task_type in hardware_events
        if name == "MEMCPY_ASYNC"
        and task_type == "PCIE_DMA_SQE"
        and is_matched_h2d(interval)
    )
    return {
        "shared_marker_count": len(shared_markers),
        "shared_streams": sorted(shared_streams),
        "shared_projection_count": sum(
            candidate_counts[stream] for stream in shared_streams
        ),
        "routed_grouped_matmul_count": sum(
            1 for _, name, _, _ in hardware_events if _GROUPED_MATMUL.search(name)
        ),
        "h2d_launch_count": len(h2d_launches),
        "h2d_dma_count": len(h2d_dma),
        "shared_routed_compute_overlap_us": round(
            _intersection_us(shared_compute, routed_compute), 3
        ),
        "shared_h2d_overlap_us": round(
            _intersection_us(shared_compute, h2d_dma), 3
        ),
    }


def _classify_device_syncs(
    events: Sequence[Mapping[str, Any]],
    *,
    hardware_pids: set[str],
) -> tuple[list[str], list[str]]:
    """Separate a profile-stop flush from a device sync during serving."""

    profile_stops = [
        interval
        for event in events
        if str(event.get("name", "")) == _PROFILE_DISABLE
        for interval in [_event_interval(event)]
        if interval is not None
    ]
    syncs = [
        (str(event.get("name", "")), interval)
        for event in events
        if _HARD_DEVICE_SYNC.search(str(event.get("name", "")))
        for interval in [_event_interval(event)]
        if interval is not None
    ]
    hardware_work = [
        interval
        for event in events
        if _event_id(event.get("pid")) in hardware_pids
        and str(event.get("name", "")) != _PROFILE_DISABLE
        and not _HARD_DEVICE_SYNC.search(str(event.get("name", "")))
        for interval in [_event_interval(event)]
        if interval is not None
    ]
    last_work_end = max((interval.end_us for interval in hardware_work), default=float("-inf"))
    teardown: list[str] = []
    runtime: list[str] = []
    for name, interval in syncs:
        stop_after_sync = any(
            0 <= stop.start_us - interval.end_us <= 100_000.0
            for stop in profile_stops
        )
        if interval.start_us >= last_work_end and stop_after_sync:
            teardown.append(name)
        else:
            runtime.append(name)
    return teardown, runtime


def analyze_trace(path: str | Path) -> dict[str, Any]:
    """Extract only profiler facts that are relevant to overlap acceptance."""

    events = load_trace_events(path)
    process_names = _process_names(events)
    thread_names = _thread_names(events)
    overlap_pids = {
        pid for pid, name in process_names.items() if name.strip().lower() == "overlap analysis"
    }
    if len(overlap_pids) != 1:
        raise VerificationError(
            f"trace {path} must expose one Overlap Analysis process, found {len(overlap_pids)}"
        )
    overlap_pid = next(iter(overlap_pids))
    communication_tids = {
        tid
        for (pid, tid), name in thread_names.items()
        if pid == overlap_pid and name.strip().lower() == "communication"
    }
    computing_tids = {
        tid
        for (pid, tid), name in thread_names.items()
        if pid == overlap_pid and name.strip().lower() == "computing"
    }
    if not communication_tids or not computing_tids:
        raise VerificationError(
            f"trace {path} is missing Overlap Analysis communication or computing threads"
        )
    communication = _intervals(events, pid=overlap_pid, tids=communication_tids)
    computing = _intervals(events, pid=overlap_pid, tids=computing_tids)
    hardware_pids = {
        pid for pid, name in process_names.items() if name.strip().lower() == "ascend hardware"
    }
    hardware_streams = {
        _event_id(event.get("tid"))
        for event in events
        if event.get("ph") == "X"
        and _event_id(event.get("pid")) in hardware_pids
        and (_number(event.get("dur")) or 0) > 0
    }
    hardware_facts = _hardware_overlap_facts(events, hardware_pids=hardware_pids)
    teardown_syncs, runtime_syncs = _classify_device_syncs(
        events,
        hardware_pids=hardware_pids,
    )
    event_synchronize_count = sum(
        1
        for event in events
        if str(event.get("name", "")).lower() == "event::synchronize"
    )

    return {
        "trace": str(Path(path)),
        "event_count": len(events),
        "overlap_analysis_pid": overlap_pid,
        "communication_interval_count": len(communication),
        "computing_interval_count": len(computing),
        "overlap_analysis_intervals_present": bool(communication and computing),
        "communication_us": round(sum(item.duration_us for item in communication), 3),
        "computing_us": round(sum(item.duration_us for item in computing), 3),
        "communication_compute_overlap_us": round(
            _intersection_us(communication, computing), 3
        ),
        "ascend_hardware_stream_count": len(hardware_streams),
        "profile_teardown_device_sync_events": teardown_syncs,
        "runtime_device_sync_events": runtime_syncs,
        # The profiler itself may emit Event::synchronize while flushing.
        # Report it for review, but do not mistake it for aclrtSynchronizeDevice.
        "event_synchronize_count": event_synchronize_count,
        **hardware_facts,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"JSON artifact {source} must be an object")
    return payload


def _additional_config(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = summary.get("ascend_additional_config")
    if isinstance(direct, Mapping):
        return direct
    diagnostics = summary.get("qualification_diagnostics")
    if isinstance(diagnostics, Mapping):
        nested = diagnostics.get("ascend_additional_config")
        if isinstance(nested, Mapping):
            return nested
    return {}


def _has_h2d_stage(path: str | Path) -> tuple[int, int]:
    stages = 0
    dependent_stages = 0
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VerificationError(f"cannot read profile {source}: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"invalid JSONL in {source}: {exc}") from exc
        if not isinstance(event, Mapping) or "stage" not in str(event.get("name", "")):
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or (_number(payload.get("h2d_bytes")) or 0) <= 0:
            continue
        stages += 1
        if payload.get("consumer_dependency_installed") is True:
            dependent_stages += 1
    return stages, dependent_stages


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise VerificationError(f"cannot read log {path}: {exc}") from exc


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def verify_matched_overlap(
    *,
    overlap_trace: str | Path,
    control_trace: str | Path,
    overlap_summary: str | Path,
    control_summary: str | Path,
    overlap_output: str | Path,
    control_output: str | Path,
    overlap_profile: str | Path,
    overlap_log: str | Path,
    control_log: str | Path,
    min_incremental_overlap_us: float = 1.0,
) -> dict[str, Any]:
    """Return an auditable report for a matched treatment/control pair."""

    treatment = analyze_trace(overlap_trace)
    control = analyze_trace(control_trace)
    treatment_summary = _read_json(overlap_summary)
    control_summary_data = _read_json(control_summary)
    treatment_h2d, treatment_dependencies = _has_h2d_stage(overlap_profile)
    treatment_log = _read_text(overlap_log)
    control_log_text = _read_text(control_log)
    treatment_config = _additional_config(treatment_summary)
    control_config = _additional_config(control_summary_data)
    treatment_analysis_overlap = float(treatment["communication_compute_overlap_us"])
    control_analysis_overlap = float(control["communication_compute_overlap_us"])
    treatment_shared_routed = float(treatment["shared_routed_compute_overlap_us"])
    control_shared_routed = float(control["shared_routed_compute_overlap_us"])
    treatment_shared_h2d = float(treatment["shared_h2d_overlap_us"])
    control_shared_h2d = float(control["shared_h2d_overlap_us"])
    incremental_shared_routed = treatment_shared_routed - control_shared_routed
    incremental_shared_h2d = treatment_shared_h2d - control_shared_h2d
    output_equal = Path(overlap_output).read_bytes() == Path(control_output).read_bytes()

    checks = [
        _check(
            "treatment_multistream_enabled",
            treatment_config.get("multistream_overlap_shared_expert") is True
            and "Multistream overlap shared expert is enabled" in treatment_log,
            "treatment config and backend log must both enable shared-expert multistream",
        ),
        _check(
            "control_multistream_disabled",
            control_config.get("multistream_overlap_shared_expert") is not True
            and "Multistream overlap shared expert is enabled" not in control_log_text,
            "control must not enable the shared-expert multistream backend",
        ),
        _check(
            "piecewise_graph_no_fallback",
            treatment_summary.get("graph_compatible_offload") is True
            and control_summary_data.get("graph_compatible_offload") is True
            and not _EAGER_FALLBACK.search(treatment_log)
            and not _EAGER_FALLBACK.search(control_log_text),
            "both runs must remain graph-compatible and contain no eager fallback",
        ),
        _check(
            "output_exactness",
            output_equal,
            "treatment and control outputs must be byte-identical",
        ),
        _check(
            "h2d_with_consumer_dependency",
            treatment_h2d > 0 and treatment_h2d == treatment_dependencies,
            f"treatment has {treatment_h2d} H2D stages and {treatment_dependencies} dependency-protected stages",
        ),
        _check(
            "ascend_hardware_streams",
            int(treatment["ascend_hardware_stream_count"]) >= 2
            and int(control["ascend_hardware_stream_count"]) >= 1,
            "profiler must contain actual Ascend Hardware stream intervals",
        ),
        _check(
            "treatment_independent_shared_stream",
            bool(treatment["shared_streams"])
            and int(treatment["shared_projection_count"]) >= 2,
            "treatment must expose a dedicated hardware stream with shared-projection MatMul events",
        ),
        _check(
            "control_has_no_independent_shared_stream",
            not control["shared_streams"],
            "no-overlap control must not expose a dedicated shared hardware stream",
        ),
        _check(
            "incremental_shared_routed_compute_overlap",
            incremental_shared_routed >= min_incremental_overlap_us,
            "treatment shared/routed={:.3f}us, control={:.3f}us, increment={:.3f}us, required>={:.3f}us".format(
                treatment_shared_routed,
                control_shared_routed,
                incremental_shared_routed,
                min_incremental_overlap_us,
            ),
        ),
        _check(
            "incremental_shared_h2d_overlap",
            incremental_shared_h2d >= min_incremental_overlap_us,
            "treatment shared/H2D={:.3f}us, control={:.3f}us, increment={:.3f}us, required>={:.3f}us".format(
                treatment_shared_h2d,
                control_shared_h2d,
                incremental_shared_h2d,
                min_incremental_overlap_us,
            ),
        ),
        _check(
            "no_runtime_device_wide_sync",
            not treatment["runtime_device_sync_events"]
            and not control["runtime_device_sync_events"],
            "device-wide sync is allowed only at the profiler teardown boundary, never during serving",
        ),
    ]
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "checks": checks,
        "treatment": treatment,
        "control": control,
        "diagnostics": {
            "overlap_analysis_intervals_present": {
                "treatment": bool(treatment["overlap_analysis_intervals_present"]),
                "control": bool(control["overlap_analysis_intervals_present"]),
            },
            "communication_compute_overlap_us": {
                "treatment": treatment_analysis_overlap,
                "control": control_analysis_overlap,
            },
        },
        "incremental_shared_routed_compute_overlap_us": round(
            incremental_shared_routed,
            3,
        ),
        "incremental_shared_h2d_overlap_us": round(incremental_shared_h2d, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix, label in (("overlap", "multistream treatment"), ("control", "no-overlap control")):
        parser.add_argument(f"--{prefix}-trace", required=True, help=f"{label} trace_view.json")
        parser.add_argument(f"--{prefix}-summary", required=True, help=f"{label} smoke summary.json")
        parser.add_argument(f"--{prefix}-output", required=True, help=f"{label} outputs.jsonl")
        parser.add_argument(f"--{prefix}-log", required=True, help=f"{label} driver log")
    parser.add_argument("--overlap-profile", required=True, help="treatment moe_offload_profile.jsonl")
    parser.add_argument("--report", required=True, help="output JSON verification report")
    parser.add_argument("--min-incremental-overlap-us", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify_matched_overlap(
            overlap_trace=args.overlap_trace,
            control_trace=args.control_trace,
            overlap_summary=args.overlap_summary,
            control_summary=args.control_summary,
            overlap_output=args.overlap_output,
            control_output=args.control_output,
            overlap_profile=args.overlap_profile,
            overlap_log=args.overlap_log,
            control_log=args.control_log,
            min_incremental_overlap_us=args.min_incremental_overlap_us,
        )
    except VerificationError as exc:
        report = {"status": "failed", "error": str(exc)}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
