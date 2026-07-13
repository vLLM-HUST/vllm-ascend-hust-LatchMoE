import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from vllm_moe_offload_ascend.moe_offload.profile_io import (
    append_jsonl,
    close_profile_writes,
    flush_profile_writes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_append_jsonl_flushes_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_MOE_PROFILE_FLUSH_EVERY", raising=False)
    close_profile_writes()
    path = tmp_path / "profile.jsonl"

    append_jsonl(path, {"name": "a", "value": 1})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"name": "a", "value": 1}
    ]
    close_profile_writes()


def test_append_jsonl_keeps_records_valid_across_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_PROFILE_FLUSH_EVERY", "64")
    close_profile_writes()
    path = tmp_path / "profile.jsonl"
    code = textwrap.dedent(
        """
        import sys
        from vllm_moe_offload_ascend.moe_offload.profile_io import append_jsonl, close_profile_writes

        path = sys.argv[1]
        worker = int(sys.argv[2])
        for index in range(100):
            append_jsonl(
                path,
                {
                    "worker": worker,
                    "index": index,
                    "payload": "x" * 512,
                },
            )
        close_profile_writes()
        """
    )
    env = dict(os.environ)
    env["VLLM_ASCEND_MOE_PROFILE_FLUSH_EVERY"] = "64"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path), str(worker)],
            cwd=str(REPO_ROOT),
            env=env,
        )
        for worker in range(4)
    ]
    for proc in procs:
        assert proc.wait(timeout=30) == 0

    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(records) == 400
    assert sorted((item["worker"], item["index"]) for item in records) == [
        (worker, index) for worker in range(4) for index in range(100)
    ]


def test_append_jsonl_can_batch_flushes(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_PROFILE_FLUSH_EVERY", "3")
    close_profile_writes()
    path = tmp_path / "profile.jsonl"

    append_jsonl(path, {"name": "a"})
    append_jsonl(path, {"name": "b"})
    flush_profile_writes()

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"name": "a"},
        {"name": "b"},
    ]
    close_profile_writes()
