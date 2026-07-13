#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Low-overhead JSONL appends for MoE profiling hot paths."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from threading import Lock


_PROFILE_FLUSH_EVERY_ENV = "VLLM_ASCEND_MOE_PROFILE_FLUSH_EVERY"


class _JsonlAppendWriter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._fds: dict[str, int] = {}
        self._pending_lines: dict[str, int] = {}

    def append(self, path: str | Path, entry: dict[str, object]) -> None:
        normalized_path = str(path)
        if not normalized_path:
            return
        line = (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8")
        with self._lock:
            fd = self._fds.get(normalized_path)
            if fd is None:
                file_path = Path(normalized_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(
                    file_path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o644,
                )
                self._fds[normalized_path] = fd
                self._pending_lines[normalized_path] = 0
            _write_all(fd, line)
            pending = self._pending_lines.get(normalized_path, 0) + 1
            flush_every = _profile_flush_every()
            self._pending_lines[normalized_path] = (
                0 if flush_every <= 1 or pending >= flush_every else pending
            )

    def flush_all(self) -> None:
        with self._lock:
            for normalized_path in tuple(self._fds):
                self._pending_lines[normalized_path] = 0

    def close_all(self) -> None:
        with self._lock:
            for fd in tuple(self._fds.values()):
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds.clear()
            self._pending_lines.clear()


_writer = _JsonlAppendWriter()
atexit.register(_writer.close_all)


def append_jsonl(path: str | Path, entry: dict[str, object]) -> None:
    _writer.append(path, entry)


def flush_profile_writes() -> None:
    _writer.flush_all()


def close_profile_writes() -> None:
    _writer.close_all()


def _profile_flush_every() -> int:
    raw_value = os.getenv(_PROFILE_FLUSH_EVERY_ENV, "1")
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 1


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]
