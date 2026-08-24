#!/usr/bin/env python3
"""Package the configured raw roots into a portable GNU-tar/zstd archive."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from build_artifact_manifest import AUXILIARY_FILES, SOURCES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["tar", "--zstd", "-cf", str(output)]
    missing: list[str] = []
    source_commands: list[tuple[Path, str, str]] = []
    for root, prefix in SOURCES.values():
        if not root.is_dir():
            missing.append(str(root))
            continue
        # Keep each selected subdirectory under the archive prefix recorded by
        # build_artifact_manifest.py (some roots are children of a shared
        # top-level directory such as latchmoe-formal-20260824).
        transform = f"s,^%s,%s," % (root.name, prefix)
        command.append(f"--transform={transform}")
        source_commands.append((root.parent, root.name, prefix))
    repo_root = Path(__file__).resolve().parents[2]
    for path, archive_path in AUXILIARY_FILES.values():
        if not path.is_file():
            missing.append(str(path))
            continue
        relative = path.relative_to(repo_root)
        command.extend(["-C", str(repo_root), str(relative)])
    if missing:
        print("missing sources:")
        for item in missing:
            print(item)
        return 1
    for parent, name, _prefix in source_commands:
        command.extend(["-C", str(parent), name])
    subprocess.run(command, check=True)
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
