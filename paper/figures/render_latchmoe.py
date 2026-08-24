#!/usr/bin/env python3
"""Render the accepted LatchMoE architecture FigureSpec to SVG and PDF."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = HERE / "specs" / "latchmoe_architecture.json"
SVG = HERE / "latchmoe_architecture.svg"
PDF = HERE / "latchmoe_architecture.pdf"
PNG = HERE / "latchmoe_architecture.png"
DEFAULT_RENDERER = Path(
    "/root/.codex/aris_repo/skills/figure-spec/scripts/figure_renderer.py"
)


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument(
        "--cairosvg",
        default=shutil.which("cairosvg"),
        help="CairoSVG executable (install paper/figures/requirements.txt)",
    )
    args = parser.parse_args()
    if not args.cairosvg:
        parser.error("CairoSVG is required; pass --cairosvg or install the pinned requirements")
    if shutil.which("gs") is None:
        parser.error("Ghostscript (gs) is required")

    run("python3", str(args.renderer), "validate", str(SPEC))
    run("python3", str(args.renderer), "render", str(SPEC), "--output", str(SVG))

    # Darken renderer-default group labels for grayscale conference printing.
    svg = SVG.read_text(encoding="utf-8")
    for old in ("#999999", "#888888", "#777777"):
        svg = svg.replace(old, "#666666")
    SVG.write_text(svg, encoding="utf-8")

    raw_pdf = HERE / ".latchmoe_architecture.raw.pdf"
    try:
        run(str(args.cairosvg), str(SVG), "-o", str(raw_pdf))
        run(str(args.cairosvg), str(SVG), "-o", str(PNG), "-s", "1.5")
        run(
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            f"-sOutputFile={PDF}",
            str(raw_pdf),
        )
    finally:
        raw_pdf.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
