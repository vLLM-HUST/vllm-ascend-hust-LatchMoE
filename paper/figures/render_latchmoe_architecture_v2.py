#!/usr/bin/env python3
"""Render the LatchMoE solution overview as a clean systems-paper diagram.

The figure deliberately uses two planes (control/data) and a compact timing
inset instead of a collection of rounded pointer boxes.  SVG is the source of
truth; CairoSVG and Ghostscript produce the PDF/PNG companions used by LaTeX.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SVG = HERE / "latchmoe_architecture.svg"
PDF = HERE / "latchmoe_architecture.pdf"
PNG = HERE / "latchmoe_architecture.png"
W, H = 1200, 650

BG = "#FFFFFF"
INK = "#252525"
MUTED = "#5F6872"
RULE = "#AAB4BE"
PANEL = "#F8FAFC"
BLUE = "#2166AC"
BLUE_LIGHT = "#E8F1FA"
ORANGE = "#D88900"
ORANGE_LIGHT = "#FFF1D7"
GREEN = "#198F72"
GREEN_LIGHT = "#E4F3EE"
HOST = "#F0F2F4"
RED = "#B5412B"


def esc(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none",
         sw: float = 1, rx: float = 0, dash: str | None = None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}"{d}/>')


def line(x1: float, y1: float, x2: float, y2: float, color: str = INK,
         sw: float = 2, dash: str | None = None, marker: str | None = None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d}{m}/>')


def text(x: float, y: float, value: str, size: float = 22, color: str = INK,
         anchor: str = "start", weight: str = "normal", family: str = "Linux Libertine O, Libertine, Times New Roman, serif",
         italic: bool = False) -> str:
    style = "font-style:italic;" if italic else ""
    return (f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" '
            f'font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" style="{style}">{esc(value)}</text>')


def multiline(x: float, y: float, lines: list[str], size: float = 22, color: str = INK,
              anchor: str = "start", weight: str = "normal", gap: float = 25) -> str:
    out = []
    for i, value in enumerate(lines):
        out.append(text(x, y + i * gap, value, size, color, anchor, weight))
    return "".join(out)


def cell(x: float, y: float, w: float, label: str, fill: str, stroke: str,
         size: float = 21, weight: str = "bold") -> str:
    return rect(x, y, w, 34, fill, stroke, 1.5, 2) + text(x + w / 2, y + 24, label, size, INK, "middle", weight)


def render() -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="Linux Libertine O, Libertine, Times New Roman, serif">'
    )
    parts.append(rect(0, 0, W, H, BG))
    parts.append("""<defs>
      <marker id="arrow-blue" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#2166AC"/></marker>
      <marker id="arrow-orange" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#D88900"/></marker>
      <marker id="arrow-green" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#198F72"/></marker>
      <marker id="arrow-gray" markerWidth="11" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#5F6872"/></marker>
    </defs>""")

    # Panel (a): only the stable interface is highlighted; the shapes are
    # intentionally square and quiet, like a systems-paper overview.
    parts.append(rect(24, 36, 1152, 174, PANEL, RULE, 1.2, 3))
    parts.append(text(44, 63, "(a) Capture-time setup", 25, MUTED, "start", "bold"))
    parts.append(line(44, 76, 1156, 76, RULE, 1))
    parts.append(text(82, 103, "model ABI + capability", 21, MUTED, "start", "bold"))
    parts.append(text(82, 132, "router · shared lane · output contract", 20, INK))
    parts.append(line(82, 151, 262, 151, RULE, 1.2))
    parts.append(text(82, 179, "fail closed", 19, RED, "start", "bold"))
    parts.append(line(315, 141, 374, 141, MUTED, 2, marker="arrow-gray"))
    parts.append(text(389, 103, "replay-visible allocations", 21, MUTED, "start", "bold"))
    parts.append(text(389, 132, "fixed slot bank + logical→slot map", 20, INK))
    parts.append(cell(390, 151, 66, "S0", GREEN_LIGHT, GREEN, 19))
    parts.append(cell(460, 151, 66, "S1", GREEN_LIGHT, GREEN, 19))
    parts.append(cell(530, 151, 66, "S2", GREEN_LIGHT, GREEN, 19))
    parts.append(cell(600, 151, 66, "S3", GREEN_LIGHT, GREEN, 19))
    parts.append(line(716, 141, 774, 141, MUTED, 2, marker="arrow-gray"))
    parts.append(text(791, 103, "captured graph pieces", 21, MUTED, "start", "bold"))
    parts.append(text(791, 132, "router  ·  routed MLP  ·  native combine", 20, INK))
    parts.append(text(791, 179, "same addresses at every replay", 19, GREEN, "start", "bold"))

    # Panel (b): the central story. Two planes and one explicit staging seam.
    parts.append(rect(24, 226, 1152, 244, BG, RULE, 1.2, 3))
    parts.append(text(44, 253, "(b) One invocation: dynamic ownership, fixed graph-visible addresses", 25, MUTED, "start", "bold"))
    parts.append(line(44, 266, 1156, 266, RULE, 1))
    parts.append(rect(52, 282, 414, 160, HOST, "#D1D7DD", 1, 2))
    parts.append(rect(500, 282, 650, 160, BLUE_LIGHT, "#A8C6E5", 1, 2))
    parts.append(text(70, 307, "CPU control plane", 21, MUTED, "start", "bold"))
    parts.append(text(518, 307, "NPU data plane (captured after setup)", 21, BLUE, "start", "bold"))
    parts.append(text(70, 335, "router output", 20, INK, "start", "bold"))
    parts.append(text(70, 362, "IDs", 19, MUTED, "start", "bold"))
    parts.append(cell(114, 339, 52, "e₂", BG, BLUE, 19))
    parts.append(cell(170, 339, 52, "e₅", BG, BLUE, 19))
    parts.append(cell(226, 339, 52, "e₈", BG, BLUE, 19))
    parts.append(text(70, 404, "weights", 19, MUTED, "start", "bold"))
    parts.append(cell(140, 382, 52, ".4", BG, BLUE, 18, "normal"))
    parts.append(cell(196, 382, 52, ".3", BG, BLUE, 18, "normal"))
    parts.append(cell(252, 382, 52, ".3", BG, BLUE, 18, "normal"))
    parts.append(line(300, 360, 340, 360, BLUE, 2, marker="arrow-blue"))
    parts.append(text(345, 335, "eager placement plan", 20, ORANGE, "start", "bold"))
    parts.append(text(345, 362, "hits · misses · wave order", 19, INK))
    parts.append(text(345, 405, "CPU-first expert pool", 19, MUTED, "start", "bold"))
    parts.append(line(345, 414, 455, 414, ORANGE, 2, marker="arrow-orange"))
    parts.append(text(466, 333, "STAGING", 18, ORANGE, "middle", "bold"))
    parts.append(line(477, 316, 477, 425, ORANGE, 2, "7 5"))
    parts.append(line(455, 414, 520, 414, ORANGE, 2, marker="arrow-orange"))
    parts.append(text(520, 335, "logical ID → slot map", 20, BLUE, "start", "bold"))
    parts.append(cell(530, 348, 125, "e₂ → S1", BG, GREEN, 19))
    parts.append(cell(660, 348, 125, "e₅ → S3", BG, GREEN, 19))
    parts.append(cell(790, 348, 125, "e₈ → S0", BG, GREEN, 19))
    parts.append(text(520, 405, "fixed slot bank", 19, GREEN, "start", "bold"))
    parts.append(cell(645, 386, 66, "S0", GREEN_LIGHT, GREEN, 19))
    parts.append(cell(715, 386, 66, "S1", GREEN_LIGHT, GREEN, 19))
    parts.append(cell(785, 386, 66, "S2", GREEN_LIGHT, GREEN, 19))
    parts.append(cell(855, 386, 66, "S3", GREEN_LIGHT, GREEN, 19))
    parts.append(line(930, 403, 969, 403, GREEN, 2, marker="arrow-green"))
    parts.append(text(978, 370, "captured", 19, BLUE, "start", "bold"))
    parts.append(text(978, 394, "routed MLP", 20, BLUE, "start", "bold"))
    parts.append(text(978, 420, "+ native combine", 19, INK))
    parts.append(text(70, 438, "shared lane: resident weights", 19, MUTED, "start", "italic"))
    parts.append(line(270, 434, 965, 434, MUTED, 1.5, "5 5", marker="arrow-gray"))
    parts.append(text(940, 456, "one combine", 18, MUTED, "end", "italic"))
    parts.append(text(50, 462, "dynamic logical owners", 18, ORANGE, "start", "bold"))
    parts.append(text(1118, 462, "stable physical addresses", 18, GREEN, "end", "bold"))

    # Panel (c): timing rather than another module map.  The two event arrows
    # are the actual safety argument in the paper.
    parts.append(rect(24, 486, 1152, 140, PANEL, RULE, 1.2, 3))
    parts.append(text(44, 513, "(c) Versioned lease: overlap without premature visibility or overwrite", 25, MUTED, "start", "bold"))
    parts.append(line(44, 526, 1156, 526, RULE, 1))
    parts.append(text(64, 557, "transfer stream", 20, ORANGE, "start", "bold"))
    parts.append(text(64, 602, "consumer stream", 20, BLUE, "start", "bold"))
    # Transfer events.
    parts.append(cell(240, 540, 170, "assign (owner,v)", ORANGE_LIGHT, ORANGE, 19))
    parts.append(line(410, 557, 450, 557, ORANGE, 2, marker="arrow-orange"))
    parts.append(cell(450, 540, 170, "H2D wave k+1", ORANGE_LIGHT, ORANGE, 19))
    parts.append(line(620, 557, 660, 557, ORANGE, 2, marker="arrow-orange"))
    parts.append(cell(660, 540, 120, "ready", GREEN_LIGHT, GREEN, 19))
    parts.append(line(780, 557, 820, 557, GREEN, 2, marker="arrow-green"))
    parts.append(text(824, 550, "publish map", 19, GREEN, "start", "bold"))
    # Consumer events.
    parts.append(cell(240, 585, 170, "replay wave k", BLUE_LIGHT, BLUE, 19))
    parts.append(line(410, 602, 450, 602, BLUE, 2, marker="arrow-blue"))
    parts.append(cell(450, 585, 170, "compute", BLUE_LIGHT, BLUE, 19))
    parts.append(line(620, 602, 660, 602, BLUE, 2, marker="arrow-blue"))
    parts.append(cell(660, 585, 120, "done", GREEN_LIGHT, GREEN, 19))
    parts.append(line(780, 602, 820, 602, GREEN, 2, marker="arrow-green"))
    parts.append(text(824, 595, "reuse bank", 19, GREEN, "start", "bold"))
    # Cross-stream happens-before edges.
    parts.append(line(720, 578, 720, 570, GREEN, 2, "5 5", marker="arrow-green"))
    parts.append(text(720, 576, "ready→publish", 16, GREEN, "middle", "italic"))
    parts.append(line(720, 575, 720, 585, GREEN, 2, "5 5", marker="arrow-green"))
    parts.append(line(715, 577, 715, 585, GREEN, 2, "5 5", marker="arrow-green"))
    parts.append(text(1040, 573, "lease = (slot, owner, version)", 18, MUTED, "middle", "italic"))
    parts.append(text(1040, 598, "publish only after ready; reuse only after done", 17, MUTED, "middle"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    SVG.write_text(render(), encoding="utf-8")
    cairosvg = shutil.which("cairosvg")
    if not cairosvg:
        raise SystemExit("cairosvg executable is required")
    if not shutil.which("gs"):
        raise SystemExit("ghostscript is required")
    raw = HERE / ".latchmoe_architecture_v2.raw.pdf"
    subprocess.run([cairosvg, str(SVG), "-o", str(raw)], check=True)
    subprocess.run([cairosvg, str(SVG), "-o", str(PNG), "-s", "1.5"], check=True)
    subprocess.run([
        "gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5", f"-sOutputFile={PDF}", str(raw)
    ], check=True)
    raw.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
