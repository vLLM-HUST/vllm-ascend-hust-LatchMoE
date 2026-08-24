#!/usr/bin/env python3
"""Render the LatchMoE architecture as a clean, vector systems figure.

The drawing follows the visual grammar used by systems papers: large physical
regions (host and NPU), a captured-region boundary, a small number of named
components, and a legend that distinguishes data/control/synchronization and
stable-compute paths.  SVG is the editable source of truth.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SVG = HERE / "latchmoe_architecture.svg"
PDF = HERE / "latchmoe_architecture.pdf"
PNG = HERE / "latchmoe_architecture.png"
W, H = 1400, 800

INK = "#20262D"
MUTED = "#5C6670"
RULE = "#7B8792"
LIGHT_RULE = "#B8C1C9"
HOST_FILL = "#F7F8F9"
STAGE_FILL = "#FFF7EA"
CAP_FILL = "#F1F7FC"
SLOT_FILL = "#E6F3EF"
BLUE = "#165A9B"
TEAL = "#158A7C"
ORANGE = "#C97700"
SYNC = "#B56C12"
RED = "#AF3E2E"


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}"{d}/>')


def line(x1, y1, x2, y2, color=INK, sw=2.0, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d}{m}/>')


def path(d, color=INK, sw=2.0, dash=None, marker=None, fill="none"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    ma = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{da}{ma}/>'


def text(x, y, value, size=20, color=INK, anchor="start", weight="normal", italic=False):
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" fill="{color}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}px" '
            f'font-weight="{weight}" text-anchor="{anchor}"{style}>{esc(value)}</text>')


def multiline(x, y, lines, size=20, color=INK, anchor="start", weight="normal", gap=24):
    return "".join(text(x, y + i * gap, value, size, color, anchor, weight) for i, value in enumerate(lines))


def cell(x, y, w, h, label, fill, stroke, size=18, weight="bold"):
    return rect(x, y, w, h, fill, stroke, 1.5, 1) + text(x + w / 2, y + h * .66, label, size, INK, "middle", weight)


def router_icon(x, y):
    out = [line(x, y + 12, x + 32, y + 12, INK, 3), line(x + 32, y + 12, x + 51, y - 4, INK, 3),
           line(x + 32, y + 12, x + 51, y + 12, INK, 3), line(x + 32, y + 12, x + 51, y + 28, INK, 3)]
    out += [line(x + 47, y - 7, x + 54, y - 4, INK, 3), line(x + 47, y + 9, x + 54, y + 12, INK, 3),
            line(x + 47, y + 25, x + 54, y + 28, INK, 3)]
    return "".join(out)


def render():
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    p.append("""<defs>
      <marker id="data" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#158A7C"/></marker>
      <marker id="control" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#C97700"/></marker>
      <marker id="sync" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#B56C12"/></marker>
      <marker id="stable" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#165A9B"/></marker>
    </defs>""")
    p.append(rect(0, 0, W, H, "#FFFFFF"))
    p.append(text(W / 2, 31, "LatchMoE system architecture: address-stable, graph-compatible MoE offloading", 27, INK, "middle", "bold"))

    # Physical host region.
    p.append(rect(28, 53, 1344, 222, HOST_FILL, RULE, 2.5, 9))
    p.append(text(49, 83, "HOST  (CPU / DRAM REGION)", 25, MUTED, "start", "bold"))
    p.append(line(49, 96, 1350, 96, LIGHT_RULE, 1.2))

    # Host control plane.
    p.append(rect(60, 116, 286, 128, "#FFFFFF", RULE, 1.5, 3))
    p.append(text(82, 141, "CONTROL PLANE", 21, INK, "start", "bold"))
    p.append(multiline(82, 168, ["scheduler + ABI preflight", "router output / wave plan", "fail closed on unsupported tuple"], 17, MUTED, "start", "normal", 23))
    p.append(text(82, 231, "dynamic ownership", 17, ORANGE, "start", "bold"))

    # Pinned expert store.
    p.append(rect(390, 108, 520, 151, "#FFFFFF", RULE, 1.5, 3))
    p.append(text(414, 135, "PINNED HOST EXPERT STORE", 21, INK, "start", "bold"))
    p.append(text(414, 158, "full experts in CPU DRAM", 17, MUTED))
    p.append(cell(414, 182, 76, 40, "E0", HOST_FILL, RULE, 18))
    p.append(cell(492, 182, 76, 40, "E1", HOST_FILL, RULE, 18))
    p.append(cell(570, 182, 76, 40, "E2", HOST_FILL, RULE, 18))
    p.append(text(672, 209, "…", 24, MUTED, "middle"))
    p.append(cell(697, 182, 76, 40, "En-1", HOST_FILL, RULE, 18))
    p.append(cell(775, 182, 76, 40, "En", HOST_FILL, RULE, 18))
    p.append(text(414, 247, "source of asynchronous H2D waves", 16, TEAL, "start", "bold"))

    # Capture contract / replay-visible allocation.
    p.append(rect(956, 116, 378, 128, "#FFFFFF", RULE, 1.5, 3))
    p.append(text(980, 141, "CAPTURE CONTRACT", 21, INK, "start", "bold"))
    p.append(multiline(980, 169, ["fixed slot-bank allocation", "logical ID → physical slot map", "same addresses at every replay"], 17, MUTED, "start", "normal", 23))
    p.append(text(980, 231, "replay-visible data plane", 17, TEAL, "start", "bold"))

    # NPU region and captured boundary.
    p.append(rect(28, 293, 1344, 414, "#FFFFFF", BLUE, 2.8, 9))
    p.append(text(49, 323, "ASCEND NPU / HBM REGION", 24, BLUE, "start", "bold"))

    # Cross-region paths are drawn after the NPU background so that the
    # transfer and control arrows visibly cross the host/NPU boundary.
    p.append(path("M 203 244 L 203 300 L 400 300 L 400 350 L 235 350 L 235 388", ORANGE, 2.4, "8 6"))
    p.append(text(430, 314, "control", 16, ORANGE, "start", "bold"))
    p.append(path("M 650 259 L 650 310 L 900 310 L 900 405 L 515 405 L 515 428", TEAL, 5.5))
    p.append(rect(704, 289, 125, 25, "#FFFFFF", "none", 0))
    p.append(text(766, 308, "multi-packet H2D", 16, TEAL, "middle", "bold"))
    p.append(rect(447, 345, 869, 253, CAP_FILL, BLUE, 2.0, 5, "10 7"))
    p.append(text(468, 371, "CAPTURED ACLGRAPH REGION BOUNDARY", 19, BLUE, "start", "bold"))
    p.append(text(468, 391, "graph-visible addresses remain unchanged", 16, MUTED))
    # Redraw the H2D path on top of the captured-region fill so its endpoint
    # visibly reaches the slot bank rather than disappearing at the boundary.
    p.append(path("M 650 259 L 650 310 L 900 310 L 900 405 L 515 405 L 515 428", TEAL, 5.5))

    # Uncaptured staging region.
    p.append(rect(62, 363, 338, 236, STAGE_FILL, ORANGE, 2.0, 4))
    p.append(text(84, 389, "UNCAPTURED STAGING", 18, ORANGE, "start", "bold"))
    p.append(text(84, 411, "CONTROL REGION", 18, ORANGE, "start", "bold"))
    # Keep the control arrow visible through the staging-region background.
    p.append(path("M 203 244 L 203 300 L 400 300 L 400 350 L 235 350 L 235 388", ORANGE, 2.4, "8 6", "control"))
    p.append(router_icon(96, 431))
    p.append(text(88, 485, "ROUTER", 17, INK, "start", "bold"))
    p.append(text(88, 508, "top-k expert IDs", 16, MUTED))
    p.append(text(88, 531, "e2 · e5 · e8", 17, INK, "start", "bold"))
    p.append(line(169, 469, 212, 469, MUTED, 2, marker="stable"))
    p.append(rect(212, 408, 165, 153, "#FFFFFF", ORANGE, 1.8, 3))
    p.append(text(229, 432, "REPLAY-BOUNDARY", 15, ORANGE, "start", "bold"))
    p.append(text(229, 453, "STAGING", 18, ORANGE, "start", "bold"))
    p.append(line(228, 463, 359, 463, LIGHT_RULE, 1))
    p.append(multiline(229, 486, ["slot-cache check", "capacity-bounded", "wave scheduler", "H2D transfer stream"], 16, MUTED, "start", "normal", 20))
    p.append(path("M 377 501 L 430 501 L 430 480 L 500 480", SYNC, 2.2, "7 5", "sync"))
    p.append(text(405, 470, "publish", 15, SYNC, "middle", "bold"))

    # Slot-stable bank.
    p.append(rect(500, 408, 385, 157, SLOT_FILL, TEAL, 2.2, 4))
    p.append(path("M 515 401 L 515 428", TEAL, 5.0))
    p.append(text(523, 435, "SLOT-STABLE NPU SLOT BANK", 20, INK, "start", "bold"))
    p.append(text(523, 458, "fixed tensor addresses in HBM", 16, MUTED))
    p.append(cell(523, 480, 70, 41, "S0", "#FFFFFF", TEAL, 18))
    p.append(cell(598, 480, 70, 41, "S1", "#FFFFFF", TEAL, 18))
    p.append(cell(673, 480, 70, 41, "S2", "#FFFFFF", TEAL, 18))
    p.append(text(771, 507, "…", 24, MUTED, "middle"))
    p.append(cell(795, 480, 70, 41, "Sk-1", "#FFFFFF", TEAL, 18))
    p.append(text(523, 548, "logical ID → slot map", 16, TEAL, "start", "bold"))
    p.append(text(710, 548, "e2→S1   e5→S3   e8→S0", 16, INK, "start"))

    # Captured compute region.
    p.append(rect(964, 408, 325, 157, "#FFFFFF", BLUE, 2.2, 4))
    p.append(text(990, 435, "CAPTURED FUSED MoE KERNEL", 19, BLUE, "start", "bold"))
    p.append(text(990, 461, "routed MLP / grouped matmul", 16, MUTED))
    p.append(text(1127, 512, "Σ ×", 33, INK, "middle", "bold"))
    p.append(text(1127, 539, "native combine", 17, INK, "middle", "bold"))
    p.append(path("M 885 495 L 955 495", BLUE, 5.0, None, "stable"))
    p.append(text(920, 476, "stable compute flow", 15, BLUE, "middle", "bold"))

    # Resident shared lane and single combine.
    p.append(path("M 500 579 L 1170 579", MUTED, 1.7, "6 6", "stable"))
    p.append(text(522, 592, "resident shared lane", 16, MUTED, "start", "italic"))
    p.append(text(1180, 592, "one combine", 16, MUTED, "start", "italic"))

    # Compact lease strip: the safety rule is shown as a timeline, not as a
    # third collection of module boxes.
    p.append(line(62, 622, 1317, 622, LIGHT_RULE, 1.2))
    p.append(text(83, 650, "VERSIONED LEASE", 18, INK, "start", "bold"))
    p.append(text(83, 673, "(slot, owner, version)", 16, MUTED, "start", "italic"))
    p.append(text(300, 643, "transfer", 15, ORANGE, "middle", "bold"))
    p.append(text(300, 683, "consumer", 15, BLUE, "middle", "bold"))
    # Transfer track.
    p.append(cell(360, 629, 148, 28, "assign", STAGE_FILL, ORANGE, 15, "bold"))
    p.append(line(508, 643, 532, 643, ORANGE, 2, marker="control"))
    p.append(cell(532, 629, 148, 28, "H2D wave k+1", STAGE_FILL, ORANGE, 15, "bold"))
    p.append(line(680, 643, 704, 643, ORANGE, 2, marker="control"))
    p.append(cell(704, 629, 82, 28, "ready", SLOT_FILL, TEAL, 15, "bold"))
    p.append(line(786, 643, 810, 643, TEAL, 2, marker="data"))
    p.append(text(825, 649, "publish map", 15, TEAL, "start", "bold"))
    # Consumer track.
    p.append(cell(360, 669, 148, 28, "replay wave k", CAP_FILL, BLUE, 15, "bold"))
    p.append(line(508, 683, 532, 683, BLUE, 2, marker="stable"))
    p.append(cell(532, 669, 148, 28, "compute", CAP_FILL, BLUE, 15, "bold"))
    p.append(line(680, 683, 704, 683, BLUE, 2, marker="stable"))
    p.append(cell(704, 669, 82, 28, "done", SLOT_FILL, TEAL, 15, "bold"))
    p.append(line(786, 683, 810, 683, TEAL, 2, marker="data"))
    p.append(text(825, 689, "reuse bank", 15, TEAL, "start", "bold"))
    p.append(path("M 745 658 L 745 669", SYNC, 1.8, "5 4", "sync"))
    p.append(path("M 790 699 L 790 707 L 704 707 L 704 697", SYNC, 1.6, "5 4", "sync"))
    p.append(text(1085, 651, "publish only after ready", 15, MUTED, "middle"))
    p.append(text(1085, 681, "reuse only after done", 15, MUTED, "middle"))

    # Figure legend.
    p.append(line(48, 746, 84, 746, TEAL, 4, marker="data")); p.append(text(94, 752, "data path", 16, MUTED))
    p.append(line(218, 746, 254, 746, ORANGE, 2.2, "8 5", "control")); p.append(text(264, 752, "control path", 16, MUTED))
    p.append(line(404, 746, 440, 746, SYNC, 2.2, "7 5", "sync")); p.append(text(450, 752, "synchronization", 16, MUTED))
    p.append(line(628, 746, 664, 746, BLUE, 4, marker="stable")); p.append(text(674, 752, "stable compute flow", 16, MUTED))
    p.append(rect(865, 733, 25, 20, HOST_FILL, RULE, 1.2)); p.append(text(900, 752, "pinned host memory", 16, MUTED))
    p.append(rect(1086, 733, 25, 20, SLOT_FILL, TEAL, 1.2)); p.append(text(1121, 752, "slot bank", 16, MUTED))
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main():
    SVG.write_text(render(), encoding="utf-8")
    cairosvg = shutil.which("cairosvg")
    gs = shutil.which("gs")
    if not cairosvg or not gs:
        raise SystemExit("cairosvg and ghostscript are required")
    raw = HERE / ".latchmoe_architecture.raw.pdf"
    subprocess.run([cairosvg, str(SVG), "-o", str(raw)], check=True)
    subprocess.run([cairosvg, str(SVG), "-o", str(PNG), "-s", "1.5"], check=True)
    subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", f"-sOutputFile={PDF}", str(raw)], check=True)
    raw.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
