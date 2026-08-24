#!/usr/bin/env python3
"""Render compact icon-assisted LatchMoE architecture artwork."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SVG, PDF, PNG = (HERE / n for n in ("latchmoe_architecture.svg", "latchmoe_architecture.pdf", "latchmoe_architecture.png"))
W, H = 1400, 760
INK="#20262D"; MUTED="#5C6670"; RULE="#7B8792"; LIGHT="#BEC7CF"
HOST="#F7F8F9"; STAGE="#FFF6E7"; CAP="#F1F7FC"; SLOT="#E5F3EF"
BLUE="#165A9B"; TEAL="#158A7C"; ORANGE="#C97700"; SYNC="#B56C12"

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def rect(x,y,w,h,fill="none",stroke="none",sw=1,rx=0,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"{d}/>'
def line(x1,y1,x2,y2,c=INK,sw=2,dash=None,marker=None):
    d=f' stroke-dasharray="{dash}"' if dash else ""; m=f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}"{d}{m}/>'
def path(d,c=INK,sw=2,dash=None,marker=None):
    da=f' stroke-dasharray="{dash}"' if dash else ""; ma=f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{sw}"{da}{ma}/>'
def text(x,y,s,size=20,c=INK,anchor="start",weight="normal",italic=False):
    # Keep the smallest glyphs readable after the figure is scaled to the
    # two-column page width; short icon labels are intentionally concise.
    size = max(size, 20)
    st=' font-style="italic"' if italic else ""
    return f'<text x="{x}" y="{y}" fill="{c}" font-family="Arial, Helvetica, sans-serif" font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}"{st}>{esc(s)}</text>'
def rows(x,y,ss,size=18,c=MUTED,gap=22): return "".join(text(x,y+i*gap,s,size,c) for i,s in enumerate(ss))
def cell(x,y,w,h,s,fill,stroke,size=17): return rect(x,y,w,h,fill,stroke,1.5,1)+text(x+w/2,y+h*.66,s,size,INK,"middle","bold")

def chip(x,y,c=INK):
    p=[rect(x+9,y+8,34,32,"#fff",c,2,2)]
    for i in range(4): p += [line(x+i*10+12,y,x+i*10+12,y+8,c,1.8),line(x+i*10+12,y+40,x+i*10+12,y+48,c,1.8)]
    p += [line(x,y+16,x+9,y+16,c,1.8),line(x+43,y+16,x+52,y+16,c,1.8),line(x,y+32,x+9,y+32,c,1.8),line(x+43,y+32,x+52,y+32,c,1.8)]
    return "".join(p)
def dram(x,y,c=RULE):
    p=[rect(x,y+20,62,30,"#fff",c,1.8,2),rect(x+8,y+10,62,30,"#fff",c,1.8,2),rect(x+16,y,62,30,"#fff",c,1.8,2)]
    return "".join(p+[line(xx,y+8,xx,y+22,c,1.4) for xx in (x+28,x+42,x+56)])
def graph(x,y,c=BLUE):
    return (line(x+5,y+32,x+24,y+12,c,2)+line(x+24,y+12,x+47,y+30,c,2)+line(x+24,y+12,x+43,y+5,c,2)+
            rect(x,y+28,10,10,"#fff",c,1.8,2)+rect(x+19,y+7,10,10,"#fff",c,1.8,2)+rect(x+42,y+25,10,10,"#fff",c,1.8,2)+rect(x+38,y,10,10,"#fff",c,1.8,2))
def router(x,y,c=INK):
    return line(x,y+18,x+27,y+18,c,3)+line(x+27,y+18,x+48,y+2,c,3)+line(x+27,y+18,x+48,y+18,c,3)+line(x+27,y+18,x+48,y+34,c,3)
def packets(x,y,c=TEAL): return rect(x+14,y+10,38,24,"#fff",c,1.7,2)+rect(x+7,y+5,38,24,"#fff",c,1.7,2)+rect(x,y,38,24,"#fff",c,1.7,2)
def wave(x,y,c=ORANGE): return rect(x,y+18,13,13,"#fff",c,1.5)+rect(x+18,y+11,13,20,"#fff",c,1.5)+rect(x+36,y+4,13,27,"#fff",c,1.5)+line(x-4,y+36,x+54,y+36,c,1.7,"4 3")
def lock(x,y,c=SYNC): return rect(x+8,y+18,29,25,"#fff",c,2,2)+path(f"M {x+14} {y+19} V {y+10} C {x+14} {y-2}, {x+31} {y-2}, {x+31} {y+10} V {y+19}",c,2)
def capture(x,y,c=BLUE): return line(x,y+10,x,y,c,2)+line(x,y,x+10,y,c,2)+line(x+42,y,x+52,y,c,2)+line(x+52,y,x+52,y+10,c,2)+line(x,y+31,x,y+41,c,2)+line(x,y+41,x+10,y+41,c,2)+line(x+42,y+41,x+52,y+41,c,2)+line(x+52,y+31,x+52,y+41,c,2)

def render():
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    p.append('''<defs><marker id="data" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#158A7C"/></marker><marker id="control" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#C97700"/></marker><marker id="sync" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#B56C12"/></marker><marker id="stable" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0,0 L11,4.5 L0,9 z" fill="#165A9B"/></marker></defs>''')
    p.append(rect(0,0,W,H,"#fff")); p.append(text(W/2,31,"LatchMoE system architecture: address-stable, graph-compatible MoE offloading",27,INK,"middle","bold"))
    # host
    p += [rect(28,53,1344,210,HOST,RULE,2.5,9),text(49,83,"HOST / CPU + DRAM",25,MUTED,"start","bold"),line(49,96,1350,96,LIGHT,1.2)]
    p += [rect(60,113,285,122,"#fff",RULE,1.5,3),chip(83,143),text(151,143,"CONTROL",21,INK,"start","bold"),rows(151,169,["route + ABI","wave plan"],18),text(151,222,"dynamic owners",17,ORANGE,"start","bold")]
    p += [rect(388,106,512,148,"#fff",RULE,1.5,3),text(412,133,"PINNED EXPERTS",21,INK,"start","bold"),dram(797,116)]
    for x,s in ((412,"E0"),(490,"E1"),(568,"E2"),(696,"En-1"),(774,"En")): p.append(cell(x,163,70,38,s,HOST,RULE,17))
    p += [text(656,188,"…",23,MUTED,"middle"),text(412,227,"CPU source",17,TEAL,"start","bold")]
    p += [rect(943,113,391,122,"#fff",RULE,1.5,3),graph(967,140),text(1033,143,"CAPTURE CONTRACT",20,INK,"start","bold"),rows(1033,171,["fixed map · stable ABI","captured graph"],18),capture(1253,151,BLUE)]
    # npu and cross-boundary flow
    p += [rect(28,282,1344,410,"#fff",BLUE,2.8,9),text(49,313,"ASCEND NPU / HBM",25,BLUE,"start","bold"),path("M 202 235 L 202 275 L 400 275 L 400 342 L 230 342 L 230 382",ORANGE,2.4,"8 6","control"),text(415,300,"control",16,ORANGE,"start","bold"),path("M 640 254 L 640 294 L 910 294 L 910 350 L 515 350",TEAL,5.2),rect(716,274,118,25,"#fff"),text(775,293,"H2D waves",16,TEAL,"middle","bold")]
    # staging
    p += [rect(62,351,338,220,STAGE,ORANGE,2,4),text(84,379,"STAGING SEAM",20,ORANGE,"start","bold"),router(91,412),packets(192,409),wave(298,406),text(112,471,"route",16,INK,"middle","bold"),text(211,471,"stage",16,INK,"middle","bold"),text(322,471,"wave",16,INK,"middle","bold"),line(115,488,351,488,LIGHT,1.1),text(84,518,"uncaptured",17,ORANGE,"start","bold"),text(84,546,"cache · capacity · H2D",17,MUTED),path("M 400 488 L 480 488 L 480 450 L 515 450",SYNC,2.2,"7 5","sync")]
    # captured graph
    p += [rect(444,334,872,253,CAP,BLUE,2,5,"10 7"),text(466,361,"CAPTURED GRAPH",20,BLUE,"start","bold"),capture(625,338,BLUE),text(694,361,"stable addresses",17,MUTED)]
    p += [rect(500,403,391,156,SLOT,TEAL,2.2,4),text(524,431,"SLOT BANK",21,INK,"start","bold"),text(524,455,"fixed HBM slots",17,MUTED)]
    for x,s in ((524,"S0"),(599,"S1"),(674,"S2"),(797,"Sk-1")): p.append(cell(x,475,69,39,s,"#fff",TEAL,17))
    p += [text(772,501,"…",23,MUTED,"middle"),text(524,543,"ID → slot",17,TEAL,"start","bold"),text(629,543,"e2→S1 · e5→S3 · e8→S0",17,INK),path("M 640 254 L 640 294 L 910 294 L 910 392 L 515 392 L 515 428",TEAL,5.2)]
    p += [rect(962,403,326,156,"#fff",BLUE,2.2,4),text(987,431,"CAPTURED MoE",21,BLUE,"start","bold"),text(987,455,"routed MLP",17,MUTED),text(1125,511,"Σ ×",34,INK,"middle","bold"),text(1125,539,"native combine",17,INK,"middle","bold"),path("M 891 482 L 952 482",BLUE,5,None,"stable"),text(922,462,"stable",15,BLUE,"middle","bold"),path("M 500 575 L 1180 575",MUTED,1.7,"6 6","stable"),text(522,568,"resident shared lane",16,MUTED,"start","normal",True),text(1188,568,"one combine",16,MUTED,"start","normal",True)]
    # lease
    p += [line(62,608,1317,608,LIGHT,1.2),lock(81,628),text(137,644,"LEASE",20,INK,"start","bold"),text(302,633,"transfer",16,ORANGE,"middle","bold"),text(302,674,"consumer",16,BLUE,"middle","bold")]
    for x,s in ((360,"assign"),(518,"H2D"),(676,"ready")): p.append(cell(x,619,126 if s!="ready" else 82,27,s,STAGE if s!="ready" else SLOT,ORANGE if s!="ready" else TEAL,15))
    p += [line(486,632,510,632,ORANGE,2,marker="control"),line(644,632,668,632,ORANGE,2,marker="control"),text(782,638,"publish",16,TEAL,"start","bold")]
    for x,s in ((360,"replay"),(518,"compute"),(676,"done")): p.append(cell(x,660,126 if s!="done" else 82,27,s,CAP if s!="done" else SLOT,BLUE if s!="done" else TEAL,15))
    p += [line(486,673,510,673,BLUE,2,marker="stable"),line(644,673,668,673,BLUE,2,marker="stable"),text(782,679,"reuse",16,TEAL,"start","bold"),path("M 717 647 L 717 660",SYNC,1.7,"5 4","sync"),text(1075,638,"ready → publish",16,MUTED,"middle"),text(1075,678,"done → reuse",16,MUTED,"middle")]
    # legend
    p += [line(48,729,84,729,TEAL,4,marker="data"),text(94,735,"data",16,MUTED),line(190,729,226,729,ORANGE,2.2,"8 5","control"),text(236,735,"control",16,MUTED),line(370,729,406,729,SYNC,2.2,"7 5","sync"),text(416,735,"sync",16,MUTED),line(524,729,560,729,BLUE,4,marker="stable"),text(570,735,"stable",16,MUTED),rect(700,718,23,18,HOST,RULE,1.2),text(733,735,"host",16,MUTED),rect(805,718,23,18,SLOT,TEAL,1.2),text(838,735,"slot bank",16,MUTED),"</svg>"]
    return "\n".join(p)+"\n"

def main():
    SVG.write_text(render(),encoding="utf-8")
    cairosvg,gs=shutil.which("cairosvg"),shutil.which("gs")
    if not cairosvg or not gs: raise SystemExit("cairosvg and ghostscript are required")
    raw=HERE/".latchmoe_architecture.raw.pdf"
    subprocess.run([cairosvg,str(SVG),"-o",str(raw)],check=True)
    subprocess.run([cairosvg,str(SVG),"-o",str(PNG),"-s","1.5"],check=True)
    subprocess.run([gs,"-q","-dNOPAUSE","-dBATCH","-sDEVICE=pdfwrite","-dCompatibilityLevel=1.5",f"-sOutputFile={PDF}",str(raw)],check=True)
    raw.unlink(missing_ok=True)
if __name__ == "__main__": main()
