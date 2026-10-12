#!/usr/bin/env python3
"""Render the checked-in CSV as a dependency-free SVG quality/cost chart."""
import argparse
import csv
import html
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",default="reports/evidence/results.csv")
    parser.add_argument("--out",default="reports/evidence/quality_cost.svg")
    args=parser.parse_args()
    rows=[]
    for row in csv.DictReader(Path(args.results).open()):
        if row["run_id"] in {"sb_seg32_24_colmap_s42_3k_v1","sb_seg32_24_colmap_s42_7k_v1",
                              "sb_seg32_24_vggt_s42_7k_v1","sb_seg32_24_vggt_ba_s42_7k_v3"}:
            rows.append(row)
    if len(rows)!=4:raise SystemExit("expected four formal quality/cost rows")
    width,height=920,560;left,right,top,bottom=100,860,65,465
    x_max=450.;y_min,y_max=10.,20.
    sx=lambda value:left+float(value)/x_max*(right-left)
    sy=lambda value:bottom-(float(value)-y_min)/(y_max-y_min)*(bottom-top)
    colors={"colmap":"#0b77d5","vggt":"#df6c20","vggt+ba":"#7c3aed"}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#f7f9fc"/>',
           '<style>text{font-family:system-ui,sans-serif;fill:#172033}.grid{stroke:#d8dee9;stroke-width:1}.axis{stroke:#334155;stroke-width:2}.pt{stroke:white;stroke-width:3}</style>',
           '<text x="100" y="36" font-size="24" font-weight="700">Held-out quality vs optimization cost</text>']
    for x in range(0,451,75):
        px=sx(x);parts += [f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}"/>',
                         f'<text x="{px:.1f}" y="492" text-anchor="middle" font-size="13">{x}</text>']
    for y in range(10,21,2):
        py=sy(y);parts += [f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}"/>',
                         f'<text x="82" y="{py+5:.1f}" text-anchor="end" font-size="13">{y}</text>']
    parts += [f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
              f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
              '<text x="480" y="530" text-anchor="middle" font-size="15">Optimization seconds (lower is better)</text>',
              '<text transform="translate(25 270) rotate(-90)" text-anchor="middle" font-size="15">PSNR dB (higher is better)</text>']
    for row in rows:
        x=sx(row["optimization_s"]);y=sy(row["psnr"]);model_mb=float(row["model_bytes"])/1e6
        radius=10+model_mb/20;kind=row["initializer"] if row["initializer"] in colors else "colmap"
        label=("COLMAP 3k" if row["steps"]=="3000" else "COLMAP 7k" if kind=="colmap" else
               "VGGT + BA 7k" if kind=="vggt+ba" else "VGGT 7k")
        label_y=y+35 if kind=="vggt+ba" else y-4
        parts += [f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{colors[kind]}" opacity="0.86"/>',
                  f'<text x="{x+radius+8:.1f}" y="{label_y:.1f}" font-size="14" font-weight="700">{html.escape(label)}</text>',
                  f'<text x="{x+radius+8:.1f}" y="{label_y+19:.1f}" font-size="12">LPIPS {float(row["lpips"]):.3f} · {model_mb:.1f} MB</text>']
    parts += ['<text x="100" y="552" font-size="12" fill="#475569">Circle radius encodes PLY size. Same scene, 24 train / 8 held-out, 1024px max edge, seed 42, A800.</text>',
              '</svg>']
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text("\n".join(parts)+"\n")
    print(out)


if __name__=="__main__":main()
