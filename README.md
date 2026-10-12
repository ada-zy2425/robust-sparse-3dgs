# Robust Sparse 3DGS

**Sparse-view reconstruction · numerical stability · controlled geometry experiments**

A COLMAP / VGGT → gsplat workflow for training, held-out rendering, PLY export and local WebGL inspection.

[中文报告](docs/project_report_zh.md) · [Reproduce](docs/reproduction.md) · [Evidence](reports/README.md)

![System overview and initialization repair — schematic, not a reconstruction image](docs/figures/initialization.svg)

## Measured results

![Matched BA: gains across two scenes and two training seeds](docs/figures/ba_gain.svg)

**+1.591–2.166 dB PSNR**, with SSIM and LPIPS improving in all four pairs. Shared initialization snapshots and numerical guard; this measures joint camera/point BA, with training randomness repeated.

![Pose sensitivity with fixed points and held-out cameras](docs/figures/pose_sensitivity.svg)

**1° rotation: −0.910 / −3.198 dB** on bonsai / room. Fixed points, intrinsics, RGB and alignment; two development scenes, one training seed and one noise draw.

## Where to look

| Path | Purpose |
|---|---|
| [`docs/`](docs/project_report_zh.md) | Three documents: report, reproduction, technical notes |
| [`reports/`](reports/README.md) | Current results + one historical archive |
| `bgs/` | Runtime implementation |
| [`scripts/`](docs/reproduction.md#tool-map) | Experiment / evaluation / figure tools; setup and diagnostics |
| `configs/`, `protocols/`, `tests/` | Inputs, frozen contracts and validation |

```bash
python -m unittest discover -s tests -v
python scripts/report.py --verify
```

[Methods, provenance and limits](docs/technical_notes.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
