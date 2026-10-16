# Robust Sparse 3DGS

**Sparse-view reconstruction · numerical stability · camera-error diagnosis and transfer**

A COLMAP / VGGT → gsplat workflow for training, held-out rendering, PLY export and local WebGL inspection.

[中文报告](docs/project_report_zh.md) · [Reproduce](docs/reproduction.md) · [Evidence](reports/README.md)

![System overview and initialization repair — schematic, not a reconstruction image](docs/figures/initialization.svg)

## Measured results

![Rotation-only camera transfer confirmation on Garden](docs/figures/rotation_transfer_v9.svg)

The frozen rotation-only transfer passed both preregistered Garden perturbations at the common 7k endpoint. Relative to fixed perturbed cameras, held-out PSNR improved **+2.613 / +2.347 dB** for perturbation seeds 2718 / 3141; injected-rotation recovery was **96.44% / 96.83%**. SSIM increased and LPIPS decreased for both seeds.

The contribution is deliberately narrow: established train-only standard BA supplies the step-0 rotation estimate; this project transfers its OpenCV c2w rotation while retaining each input camera's world-space center, then holds cameras fixed during GS. Garden is a historical development scene, the training seed is 42, and each arm uses 24 training views, 24 held-out views and 7,000 steps. This is not an absolute-pose or new-BA claim.

![Garden seed 3141 DSC08108 cached prediction and error comparison](docs/figures/garden_3141_dsc08108_counterexample.png)

The aggregate pass retains a real counterexample. PSNR improved on **24/24** held-out views for seed 2718 but **23/24** for seed 3141; SSIM and LPIPS improved on 24/24 in both. On `DSC08108.JPG`, seed 3141 C loses 3.257 dB PSNR while still improving SSIM and LPIPS. Existing 1k/3k/7k caches localize 92.2% of its final squared RGB error to the upper half and show a regional blue/brightness shift without broad clipping. They do not identify a causal Gaussian or densification event.

## Evidence chain

1. Controlled 1° camera rotation exposed a −0.910 / −3.198 dB held-out PSNR loss on Bonsai / Room.
2. Photometric pose repair did not meet its recovery contract (`v3 repair_accepted=false`); an early pose-only schedule was also negative.
3. Train-only 2D tracks plus standard BA recovered relative rotation, but transferring the full BA cameras failed Room's PSNR gate (`v7 FAIL`) because BA also moved centers under a pure-rotation intervention.
4. Keeping the input centers and transferring only BA rotations passed Bonsai / Room development (`v8 development PASS`) and both frozen Garden perturbations (`v9 CROSS_SCENE_CONFIRMATION_PASS`).

Earlier matched BA evidence remains useful but distinct: it jointly changes cameras and points and produced **+1.591–2.166 dB PSNR** across two scenes and two training seeds. The v8/v9 Gaussian initialization is held fixed between A/C and does not import BA's newly reconstructed points.

## Where to look

| Path | Purpose |
|---|---|
| [`docs/`](docs/project_report_zh.md) | Three documents: report, reproduction, technical notes |
| [`reports/`](reports/README.md) | Current results + one historical archive |
| `bgs/` | Runtime implementation |
| [`scripts/`](docs/reproduction.md#rotation-transfer-v9) | Experiment / evaluation / figure tools; v9 runner and CPU counterexample audit |
| [`configs/rotation_transfer_v9/`](configs/rotation_transfer_v9/) | Exact immutable machine task snapshots for the four successful endpoints |
| [`protocols/`](protocols/rotation_transfer_cross_scene_confirmation_v9.json) | Frozen contracts and append-only execution authorization record |

```bash
python -m unittest discover -s tests -v
python scripts/report.py --verify
python scripts/report.py --figures
```

[Methods, provenance and limits](docs/technical_notes.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
