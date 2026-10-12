# Robust Sparse 3DGS

**Sparse-view reconstruction, failure diagnosis, and controlled geometric experiments.**

I built a COLMAP / VGGT → pinned gsplat workflow, then investigated why sparse-view initialization fails and how camera geometry affects held-out rendering. The emphasis is **root-cause diagnosis and matched experiments**, with reproducible evidence and explicit limits.

[中文项目报告](docs/project_report_zh.md) · [Personal contributions](docs/contributions.md) · [Reproduction](docs/reproduction.md) · [Experiment index](reports/experiment_index.md)

![Initialization failure and controlled repair](docs/figures/initialization.svg)

## Three findings to start with

| Contribution | Evidence | What it demonstrates |
|---|---|---|
| **Locate and repair initialization failure** | Duplicate-point neighborhoods caused zero scales before the first forward pass. A shared dtype/scene-scale floor changed only 4 / 8 degenerate scales in bonsai / room, preserving points and colors. | Failure localization, numerical reasoning, and a controlled repair. |
| **Measure a matched BA operation** | Held-out PSNR improved **+1.591–2.166 dB** across two scenes and two training seeds; SSIM and LPIPS improved in every pair. | Paired controls and attribution of the joint camera/point operation. |
| **Isolate sensitivity to training poses** | With fixed initialization points, 1° rotation cost **0.910 / 3.198 dB** on bonsai / room. | Separation of pose perturbation from point-cloud changes. |

![Matched BA gains on two scenes and two training seeds](docs/figures/ba_gain.svg)

Two seeds reuse the same immutable initialization snapshots: this repeats **training randomness**, not independent initialization. BA jointly changes cameras and points; its gain is not a camera-only effect.

![Controlled rotation and translation sensitivity](docs/figures/pose_sensitivity.svg)

The sensitivity study uses two development scenes, one training seed and one noise-direction set. Added errors are relative to estimated poses; this does not measure absolute pose error or geometry accuracy.

## System and delivery

Train-only staging and immutable splits → COLMAP / VGGT initialization → finite-state checks → bounded gsplat training → calibrated held-out evaluation → PLY export and local WebGL inspection.

Implemented here: data/split integrity, pinned trainer contracts, budgeted experiment queue, matched BA snapshots, numerical diagnostics, pose perturbation controls, evaluation/export and render benchmarking. Upstream models, BA, rasterization and densification are reused; see [provenance](docs/contributions.md).

**Quality / cost tradeoff:** on one South Building scene, 3k versus 7k training reduced optimization time by 60.8% and PLY size by 65.6%, while LPIPS worsened (0.4633 versus 0.3516). See [performance evidence](docs/performance.md).

**Visual outputs:** a real local WebGL interaction and a private 39.97-second reference / prediction / error video were validated. Benchmark-derived RGB/video are excluded from this public repository pending redistribution review. The charts above are measured evidence, not reconstruction images. See [demo record](docs/demo.md).

## Read by purpose

| You want to… | Open |
|---|---|
| Understand the research story in a few minutes | [中文项目报告](docs/project_report_zh.md) |
| Inspect exact measurements and run IDs | [Experiment index](reports/experiment_index.md) |
| Audit failures and historical corrections | [Detailed evidence status](docs/evidence_status.md), [failure analysis](docs/failure_analysis.md) |
| Reproduce the workflow | [Reproduction](docs/reproduction.md), [protocol](docs/protocol.md) |
| Check authorship and technical scope | [Contributions](docs/contributions.md) |

## Quick checks

```bash
python -m unittest discover -s tests -v
python -m bgs --help
python -m bgs.queue --help
```

Optional chart regeneration (requires matplotlib): `python scripts/build_presentation_figures.py`.

## Current boundaries

Production resume has **not passed** the cross-densification qualification. Scene coverage and quality are initializer dependent. No new BA / pose-estimation algorithm, SOTA result, or geometric-ground-truth accuracy is claimed. VGGT-1B checkpoint licensing and potential benchmark pretraining overlap must be considered; weights are not distributed. See [third-party notices](THIRD_PARTY_NOTICES.md).
