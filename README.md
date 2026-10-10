# Robust Sparse 3DGS

An evidence-first workflow for reconstructing static scenes from low-cost sparse captures with COLMAP or VGGT initialization and a pinned gsplat trainer.

The project focuses on a practical failure mode: a pipeline can render attractive training views while sparse-view registration, held-out quality, or resumability is broken. The harness therefore treats failures as results, keeps held-out RGB out of initialization, and records immutable run identities, global GPU budget, artifacts, and hardware.

## What is implemented

- Train-only staging and immutable nested view splits.
- Pinned COLMAP/VGGT adapters and source-contract checks.
- Held-out RGB evaluation after training-camera-only Sim(3) alignment.
- A persistent single-node queue with atomic global GPU-hour reservations, GPU UUID checks, dependency tracking, failure classification, and duplicate-coordinator protection.
- Full-state checkpoints and a strict interrupted-versus-continuous qualification test.
- PLY export, localhost-only viewer integration with calibrated camera presets, and bounded multi-camera render timing.

The initial 200-step runs are integration results, not quality claims. The first formal 24-view comparison and an evidence-led 3k/7k quality-cost control are recorded in [the result report](reports/formal_stage_v1.md) and [CSV](reports/results.csv).

## Quick checks

```bash
python -m unittest discover -s tests -v
python -m bgs --help
python -m bgs.queue --help
```

Deployment intentionally remains explicit because CUDA, COLMAP, gsplat and VGGT need separate pinned environments. See [reproduction](docs/reproduction.md) and the sanitized [config example](configs/run.example.json).

## Evidence status

- Engineering path: validated on A800 GPUs from immutable staging through initialization, training, genuine held-out renders, metrics, PLY and a localhost viewer. A browser/WebGL test verified a recognizable calibrated default view and a changed render after orbit drag; the viewer was then stopped.
- Formal first-scene baseline: 24 train / 8 held-out views. COLMAP+gsplat at 7k reached 16.803 dB PSNR, 0.4274 SSIM and 0.3516 LPIPS; VGGT+gsplat at the same schedule reached 12.606/0.2659/0.5646. This is one development scene, not cross-scene evidence.
- Efficiency control: a separately scheduled COLMAP 3k run improved PSNR/SSIM to 18.120/0.4791 and cut optimization time 60.8% and PLY size 65.6%, but LPIPS worsened to 0.4633. Its formal 8-camera render benchmark reduced mean latency 20.7%, increased FPS 26.1% and reduced peak allocated memory 62.2% versus 7k. It is reported as a tradeoff, not a universal quality gain.
- Registration reliability: five sparse selections failed full coverage and remain in the denominator.
- Resume: a 20-step no-densification check passed, but shape-changing resume did not reproduce the continuous trajectory. Production resume is therefore not claimed.
- VGGT: the separate pinned runtime, official no-BA baseline and official-BA diagnostic are qualified. BA reduced mean camera-center residual from 0.0499 to 0.00475 and improved the no-BA held-out result from 12.606/0.2659/0.5646 to 16.878/0.3548/0.3189. This is a one-scene upstream-technique diagnostic, not a new method. The original VGGT-1B checkpoint used for research is non-commercial and is never included here.

![First-scene quality/cost chart](reports/quality_cost.svg)

See [limitations](docs/failure_analysis.md), [contributions](docs/contributions.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
