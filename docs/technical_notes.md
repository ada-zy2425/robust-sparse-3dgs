# Technical notes

[Project report](project_report_zh.md) · [Reproduction](reproduction.md) · [Evidence](../reports/README.md)

<a id="architecture"></a>
## Architecture

The workflow separates immutable input staging, pinned upstream adapters, bounded execution and held-out evaluation. Initializers receive training RGB only. Full training-view registration is required; failures remain in the reported denominator. Training cameras fit Sim(3); calibrated held-out cameras are evaluated after training. PLY, checkpoints, metrics and resource receipts are outputs of each immutable run.

<a id="provenance"></a>
## Implemented here / reused upstream

**Implemented:** split/hash integrity, exact-source trainer contracts, budgeted queue and task snapshots, matched pre/post-BA capture, initialization diagnostics and scale guard, controlled pose perturbations, evaluation/export and render benchmarks.

**Reused:** COLMAP, VGGT model and official BA branch, gsplat trainer/rasterizer/densification/viewer, LightGlue, standard PSNR/SSIM/LPIPS and Sim(3). The contribution is system integration and controlled diagnosis; no custom BA, pose estimator or CUDA kernel is claimed.

<a id="protocol"></a>
## Experimental contracts

Comparisons fix views, resolution, training schedule, seed and GPU family. A changed split, source adapter or parameter creates a new identity. Unchanged failed tasks may be retried once with a recorded reason. Final scene/run identities are frozen before scoring; final results do not tune the method.

The 7k schedule uses batch size 1, SH degree 3 (1k interval), refinement after step 500 every 100 steps, opacity reset every 3k steps, opacity prune 0.005 and image-gradient growth 0.0002. Initial Adam rates: means `1.6e-4 × scene_scale`, scales `5e-3`, opacities `5e-2`, quaternions `1e-3`, SH0 `2.5e-3`, higher SH `1.25e-4`. Means decay to 1% at the predeclared endpoint; pose optimization and opacity/scale regularizers are disabled. Full settings remain in the archive and source locks.

The numerical guard applies a positive floor before the log:

```python
raw_scale = sqrt(mean(nonself_knn_distances ** 2)) * init_scale
floor = eps(dtype) * max(max_abs_coordinate, scene_scale) * init_scale
log_scale = log(max(raw_scale, floor))
```

The implemented tensor path validates shape, dtype and finite inputs, reports clamped counts and preserves points/colors. Four bonsai and eight room pre-BA scales were clamped; neither post-BA snapshot triggered the floor.

<a id="performance"></a>
## Quality / cost

On South Building, 3k versus 7k training used 220,795 versus 641,484 Gaussians, 118.7 versus 302.4 optimization seconds and 52.1 versus 151.4 MB PLY. LPIPS worsened from 0.3516 to 0.4633. This is an early-stopping tradeoff on one scene.

All-held-out-camera A800 render benchmarks use five warmups and 30 synchronized repeats per camera. Across eight development checkpoints, mean latency ranged 1.488–3.151 ms. Camera tensors were GPU resident; browser/network, evaluation metrics and file I/O are excluded. Raw distributions, resolution and checkpoint hashes are in the `performance` record.

<a id="visuals"></a>
## Visual delivery

A real localhost WebGL viewer loaded a 546,566-Gaussian room model, displayed a calibrated preset and changed the canvas under a small orbit drag. A large 150-pixel drag produced a black view; calibrated presets define the validated interaction range.

A private 39.9667-second 1280×720 / 30 FPS video shows held-out reference / prediction / absolute RGB error panels. It is not a novel-view trajectory. Benchmark-derived RGB/video, PLY and model weights are excluded pending redistribution review. Public mechanism diagrams are explicitly schematic; plots use measured results.

<a id="limitations"></a>
## Boundaries and unresolved issues

- Matched BA changes cameras and points jointly. Two training seeds reuse the same initialization snapshots.
- VGGT's complete `use_ba` branch also changes tracking, point construction/filtering and camera-model handling; a branch comparison is not a BA-only ablation.
- Pose perturbations are added to estimated training poses. There is no absolute pose-error or geometry-ground-truth accuracy claim; two scenes / one noise draw do not establish general robustness.
- Production resume remains unqualified across densification. Exact restored state and forward results were followed by first backward differences of `6.92e-8` (continuous/reload) and `4.84e-8` (same-condition reloads); structural thresholds later amplified them.
- Sealed final scenes completed all three VGGT official-branch arms, while COLMAP returned partial registrations (20/24, 21/24, 4/24) and failed the coverage gate. Failed arms are not assigned quality scores.
- VGGT-1B research weights have non-commercial restrictions; potential benchmark pretraining overlap is not ruled out. See [third-party notices](../THIRD_PARTY_NOTICES.md).

<a id="evidence"></a>
## Evidence preservation

[Current ledger](../reports/results.json) contains full primary JSON records, grouped by study. [archive.zip](../reports/archive.zip) preserves all pre-consolidation evidence, detailed notes and merged script sources with a byte-count / SHA-256 manifest. Historical private paths and run hashes describe their original execution. New checkouts regenerate tasks using the current CLI; they do not replay historical paths blindly.
