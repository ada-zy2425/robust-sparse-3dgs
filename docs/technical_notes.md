# Technical notes

[Project report](project_report_zh.md) · [Reproduction](reproduction.md) · [Evidence](../reports/README.md)

<a id="architecture"></a>
## Architecture

The workflow separates immutable input staging, pinned upstream adapters, bounded execution and held-out evaluation. Initializers receive training RGB only. Full training-view registration is required; failures remain in the reported denominator. Training cameras fit Sim(3); calibrated held-out cameras are evaluated after training. PLY, checkpoints, metrics and resource receipts are outputs of each immutable run.

<a id="provenance"></a>
## Implemented here / reused upstream

**Implemented:** split/hash integrity, exact-source trainer contracts, budgeted queue and task snapshots, matched pre/post-BA capture, initialization diagnostics and scale guard, controlled pose perturbations, single-application fixed-camera wiring, rotation-only camera transfer, evaluation/export, counterexample audit and render benchmarks.

**Reused:** COLMAP, VGGT model and official BA branch, gsplat trainer/rasterizer/densification/viewer, LightGlue, standard PSNR/SSIM/LPIPS and Sim(3). The contribution is system integration and controlled diagnosis; no custom BA, pose estimator or CUDA kernel is claimed.

<a id="protocol"></a>
## Experimental contracts

Comparisons fix views, resolution, training schedule, seed and GPU family. A changed split, source adapter or parameter creates a new identity. Unchanged failed tasks may be retried once with a recorded reason. Final scene/run identities are frozen before scoring; final results do not tune the method.

The 7k schedule uses batch size 1, SH degree 3 (1k interval), refinement after step 500 every 100 steps, opacity reset every 3k steps, opacity prune 0.005 and image-gradient growth 0.0002. Initial Adam rates: means `1.6e-4 × scene_scale`, scales `5e-3`, opacities `5e-2`, quaternions `1e-3`, SH0 `2.5e-3`, higher SH `1.25e-4`. Means decay to 1% at the predeclared endpoint; pose optimization and opacity/scale regularizers are disabled. Full settings remain in the archive and source locks.

The v9 fixed-camera contract compares A (registered perturbed cameras) with C (standard-BA OpenCV c2w rotations plus A's world-space centers). Both arms start from one normal post-BA Gaussian initialization; the per-seed BA points are excluded. The override happens once after Runner construction, then terminal camera bytes must equal the override. Initial Gaussian state, intrinsics, Parser normalization, training-only held-out alignment, training stream, optimizer clocks and default structural policy match. Camera 0 is fixed within `1e-12`; all 24 training cameras, 276 pairs and 24 held-out views remain in each seed's denominator.

This differs from two earlier uses of BA:

- matched pre/post BA changes cameras and 3D points jointly;
- v7 transfers the full BA cameras, including center motion, into a fixed Gaussian initialization;
- v8/v9 transfer only BA rotations and retain the perturbed-input centers.

Standard BA receives credit for all step-0 pose recovery. Rotation transfer is the tested integration choice, not a new solver.

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

For v9 Garden seed 2718, A/C means were 17.0403/19.6529 PSNR, 0.2113/0.4418 SSIM and 0.4560/0.2509 LPIPS. Seed 3141 used 16.9650/19.3116, 0.2130/0.4399 and 0.4505/0.2583. Rotation recovery was 96.44%/96.83% and all-pairs relative improvement 97.10%/97.30%. V9 charged 5.692589 GPUh including asset preparation, failed work, checks and all four endpoints; the complete campaign charged 15.390247/16 GPUh. Fewer or more final Gaussians are recorded outcomes, not a wall-time or algorithmic-efficiency claim.

All-held-out-camera A800 render benchmarks use five warmups and 30 synchronized repeats per camera. Across eight development checkpoints, mean latency ranged 1.488–3.151 ms. Camera tensors were GPU resident; browser/network, evaluation metrics and file I/O are excluded. Raw distributions, resolution and checkpoint hashes are in the `performance` record.

<a id="visuals"></a>
## Visual delivery

A real localhost WebGL viewer loaded a 546,566-Gaussian room model, displayed a calibrated preset and changed the canvas under a small orbit drag. A large 150-pixel drag produced a black view; calibrated presets define the validated interaction range.

A private 39.9667-second 1280×720 / 30 FPS video shows held-out reference / prediction / absolute RGB error panels. It is not a novel-view trajectory. Dataset RGB, video, PLY and model weights remain excluded. The public DSC08108 figure uses cached model predictions and derived mean-absolute-error maps only; it deliberately omits target RGB. Public schematics are labelled as such and quantitative plots use measured machine results.

<a id="limitations"></a>
## Boundaries and unresolved issues

- Matched BA changes cameras and points jointly. Two training seeds reuse the same initialization snapshots.
- VGGT's complete `use_ba` branch also changes tracking, point construction/filtering and camera-model handling; a branch comparison is not a BA-only ablation.
- Pose perturbations are added to estimated training poses. There is no absolute pose-error or geometry-ground-truth accuracy claim.
- V3 photometric repair remains rejected and v7 full-camera transfer remains failed. V8 is a two-scene development pass; v9 confirms two new perturbation seeds only on Garden, itself a historical development scene rather than a never-used external set.
- V8/v9 cover a registered 1° pure-rotation/zero-translation intervention, 24 training views, fixed cameras and one 7k trainer. Mixed rotation/translation errors may require a different center treatment.
- Garden seed 3141 `DSC08108.JPG` loses 3.257 dB PSNR under C even though SSIM/LPIPS improve. Existing caches support a regional upper-image brightness/blue shift emerging between 1k and 3k, with no broad clipping. They do not identify the responsible Gaussian or prove a densification cause.
- Production resume remains unqualified across densification. Exact restored state and forward results were followed by first backward differences of `6.92e-8` (continuous/reload) and `4.84e-8` (same-condition reloads); structural thresholds later amplified them.
- Sealed final scenes completed all three VGGT official-branch arms, while COLMAP returned partial registrations (20/24, 21/24, 4/24) and failed the coverage gate. Failed arms are not assigned quality scores.
- VGGT-1B research weights have non-commercial restrictions; potential benchmark pretraining overlap is not ruled out. See [third-party notices](../THIRD_PARTY_NOTICES.md).

<a id="evidence"></a>
## Evidence preservation

[Current ledger](../reports/results.json) contains primary JSON records grouped by study. [rotation_transfer_v9.json](../reports/rotation_transfer_v9.json) is a compact machine summary with all per-view A/C metrics, task/config hashes, costs, preserved failure summaries and the CPU counterexample audit. [archive.zip](../reports/archive.zip) remains the byte-verified pre-v9 historical archive; it was not rewritten to masquerade new evidence as old. Historical paths and immutable configs document execution, while new runs require new identities and regenerated paths/hashes.
