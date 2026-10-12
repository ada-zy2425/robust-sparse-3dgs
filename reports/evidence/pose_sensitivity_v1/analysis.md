# Controlled pose sensitivity: analysis

This development-only experiment adds controlled errors to the existing estimated **training** poses. It does not measure known absolute pose error, geometry accuracy or general robustness. The frozen protocol is [`protocols/pose_sensitivity_v1.json`](../../../protocols/pose_sensitivity_v1.json); machine-readable endpoints and immutable artifact paths are in [`results.json`](results.json).

## Control validity

- Scenes: bonsai and room, 24 fixed training views and the existing 37/39-image holdouts.
- Initialization: the same immutable post-BA reconstruction files used by `ba_operation_ablation_floor_v1`; camera/point files match by SHA256 across every condition.
- Fixed downstream contract: `dtype_scene_scale_floor_v1`, max edge 1024, 7k steps, training seed 42, noise seed 1729 and pose optimization disabled.
- Alignment was fitted from unperturbed training cameras and is byte-identical across conditions. Initial points, intrinsics, RGB and held-out cameras are unchanged.
- A CPU probe exercised the pinned gsplat `Parser` and `Dataset`, including the adapter's all-selected/max-edge wrapper. All 24 samples returned the perturbed float32 camera, identical RGB/intrinsics and an unchanged index-0 anchor.
- The old zero runs and new runs do not have byte-identical wrapper source trees: `__main__.py` later gained a configure flag, `pipeline.py` gained policy validation and one expected-output check, and three unused macOS AppleDouble files were removed. Every training/evaluation module matches, including data, geometry, initialization, numerical policy, metric code and `train_adapter`; the report records this narrower behavior audit instead of claiming full-source identity.

Camera radius is the maximum distance from a training center to the arithmetic mean of all 24 unperturbed centers in normalized model coordinates: 1.193105 for bonsai and 0.746454 for room. The anchored cameras are `DSCF5566.JPG` and `DSCF4668.JPG`. Rotation left-multiplies camera-to-world orientation by a fixed-angle world/model-frame Rodrigues rotation; translation adds a fixed-direction displacement of the declared radius fraction. Within each modality, 0.5/1.0 levels share directions and differ only in magnitude.

## Formal 7k endpoints

| Scene | Condition | PSNR Δ | SSIM Δ | LPIPS Δ | GS | PLY bytes | train s | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bonsai | zero | 0.000 | 0.0000 | 0.0000 | 738,354 | 174,253,021 | 1,741.8 | 5.62 |
| bonsai | rotation 0.5° | -0.312 | -0.0120 | +0.0605 | 814,686 | 192,267,373 | 1,805.6 | 6.07 |
| bonsai | rotation 1.0° | -0.910 | -0.0324 | +0.1155 | 786,091 | 185,518,953 | 1,791.0 | 5.81 |
| bonsai | translation 0.5% radius | -0.081 | -0.0038 | +0.0244 | 784,420 | 185,124,597 | 1,766.2 | 5.94 |
| bonsai | translation 1.0% radius | -0.207 | -0.0103 | +0.0503 | 800,094 | 188,823,661 | 1,744.7 | 6.03 |
| room | zero | 0.000 | 0.0000 | 0.0000 | 547,025 | 129,099,377 | 1,689.7 | 3.60 |
| room | rotation 0.5° | -2.034 | -0.0776 | +0.1159 | 616,602 | 145,519,549 | 1,693.8 | 3.90 |
| room | rotation 1.0° | -3.198 | -0.1140 | +0.1836 | 562,262 | 132,695,309 | 1,701.7 | 3.53 |
| room | translation 0.5% radius | -0.278 | -0.0167 | +0.0568 | 651,303 | 153,708,985 | 1,717.7 | 4.17 |
| room | translation 1.0% radius | -1.372 | -0.0598 | +0.0988 | 599,465 | 141,475,217 | 1,694.4 | 3.80 |

All eight new arms succeeded once and continuously. Their total cost was 3.987086 GPUh; the shared ledger ended at 20.604399/24 GPUh. The zero rows are cited immutable prior runs and are not double-charged to this round. Curves are in [`rotation_sensitivity.svg`](rotation_sensitivity.svg) and [`translation_sensitivity.svg`](translation_sensitivity.svg).

## Fixed-view visual review

The views were fixed before scoring: bonsai `DSCF5565.JPG` and room `DSCF4667.JPG`. Each private panel is reference | prediction | absolute RGB error; paths and hashes are recorded in `results.json`.

- Bonsai already has baseline blur/error on foliage, bicycle spokes, tablecloth texture and disocclusion boundaries. Rotation increases double contours, ghosting and floater-like colored streaks around branches, bicycle tubes and the table edge; 1° is visibly worse than 0.5°. Translation produces the same categories more mildly at 0.5% and more broadly at 1%.
- Room zero is substantially sharper. Rotation introduces global smearing and double edges around the chair, table, television and foreground occlusion; keyboard, glassware, cables and furniture edges lose thin detail, especially at 1°. Translation is milder at 0.5%, while 1% adds visible foreground/edge blur and larger residuals.
- Boundary/occlusion and poorly observed regions already fail in the zero panels, so those pixels are not attributed to pose perturbation. A 2D error panel cannot establish whether a streak is a 3D floater; the description is deliberately appearance-level.

The curve is monotone for these two scenes and this direction set, but one training seed and one perturbation draw do not support a statistical or cross-domain claim. No final scene was opened and no geometry ground truth exists.
