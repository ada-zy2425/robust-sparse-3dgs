# Mip-NeRF 360 development stage

This report covers three development scenes chosen before scoring: bonsai (object-centric), room (indoor), and garden (outdoor). The official `360_v2` archive is pinned by SHA256 `77332bf4eba3b8ca0c7f70130849b1e394efdd60d8f20efa6f217081d08a8b2a`; the frozen development registry is `dfe226610f84bec694792c20861093ec745c9602215c6b6bb1822c13154b8b1b`.

Every arm uses nested 6/12/24 training views, a fixed every-eighth-image holdout (37/39/24 images), max edge 1024, seed 42, and a predeclared 7k total schedule with 1k/3k/7k checkpoints. Held-out RGB is absent from initialization and training. Full-capture SfM poses are calibration metadata rather than geometry truth; Sim(3) alignment is fitted on training cameras only. All GPU work ran on A800-SXM4-80GB devices under one 24 GPU-hour queue.

## Registration and initialization reliability

Full training-view coverage is required. Partial COLMAP models and zero-point VGGT runs remain failures in the denominator.

| Scene | Views | COLMAP registered | COLMAP result | VGGT no-BA result |
|---|---:|---:|---|---|
| bonsai | 6 | 0/6 | failed | failed: too few points |
| bonsai | 12 | 2/12 | failed | failed: too few points |
| bonsai | 24 | 23/24 | failed | valid: 24/24, 13 points |
| room | 6 | 0/6 | failed | valid: 6/6, 100,000 points |
| room | 12 | 6/12 | failed | valid: 12/12, 100,000 points |
| room | 24 | 13/24 | failed | valid: 24/24, 100,000 points |
| garden | 6 | 2/6 | failed | failed: too few points |
| garden | 12 | 5/12 | failed | failed: too few points |
| garden | 24 | 24/24 | valid: 6,883 points | failed: too few points |

The fixed matrix therefore produced full-coverage initializations in 1/9 COLMAP arms and 4/9 no-BA VGGT arms; overall end-to-end success was 5/18. The 13 failed identities consumed 0.3834 GPU-hours and are not removed from success rates. Earlier immutable v1 and v2 matrices are also retained: v1 exposed an outer reservation smaller than the inner timeout-plus-cleanup requirement, while v2 exposed missing initialization headroom after a successful initializer. Versioned v3 corrected the reservation only; no failed identity was rewritten.

## Successful v3 quality runs

| Scene / views / initializer | Init pts | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Final GS | Optimization s | End-to-end s | GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai / 24 / VGGT no BA | 13 | 13.102 | 0.3713 | 0.6813 | 631,496 | 1,564.0 | 1,795.6 | 0.4988 |
| room / 6 / VGGT no BA | 100,000 | 14.175 | 0.4246 | 0.4899 | 297,925 | 1,894.8 | 2,272.1 | 0.6311 |
| room / 12 / VGGT no BA | 100,000 | 16.738 | 0.5447 | 0.3767 | 493,660 | 1,675.9 | 2,095.5 | 0.5821 |
| room / 24 / VGGT no BA | 100,000 | 17.614 | 0.5859 | 0.3356 | 728,207 | 1,595.5 | 1,841.9 | 0.5116 |
| garden / 24 / COLMAP | 6,883 | 20.126 | 0.4642 | 0.2737 | 1,389,863 | 4,254.3 | 4,541.7 | 1.2616 |

More views helped room quality, but method success was scene dependent. Point count alone was not a reliability signal: bonsai's nominally valid 13-point VGGT model was weak, while garden's no-BA path produced no usable points. A later pinned inference replay showed that bonsai did not “predict only 13 points”: 13 of 6,439,776 finite confidence values passed the fixed 5.0 gate, whereas room had 3,358,153 pass and was then randomly capped at 100,000. The hard gate is therefore the direct cause of the retained count difference; it is not a geometry-absence diagnosis.

## Official BA-branch system comparison

The development observation implicated camera estimation, so the next round enabled VGGT's official upstream `use_ba` branch at 24 views. This is a system comparison, not a one-factor BA ablation: the branch uses VGGSfM tracking, track/reprojection filtering, tracked-point construction and `SIMPLE_PINHOLE` handling before calling `pycolmap.bundle_adjustment`. The feed-forward branch instead applies `conf_thres_value`, randomly limits accepted depth points to 100k and uses a `PINHOLE` model. Although the old submitted configs contain confidence and max-point fields for both arms, those fields are inactive in the BA branch. Manifest, views, gsplat schedule, resolution, seed, renderer and GPU family stayed fixed. BA is established upstream work; this repository contributes the controlled diagnosis, offline dependency qualification, corrected effective-configuration receipts and evidence accounting.

| Scene | Reference arm | Reference PSNR / SSIM / LPIPS | Official BA PSNR / SSIM / LPIPS | BA init pts | BA final GS | BA center / orientation residual |
|---|---|---|---|---:|---:|---:|
| bonsai | VGGT no BA | 13.102 / 0.3713 / 0.6813 | 17.390 / 0.5021 / 0.3487 | 22,192 | 742,208 | 0.0300 / 0.666° |
| room | VGGT no BA | 17.614 / 0.5859 / 0.3356 | 19.756 / 0.6657 / 0.2674 | 36,222 | 546,566 | 0.0520 / 0.300° |
| garden | VGGT no BA failed; COLMAP shown | 20.126 / 0.4642 / 0.2737 | 19.611 / 0.4362 / 0.2495 | 26,941 | 1,542,709 | 0.00861 / 0.0729° |

At 3k, garden BA reached 19.871/0.4462/0.3318 versus COLMAP's 20.359/0.4579/0.3815: LPIPS improved while PSNR and SSIM did not. This mixed metric outcome is why the final protocol tests a paired diagnosis rather than assuming a universal win.

The same tradeoff remained at 7k: BA improved garden LPIPS by 0.0242 and converted a no-BA initialization failure into a complete model, while PSNR fell 0.515 dB, SSIM fell 0.0280, and the model grew by 152,846 Gaussians relative to COLMAP. BA end-to-end costs were 0.5113/0.5179/1.1422 GPU-hours for bonsai/room/garden respectively.

The first BA-control identity is retained as a dependency failure caused by an incorrect local DINOv2 source root. Versioned v2 fixed that configuration before GPU initialization and used no retry.

## Matched BA-operation ablation

The branch-level comparison above cannot identify the BA call. The matched control therefore saved each official-branch reconstruction immediately before `pycolmap.bundle_adjustment`, loaded two distinct `pycolmap.Reconstruction` objects from that immutable snapshot, applied BA only to the post copy and wrote the two models to separate directories. The receipts verify different object identities and an unchanged source snapshot. The downstream image set, tracks, initial points, camera model, preprocessing, 7k schedule, resolution and training seed were otherwise matched. The original feed-forward/no-BA pipeline is not one of these arms.

| Scene | Images / points / observations | Mean reprojection px pre → post | Camera center / rotation mean change | Point mean change | BA time |
|---|---:|---:|---:|---:|---:|
| bonsai | 24 / 22,192 / 64,192 | 10.925 → 2.491 | 0.0310 / 1.432° | 0.0629 | 1.49 s |
| room | 24 / 36,226 / 263,074 | 10.500 → 2.457 | 0.0415 / 0.283° | 0.0773 | 4.93 s |

At seed 42, both pre-BA arms became non-finite at the first step-100 finite/checkpoint gate; the single allowed exact retry failed in the same place. The post-BA arms completed: bonsai reached 17.4095/0.5015/0.3464 with 740,972 Gaussians, while room reached 19.7717/0.6735/0.2683 with 546,696. Their formal all-heldout-camera render means were 2.065 and 1.831 ms. A seed-43 repeat reused the immutable seed-42 initialization pair, so it tests downstream training randomness only: both pre arms again failed at step 100 and both post arms completed at 17.4630/0.5043/0.3455 and 19.8260/0.6718/0.2681.

There are therefore zero valid paired quality/model/render endpoints. The supported conclusion is that the joint BA operation changed downstream trainability for these two snapshots across two training seeds, while reducing their reprojection error. It is not evidence for a numerical PSNR/SSIM/LPIPS gain, cannot separate camera optimization from point optimization, and is not an independent-initialization repeat. Failed rows, logs and costs remain in the [seed-42 report](ba_operation_ablation_s42_v1/results.md), [seed-43 report](ba_operation_ablation_s43_v2/results.md) and [aggregate receipt](ba_operation_ablation_seed_aggregate_v1.json).

## Multi-camera rendering performance

The benchmark visits every fixed held-out camera, warms up five times per camera, then records 30 synchronized render-only samples. Tensors are resident on the GPU; these numbers exclude browser, transfer, metric, and file-writing time.

| Checkpoint | Cameras / samples | GS | Mean ms ↓ | p95 ms | FPS ↑ | Peak allocated MB |
|---|---:|---:|---:|---:|---:|---:|
| bonsai VGGT no BA | 37 / 1,110 | 631,496 | 1.865 | 2.939 | 536.2 | 683.3 |
| bonsai VGGT + official BA | 37 / 1,110 | 742,208 | 2.063 | 2.319 | 484.6 | 782.1 |
| room 6-view VGGT no BA | 39 / 1,170 | 297,925 | 1.488 | 1.826 | 672.0 | 340.5 |
| room 12-view VGGT no BA | 39 / 1,170 | 493,660 | 1.728 | 2.207 | 578.7 | 536.6 |
| room 24-view VGGT no BA | 39 / 1,170 | 728,207 | 2.071 | 2.517 | 482.8 | 784.2 |
| room 24-view VGGT + official BA | 39 / 1,170 | 546,566 | 1.842 | 2.228 | 542.9 | 586.5 |
| garden 24-view COLMAP | 24 / 720 | 1,389,863 | 2.864 | 3.598 | 349.1 | 1,429.5 |
| garden 24-view VGGT + official BA | 24 / 720 | 1,542,709 | 3.151 | 3.489 | 317.4 | 1,590.0 |

BA's performance consequence is scene dependent: bonsai quality improved with a larger, slower model; room quality improved with a smaller, faster model. The cross-check reinforces Gaussian count as a practical latency/memory driver, but it is not a custom CUDA or kernel contribution.

The garden BA benchmark also completed on its originally assigned A800 after external contention cleared. Its 720-sample mean was 3.14795 ms versus 3.15096 ms for the independent replacement task, a 0.096% relative difference; both receipts report the same peak allocation. The table uses the slightly slower replacement result and both receipts are retained.

## Interpretation and limits

- The 6/12/24 matrix establishes failure rates and one-view-count curve on development data; it does not create a strong statistical generalization claim.
- The official BA branch gives full-coverage initialization on all three 24-view inputs and materially improves bonsai and room held-out quality. This supports the complete upstream branch, not the isolated BA call. Garden confirms a reliability and LPIPS benefit, but not a PSNR/SSIM or performance benefit.
- The matched operation control verifies a trainability difference on two development snapshots, but its pre-BA failures prevent an image-quality effect estimate. A separate predeclared mechanism study would be needed before treating that stability observation as a general algorithmic result.
- Mip-NeRF 360 redistribution terms were not resolved, so RGB, derived comparison canvases, PLYs, and videos stay in private evidence storage.
- VGGT pretraining overlap with benchmark scenes is unknown and explicitly not ruled out.
- Shape-changing resume remains unqualified; every formal result completed continuously.
