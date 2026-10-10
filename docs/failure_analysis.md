# Failure analysis

## Sparse COLMAP registration

The first formal South Building sampling spanned two separated capture ranges. COLMAP registered only 3/12 and 5/24 views. A continuous 48-frame segment improved coverage to 6/12 and 15/24 but still left gaps after the camera baseline increased. These are retained results, not discarded setup noise.

The next protocol froze a shorter segment using only ordered filenames and calibration-pose spacing before quality inspection. Its dense 24-view arm registered all views; its nested 12-view arm still failed.

The three-scene Mip-NeRF 360 development matrix made this denominator explicit. Full coverage was obtained in only 1/9 COLMAP arms: bonsai registered 0/6, 2/12 and 23/24; room registered 0/6, 6/12 and 13/24; garden registered 2/6, 5/12 and 24/24. All partial models remain failures. This supports insufficient overlap and scene-dependent feature geometry under the fixed sparse selections more strongly than a GPU, renderer or trainer fault.

## Feed-forward VGGT confidence gate

The 13-point bonsai model is a threshold outcome, not evidence that VGGT predicted only 13 geometric samples. A read-only replay of the pinned 24-view forward pass produced 6,439,776 finite confidence values. At the fixed threshold 5.0, only 13 passed (`0.000202%`); the median was 1.064, p99 was 4.125 and the maximum was 5.137. The 100k cap therefore did not activate. On room, the same replay had median 5.629 and 3,358,153 values (`52.15%`) passed, after which the official seeded random cap selected 100,000. This explains the observed 13 versus 100,000 point counts as a scene-sensitive hard confidence gate plus cap. It does not establish geometric correctness of the rejected or accepted points, and these controls are inactive in the official BA branch.

## Resume qualification

A short no-densification interrupted run passed tolerance, but this is not sufficient. The shape-changing qualification interrupted after the model grew from 208 to 262 Gaussians. With identical saved state, continuous and resumed branches ended with different Gaussian counts and held-out renders. Reliable resume is not claimed; formal runs complete continuously.

The first-divergence diagnostic checked optimizer-to-parameter binding, momentum/step/LR, scheduler and strategy state, Python/NumPy/Torch/CUDA RNG, camera order and rebuilt Scene/Stage references. A continuous branch and two reloads of the same prefix were exact through the contributing forward render; both comparisons first differed in backward at `6.92e-8` and `4.84e-8`. The exercised upstream dense rasterizer uses `atomicAdd_system` for gradient accumulation. Same-condition reloads first made different grow/prune counts at step 900, before continuous versus reload split at step 1000. This localizes the first measured perturbation to backward accumulation and its later amplification by thresholded structural decisions; it does not establish bitwise CUDA determinism or claim a custom kernel fix. The fixed qualification tolerance was not relaxed.

## Paired VGGT diagnosis

At fixed 24 training views, 8 held-out views, 7k schedule, resolution, seed and A800 model, the official VGGT/no-BA initialization produced 100,000 points but underperformed the 9,815-point COLMAP arm on every aggregate image metric. It also grew to 834,713 Gaussians, took 410.6 s of optimization and produced a 197.0 MB PLY, versus 641,484 / 302.4 s / 151.4 MB for COLMAP.

The training-camera alignment diagnostic points more strongly to camera error than point count: VGGT's mean center residual was 0.0499 versus 0.00631 for COLMAP, and mean orientation error was 2.258° versus 0.145°.

The complete official `use_ba` branch supported that hypothesis on this scene. It reduced the center residual to 0.00475 and orientation error to 0.219°, gained 4.271 dB PSNR over the feed-forward branch, raised SSIM by 0.0889 and reduced LPIPS by 0.2457. It reached 16.878/0.3548/0.3189 versus COLMAP's 16.803/0.4274/0.3516, so the metrics remained mixed. Because the switch also changed tracking, point construction/filtering and camera-model handling, these historical differences are branch-level evidence rather than an isolated BA-operation effect.

The controlled 24-view development replication was more informative. VGGT's complete official BA branch produced valid full-coverage initializations on bonsai, room and garden. Relative to the separate feed-forward branch, it improved all three image metrics on bonsai and room. Because the branch also changes tracking, point construction/filtering and camera-model handling, the difference cannot be attributed solely to BA or pose refinement. On garden, the feed-forward branch failed initialization; relative to the successful COLMAP arm, the BA branch improved LPIPS by 0.0242 but reduced PSNR by 0.515 dB, reduced SSIM by 0.0280 and produced 152,846 more Gaussians. The supported conclusion is branch-level reliability plus scene-dependent image-quality behavior—not universal metric or performance superiority.

The later same-reconstruction control isolates the joint BA call more narrowly. Before/after copies retained the same 24 images, tracks, 22,192/36,226 points and camera model. BA reduced mean reprojection error from 10.925 to 2.491 px on bonsai and 10.500 to 2.457 px on room in 1.49 and 4.93 seconds. Historical runs reported both pre-BA snapshots as non-finite at step 100, but the old all-parameter gate ran only at checkpoint intervals. Those failures remain intact; they do not establish that step 100 was the first abnormal operation.

A read-only replay of the immutable raw reconstruction, parser-normalized arrays and exact float32 trainer inputs found the first abnormal operation during initialization. Bonsai contained 745 redundant rows in 733 exact-coordinate clusters and room contained 1,517 in 1,469 clusters. Four bonsai points and eight room points had three zero non-self KNN distances, making `dist2_avg` zero in the pinned `log(sqrt(dist2_avg) * init_scale)` path. This created 12 and 24 negative-infinite scale components, with zero physical scales, before any forward, loss, backward, optimizer or strategy operation. All other initial parameter fields were finite; optimizer parameters were correctly bound and their state was empty. The post-BA snapshots had no exact duplicate points or zero scales. The audit receipt is [splat initialization non-finite v1](../reports/diagnostics/splat_initialization_nonfinite_v1.json).

Two conventional repairs were then compared without changing the historical identities. Exact deduplication passed but removed 745/1,517 points and averaged colors in 551/1,200 conflicting clusters. The selected `dtype_scene_scale_floor_v1` instead clamps the raw scale to `eps(dtype) * max(max_abs_coordinate, scene_scale) * init_scale`; it changed only the 4/8 degenerate pre-BA points, changed no point, color or coverage, and was a zero-change operation on both post-BA snapshots. All four selected-policy arms passed an every-step sample/forward/loss/backward/update/strategy scan through steps 600, 700 and 800 densification. In the frozen seed-42 7k comparison, every arm completed: post minus pre was +1.6061 dB PSNR, +0.0507 SSIM and -0.1377 LPIPS on bonsai, and +2.1659 dB, +0.0806 and -0.0895 on room. Seed 43 retained all effect directions, but reuses the same initialization snapshots and therefore tests downstream training randomness only. These are development-scene estimates of the joint camera-and-point BA operation after a shared numerical guard; they still do not separate camera from point optimization. The guard is ordinary numerical robustness, not a new reconstruction algorithm.

The full development matrix completed 5/18 predeclared no-BA/COLMAP runs. VGGT no-BA succeeded on bonsai-24 and room-6/12/24, while it failed all garden view counts and bonsai-6/12 because too few points survived. Point count and initializer label alone therefore do not predict end-to-end success.

The untouched final scenes confirmed the reliability direction. COLMAP produced partial models for counter (20/24), kitchen (21/24) and bicycle (4/24), so all three failed the frozen full-coverage gate. Official VGGT BA reached 24/24 and completed training on all three. No retry was used because each COLMAP failure arose from the fixed image overlap/matching result rather than a transient runtime error.

This confirmation does not imply a final-set image-quality victory: failed COLMAP arms have no valid held-out score, and averaging only successful methods would change the denominator. The bicycle BA result also retained conspicuous foreground grass blur/floater artifacts and had the weakest final metrics despite complete initialization. Reliability and reconstruction fidelity remain separate axes.

Three dependency failures are preserved rather than rewritten: v1 lacked the VGGSfM tracker, v2 lacked offline DINOv2 source, and v3's first attempt lacked ALIKED. After exact cache qualification, unchanged v3 used its sole reasoned retry and completed. The successful retry also exposed and cached SuperPoint. BA is an upstream established technique; the project contribution here is controlled diagnosis, offline qualification and evidence accounting.

## Limits

- Evidence covers three development and three final scenes with one seed per main arm; it diagnoses repeatable behavior but is not a strong statistical generalization claim.
- No ground-truth geometry is available, so PSNR/SSIM/LPIPS are image-space metrics only.
- Mip-NeRF 360/South Building redistribution terms for derived presentation assets have not been resolved here; images, rendered panels, videos and PLYs remain private.
- Possible overlap between VGGT pretraining and benchmark scenes is unknown.
- Official BA is upstream established work. This project implements the controlled protocol, isolation, qualification and evidence accounting; it does not present BA as a new algorithm.
