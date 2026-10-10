# Failure analysis

## Sparse COLMAP registration

The first formal South Building sampling spanned two separated capture ranges. COLMAP registered only 3/12 and 5/24 views. A continuous 48-frame segment improved coverage to 6/12 and 15/24 but still left gaps after the camera baseline increased. These are retained results, not discarded setup noise.

The next protocol froze a shorter segment using only ordered filenames and calibration-pose spacing before quality inspection. Its dense 24-view arm registered all views; its nested 12-view arm still failed.

The three-scene Mip-NeRF 360 development matrix made this denominator explicit. Full coverage was obtained in only 1/9 COLMAP arms: bonsai registered 0/6, 2/12 and 23/24; room registered 0/6, 6/12 and 13/24; garden registered 2/6, 5/12 and 24/24. All partial models remain failures. This supports insufficient overlap and scene-dependent feature geometry under the fixed sparse selections more strongly than a GPU, renderer or trainer fault.

## Resume qualification

A short no-densification interrupted run passed tolerance, but this is not sufficient. The shape-changing qualification interrupted after the model grew from 208 to 262 Gaussians. With identical saved RNG and scheduler states, continuous and resumed branches ended with different Gaussian counts and held-out renders. Reliable resume is not claimed; formal runs complete continuously.

CUDA/rasterization non-determinism amplified by discrete densification thresholds is a working explanation, not yet a proven root cause. The preserved checkpoint comparison is the next diagnostic input.

## Paired VGGT diagnosis

At fixed 24 training views, 8 held-out views, 7k schedule, resolution, seed and A800 model, the official VGGT/no-BA initialization produced 100,000 points but underperformed the 9,815-point COLMAP arm on every aggregate image metric. It also grew to 834,713 Gaussians, took 410.6 s of optimization and produced a 197.0 MB PLY, versus 641,484 / 302.4 s / 151.4 MB for COLMAP.

The training-camera alignment diagnostic points more strongly to camera error than point count: VGGT's mean center residual was 0.0499 versus 0.00631 for COLMAP, and mean orientation error was 2.258° versus 0.145°.

The official bundle-adjustment control supported that hypothesis on this scene. It reduced the center residual to 0.00475 and orientation error to 0.219°, gained 4.271 dB PSNR over no-BA, raised SSIM by 0.0889 and reduced LPIPS by 0.2457. It reached 16.878/0.3548/0.3189 versus COLMAP's 16.803/0.4274/0.3516, so the metrics remained mixed.

The controlled 24-view development replication was more informative. Official BA produced valid full-coverage initializations on bonsai, room and garden. Relative to no-BA, it improved all three image metrics on bonsai and room. On garden, no-BA failed initialization; relative to the successful COLMAP arm, BA improved LPIPS by 0.0242 but reduced PSNR by 0.515 dB, reduced SSIM by 0.0280 and produced 152,846 more Gaussians. The supported conclusion is reliability plus scene-dependent image-quality behavior—not universal metric or performance superiority.

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
