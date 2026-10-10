# Failure analysis

## Sparse COLMAP registration

The first formal South Building sampling spanned two separated capture ranges. COLMAP registered only 3/12 and 5/24 views. A continuous 48-frame segment improved coverage to 6/12 and 15/24 but still left gaps after the camera baseline increased. These are retained results, not discarded setup noise.

The next protocol froze a shorter segment using only ordered filenames and calibration-pose spacing before quality inspection. Its dense 24-view arm registered all views; its nested 12-view arm still failed. Current evidence therefore supports “insufficient overlap under the sparse selection” more strongly than a GPU, renderer or training fault.

## Resume qualification

A short no-densification interrupted run passed tolerance, but this is not sufficient. The shape-changing qualification interrupted after the model grew from 208 to 262 Gaussians. With identical saved RNG and scheduler states, continuous and resumed branches ended with different Gaussian counts and held-out renders. Reliable resume is not claimed; formal runs complete continuously.

CUDA/rasterization non-determinism amplified by discrete densification thresholds is a working explanation, not yet a proven root cause. The preserved checkpoint comparison is the next diagnostic input.

## Paired VGGT diagnosis

At fixed 24 training views, 8 held-out views, 7k schedule, resolution, seed and A800 model, the official VGGT/no-BA initialization produced 100,000 points but underperformed the 9,815-point COLMAP arm on every aggregate image metric. It also grew to 834,713 Gaussians, took 410.6 s of optimization and produced a 197.0 MB PLY, versus 641,484 / 302.4 s / 151.4 MB for COLMAP.

The training-camera alignment diagnostic points more strongly to camera error than point count: VGGT's mean center residual was 0.0499 versus 0.00631 for COLMAP, and mean orientation error was 2.258° versus 0.145°.

The official bundle-adjustment control supported that hypothesis on this scene. It reduced the center residual to 0.00475 and orientation error to 0.219°, gained 4.271 dB PSNR over no-BA, raised SSIM by 0.0889 and reduced LPIPS by 0.2457. It reached 16.878/0.3548/0.3189 versus COLMAP's 16.803/0.4274/0.3516, so the metrics remain mixed rather than establishing a universal winner.

Three dependency failures are preserved rather than rewritten: v1 lacked the VGGSfM tracker, v2 lacked offline DINOv2 source, and v3's first attempt lacked ALIKED. After exact cache qualification, unchanged v3 used its sole reasoned retry and completed. The successful retry also exposed and cached SuperPoint. BA is an upstream established technique; the project contribution here is controlled diagnosis, offline qualification and evidence accounting.

## Limits

- Current scored evidence is from one development capture and cannot establish cross-scene generalization.
- No ground-truth geometry is available, so PSNR/SSIM/LPIPS are image-space metrics only.
- The South Building redistribution status has not been cleared; images and rendered assets remain private.
