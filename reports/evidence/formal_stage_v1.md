# Formal stage v1 — first development scene

Protocol: South Building segment, 24 training views and 8 genuine held-out views, max edge 1024, seed 42, A800-SXM4-80GB. Held-out RGB is excluded from initialization/training; full-capture SfM poses are calibration metadata, and Sim(3) is fitted with training cameras only. Image redistribution rights are unresolved, so no RGB or derived renders are stored here.

| Arm | Init points | Final GS | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Optim. s | PLY MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| COLMAP, 7k | 9,815 | 641,484 | 16.803 | 0.4274 | **0.3516** | 302.4 | 151.4 |
| COLMAP, 3k | 9,831 | 220,795 | **18.120** | **0.4791** | 0.4633 | **118.7** | **52.1** |
| VGGT, 7k, BA off | 100,000 | 834,713 | 12.606 | 0.2659 | 0.5646 | 410.6 | 197.0 |
| VGGT, 7k, official BA | 29,623 | 684,451 | 16.878 | 0.3548 | **0.3189** | 306.5 | 161.5 |

The 3k arm is a separately declared total schedule, not a resumed 7k run. Relative to COLMAP 7k it cut optimization time 60.8% and PLY size 65.6%, with +1.317 dB PSNR and +0.0516 SSIM, but LPIPS worsened by 0.1117. The result is a useful selectable tradeoff, not a universal improvement.

VGGT's much larger no-BA point pool did not help. Its training-camera Sim(3) residuals were substantially worse: center mean 0.0499 vs 0.00631, orientation mean 2.258° vs 0.145°. The controlled official-BA arm reduced these to 0.00475 and 0.219°, respectively. Relative to no-BA it gained 4.271 dB PSNR and 0.0889 SSIM and reduced LPIPS by 0.2457 at a similar optimization cost to COLMAP. Against COLMAP 7k it had comparable PSNR (+0.075 dB), lower SSIM (-0.0726), and better LPIPS (-0.0327). This supports the camera-error diagnosis on this scene; BA is an established upstream method, and the result is not a new algorithm or cross-scene claim.

The BA dependency failures remain evidence. The first identity lacked the pinned tracker, the second lacked a local DINOv2 source, and v3's first attempt lacked ALIKED. Exact caches and hashes were then qualified; the unchanged v3 identity used its one reasoned retry and succeeded. Its reported 0.2676 GPU-hours and 963.5 seconds include that preserved failed attempt.

Formal render-only timing used every held-out camera with synchronized GPU measurements. COLMAP 7k averaged 2.607 ms (383.6 FPS), COLMAP 3k averaged 2.067 ms (483.7 FPS), and VGGT+BA 7k averaged 2.752 ms (363.4 FPS), 240 timed samples per arm. The 3k speed and memory improvement is paired with its LPIPS regression; it is not presented as free quality.

Five incomplete COLMAP registrations (3/12, 5/24, 6/12, 15/24 and 7/12) remain in the reliability denominator. Shape-changing resume also remains unqualified. See `results.csv` for machine-readable rows and `quality_cost.svg` for the plotted tradeoff.
