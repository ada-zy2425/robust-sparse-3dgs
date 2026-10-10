# Corrected matched BA-operation ablation — seed 43

This repeat reuses the immutable seed-42 pre/post reconstruction pair and the same `dtype_scene_scale_floor_v1` policy. It tests downstream training randomness only; it is not an independent initialization repeat and does not separate camera from point optimization.

All four 7,000-step A800 runs completed once and continuously. Render benchmarks were not repeated; seed-42 contains the formal all-heldout-camera performance comparison.

| Scene | Arm | Status | PSNR | SSIM | LPIPS | GS | PLY bytes | train/eval s | queue GPUh |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| bonsai | pre BA | SUCCEEDED | 15.8198 | 0.4510 | 0.4897 | 744,652 | 175,739,349 | 1846.1 | 0.5133 |
| bonsai | post BA | SUCCEEDED | 17.4107 | 0.5030 | 0.3485 | 730,535 | 172,407,737 | 1846.1 | 0.5134 |
| room | pre BA | SUCCEEDED | 17.6889 | 0.5941 | 0.3553 | 674,071 | 159,082,233 | 1778.2 | 0.4945 |
| room | post BA | SUCCEEDED | 19.6931 | 0.6723 | 0.2671 | 547,504 | 129,212,421 | 1751.7 | 0.4873 |

Post minus pre:

- Bonsai: PSNR +1.5909 dB, SSIM +0.0520, LPIPS -0.1412, 14,117 fewer Gaussians and 3,331,612 fewer PLY bytes.
- Room: PSNR +2.0042 dB, SSIM +0.0782, LPIPS -0.0882, 126,567 fewer Gaussians and 29,869,812 fewer PLY bytes.

Both scenes retain the seed-42 effect direction, but two downstream seeds on the same snapshots remain a narrow development result rather than a broad generalization claim.
