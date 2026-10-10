# Corrected matched BA-operation ablation — seed 42

Both arms reuse the immutable seed-42 pre/post reconstruction pair and apply `dtype_scene_scale_floor_v1`. This seed therefore tests downstream training randomness only. The comparison estimates the joint camera-and-point `pycolmap.bundle_adjustment` operation; it does not separate camera from point optimization.

The shared guard clamped 4/22,192 pre-BA bonsai points and 8/36,226 pre-BA room points. It clamped zero post-BA points and changed no point count, color or spatial coverage. All four runs completed once, continuously, at 7,000 steps on A800-SXM4-80GB GPUs.

| Scene | Arm | Status | PSNR | SSIM | LPIPS | GS | PLY bytes | train/eval s | render mean/p95 ms | queue GPUh |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai | pre BA | SUCCEEDED | 15.7785 | 0.4511 | 0.4889 | 751,476 | 177,349,813 | 1769.5 | 2.073/2.311 | 0.4919 |
| bonsai | post BA | SUCCEEDED | 17.3846 | 0.5018 | 0.3512 | 738,354 | 174,253,021 | 1792.7 | 2.072/2.334 | 0.4985 |
| room | pre BA | SUCCEEDED | 17.6059 | 0.5929 | 0.3583 | 668,402 | 157,744,349 | 1825.4 | 2.079/2.530 | 0.5075 |
| room | post BA | SUCCEEDED | 19.7719 | 0.6735 | 0.2688 | 547,025 | 129,099,377 | 1742.6 | 1.836/2.221 | 0.4846 |

Post minus pre:

- Bonsai: PSNR +1.6061 dB, SSIM +0.0507, LPIPS -0.1377, 13,122 fewer Gaussians, 3,096,792 fewer PLY bytes and essentially unchanged mean render latency (-0.001 ms).
- Room: PSNR +2.1659 dB, SSIM +0.0806, LPIPS -0.0895, 121,377 fewer Gaussians, 28,644,972 fewer PLY bytes and 0.243 ms lower mean render latency.

BA changed mean reprojection error from 10.9249 to 2.4908 px on bonsai and 10.4997 to 2.4567 px on room. BA itself took 1.494 and 4.932 seconds. Downstream training and render differences include scene-dependent model growth and must not be labeled direct BA-call overhead.

Historical failed identities, configs, logs and reports remain unchanged. Their step-100 gate was the first check, while the exact-path audit found `-inf` scales during initialization; see [the root-cause receipt](../diagnostics/splat_initialization_nonfinite_v1.json).
