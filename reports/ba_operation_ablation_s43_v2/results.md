# Matched BA operation ablation

Both arms reuse the immutable seed-42 pre/post reconstruction pair; this seed tests downstream training randomness only. The comparison does not separate camera from point optimization.

| Scene | Arm | Status | PSNR | SSIM | LPIPS | GS | PLY bytes | init s | train/eval s | render mean/p95 ms | queue GPUh |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai | pre_ba | FAILED | n/a | n/a | n/a | n/a | n/a | 0.9 | 31.7 | n/a | 0.0093 |
| bonsai | post_ba | SUCCEEDED | 17.4630 | 0.5043 | 0.3455 | 733150 | 173024877 | 1.0 | 1735.7 | n/a | 0.4826 |
| room | pre_ba | FAILED | n/a | n/a | n/a | n/a | n/a | 1.0 | 31.1 | n/a | 0.0091 |
| room | post_ba | SUCCEEDED | 19.8260 | 0.6718 | 0.2681 | 555008 | 130983365 | 1.0 | 1677.8 | n/a | 0.4666 |

Post minus pre:
- bonsai: unavailable (FAILED pre, SUCCEEDED post); no failed arm was assigned quality metrics.
- room: unavailable (FAILED pre, SUCCEEDED post); no failed arm was assigned quality metrics.

Both pre-BA arms failed the finite-state gate at step 100. This validates no paired image-quality, model-size or render-cost delta.
