# Matched BA operation ablation

Both arms reload isolated copies of the same pre-BA official-branch reconstruction. Only the post arm executes `pycolmap.bundle_adjustment`; this does not separate camera from point optimization.

| Scene | Arm | Status | PSNR | SSIM | LPIPS | GS | PLY bytes | init s | train/eval s | render mean/p95 ms | queue GPUh |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai | pre_ba | FAILED | n/a | n/a | n/a | n/a | n/a | 1.0 | 59.5 | n/a | 0.0173 |
| bonsai | post_ba | SUCCEEDED | 17.4095 | 0.5015 | 0.3464 | 740972 | 174870869 | 130.7 | 1748.7 | 2.065/2.316 | 0.5225 |
| room | pre_ba | FAILED | n/a | n/a | n/a | n/a | n/a | 1.1 | 61.3 | n/a | 0.0178 |
| room | post_ba | SUCCEEDED | 19.7717 | 0.6735 | 0.2683 | 546696 | 129021733 | 151.0 | 1703.4 | 1.831/2.269 | 0.5155 |

Post minus pre:
- bonsai: unavailable (FAILED pre, SUCCEEDED post); no failed arm was assigned quality metrics.
- room: unavailable (FAILED pre, SUCCEEDED post); no failed arm was assigned quality metrics.

Both pre-BA arms failed the finite-state gate at step 100 on the original attempt and the one allowed exact retry. This validates no paired image-quality, model-size or render-cost delta.
