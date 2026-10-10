# Frozen Mip-NeRF 360 final evaluation

This report was produced after all six predeclared tasks became terminal. The source final protocol was frozen before scoring with SHA256 `face3d6a803c89a27539cf3c3d8a567494f22c865b5c813bf13855b7e7961d58`; the subsequent scene registry SHA256 is `ec6747a8becd2e6680f2eee9f684f25603d65ff4551b9b7ce9add66af4af0a1e`. No final result changed a view, initializer setting, schedule, seed or retry decision.

Each arm uses 24 fixed training views, max edge 1024, seed 42 and the predeclared 7k schedule. Fixed holdouts contain 30 counter, 35 kitchen and 25 bicycle images. Held-out RGB is excluded from initialization and training; reference poses are calibration metadata, and Sim(3) alignment uses training cameras only. All tasks ran continuously on A800-SXM4-80GB devices because shape-changing resume remains unqualified.

## Complete final denominator

| Scene / category | COLMAP coverage | COLMAP status | VGGT + official BA | BA init pts | BA 7k PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---|---|---:|---:|---:|---:|
| counter / object-centric | 20/24 | failed | 24/24 valid | 34,417 | 19.8000 | 0.6141 | 0.2639 |
| kitchen / indoor | 21/24 | failed | 24/24 valid | 39,364 | 18.1994 | 0.4275 | 0.2365 |
| bicycle / outdoor | 4/24 | failed | 24/24 valid | 13,783 | 16.9353 | 0.2381 | 0.4687 |

All three COLMAP arms failed the immutable full-coverage gate and therefore have no quality score; they are not silently removed or averaged. The missing counts were 4, 3 and 20 images respectively. Each failed after one attempt, and the logs show reproducible partial registration rather than a transient GPU, dependency or disk error, so no retry was used.

Official upstream BA completed 3/3 final scenes versus COLMAP's 0/3. This externally confirms the development-stage reliability diagnosis, where 24-view official BA completed 3/3 scenes, COLMAP completed 1/3 and no-BA VGGT completed 2/3. The supported claim is that the qualified official-BA path was materially more reliable under these fixed sparse selections. BA is established upstream work; the project's contribution is the train-only adapter, controlled comparison, qualification and complete failure accounting—not a new BA algorithm.

## Predeclared quality trajectory

| Scene | 1k PSNR / SSIM / LPIPS | 3k PSNR / SSIM / LPIPS | 7k PSNR / SSIM / LPIPS | Final GS |
|---|---|---|---|---:|
| counter | 19.304 / 0.6305 / 0.4706 | 19.800 / 0.6238 / 0.3028 | 19.800 / 0.6141 / 0.2639 | 520,936 |
| kitchen | 18.093 / 0.4582 / 0.4392 | 17.959 / 0.4267 / 0.2811 | 18.199 / 0.4275 / 0.2365 | 656,347 |
| bicycle | 17.586 / 0.2976 / 0.7754 | 17.182 / 0.2623 / 0.5517 | 16.935 / 0.2381 / 0.4687 | 1,738,960 |

LPIPS improved from 1k to 7k on every final scene, while PSNR/SSIM were flat or worse on parts of the trajectory. The fixed 7k endpoint is reported as planned; this observation does not reopen final-scene schedule selection. It also reinforces that equal step counts do not imply equal compute or model size.

## Cost and artifacts

| Scene, official BA | Optimization s | End-to-end s | GPU h | Peak allocated GB | PLY MB | PLY SHA256 prefix |
|---|---:|---:|---:|---:|---:|---|
| counter | 1,531.3 | 1,803.8 | 0.5013 | 4.85 | 122.9 | `06545f802e46` |
| kitchen | 1,561.9 | 1,838.2 | 0.5108 | 6.01 | 154.9 | `193a9a278324` |
| bicycle | 3,589.5 | 4,067.7 | 1.1302 | 14.89 | 410.4 | `1edb491c6172` |

The six final attempts consumed 2.3302 GPU-hours. The shared queue charged 10.0809/24 GPU-hours for the complete new stage at collection time, including development experiments and formal performance tasks; no reservation remained. Final output hashes, complete 37-column rows and full failure strings are in `mipnerf360_final_v1.json` and `.csv`.

Manual inspection of one held-out canvas per successful final scene found recognizable, aligned predictions rather than black frames or gross camera errors. Counter retained errors around reflective metal and fine detail; kitchen around foliage and object edges; bicycle showed substantial foreground grass blur/floater artifacts despite a recognizable bicycle. These limitations agree with the metrics and are not presented as geometry accuracy.

## Scope and limits

- The final result confirms a reliability policy on three category-matched scenes and one seed, not broad statistical generalization.
- Conditional quality averages cannot compare BA with failed COLMAP arms. Development scenes provide the only quality comparisons, including the mixed garden result.
- There is no ground-truth geometry; PSNR/SSIM/LPIPS are image-space held-out metrics.
- Possible VGGT pretraining overlap with these benchmark scenes is unknown.
- Dataset RGB, derived canvases, video and PLYs remain private; only sanitized metrics and hashes are committed.
- Production resume, public deployment, commercial VGGT rights, custom CUDA work and SOTA claims remain explicitly out of scope.
