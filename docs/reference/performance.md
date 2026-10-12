# Performance protocol and results

Cold setup, initialization, optimization, milestone evaluation and final evaluation are reported separately. `scripts/evaluation/benchmark_render.py` loads a completed checkpoint read-only, pre-places camera tensors on the GPU, visits every fixed held-out camera, performs five warmups per camera, synchronizes CUDA, then records 30 repetitions per camera. Its receipt contains resolution, Gaussian count, precision, GPU/driver/runtime versions, checkpoint/config hashes, latency distributions and peak memory. The queue refuses to overwrite evidence.

The early single-camera FPS number is integration evidence only. The formal development results below all ran on A800-SXM4-80GB devices and include every held-out camera; tensors were resident on the GPU, so browser, network, metric and file-writing time are excluded.

| Development checkpoint | Cameras / samples | Gaussians | Mean ms ↓ | p95 ms | FPS ↑ | Peak allocated MB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| bonsai VGGT no BA | 37 / 1,110 | 631,496 | 1.865 | 2.939 | 536.2 | 683.3 |
| bonsai VGGT official BA branch | 37 / 1,110 | 742,208 | 2.063 | 2.319 | 484.6 | 782.1 |
| room 6-view VGGT no BA | 39 / 1,170 | 297,925 | 1.488 | 1.826 | 672.0 | 340.5 |
| room 12-view VGGT no BA | 39 / 1,170 | 493,660 | 1.728 | 2.207 | 578.7 | 536.6 |
| room 24-view VGGT no BA | 39 / 1,170 | 728,207 | 2.071 | 2.517 | 482.8 | 784.2 |
| room 24-view VGGT official BA branch | 39 / 1,170 | 546,566 | 1.842 | 2.228 | 542.9 | 586.5 |
| garden 24-view COLMAP | 24 / 720 | 1,389,863 | 2.864 | 3.598 | 349.1 | 1,429.5 |
| garden 24-view VGGT official BA branch | 24 / 720 | 1,542,709 | 3.151 | 3.489 | 317.4 | 1,590.0 |

Gaussian count is a practical latency and memory driver, but the complete official BA branch changes initialization and the optimized model: it made room smaller/faster and bonsai and garden larger/slower. This is measured system behavior, not an isolated BA-operation or custom CUDA-kernel claim.

The garden BA checkpoint was independently repeated after its originally assigned A800 became idle. The two 720-sample means were 3.14795 and 3.15096 ms (0.096% relative difference), with identical 1,589,956,096-byte peak allocation. The table conservatively reports the slightly slower second receipt; both immutable receipts are retained.

The corrected matched BA-operation seed-42 controls applied the same `dtype_scene_scale_floor_v1` policy to both arms. Bonsai pre/post produced 751,476/738,354 Gaussians and 177,349,813/174,253,021-byte PLYs; their all-heldout-camera means were 2.073/2.072 ms over 37 cameras. Room pre/post produced 668,402/547,025 Gaussians and 157,744,349/129,099,377-byte PLYs; their means were 2.079/1.836 ms over 39 cameras. End-to-end time changed by +23.3 s on bonsai and -82.7 s on room. BA itself took 1.49 s and 4.93 s; these downstream differences are scene-dependent model-growth outcomes, not direct BA-call overhead or a custom kernel result.

The earlier South Building 3k control remains a separate quality/cost example. It used 220,795 instead of 641,484 Gaussians, reduced optimization time from 302.4 to 118.7 seconds and PLY size from 151.4 to 52.1 MB. Its mean render latency fell from 2.607 to 2.067 ms and peak memory from 707.4 to 267.1 MB, but LPIPS worsened from 0.3516 to 0.4633. It is therefore a selectable schedule tradeoff rather than a universal optimization win.

Machine-readable distributions and hashes are under `reports/performance/`. Viewer interaction has been browser-validated, but no browser end-to-end latency claim is made.

The separate controlled pose matrix did not add another all-camera latency benchmark because its frozen deliverables were quality, training cost, peak memory and model size. Across eight new 7k arms, training took 1,693.8–1,805.6 seconds, peak allocated memory was 3.53–6.07 GiB and final models contained 562,262–814,686 Gaussians. The matrix consumed 3.987086 GPUh. These costs accompany measured quality losses and are not presented as a performance optimization.
