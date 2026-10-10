# Performance protocol

Report cold setup separately from initialization, optimization, milestone evaluation and final evaluation. For render-only timing, pre-place camera tensors on the GPU, warm up five times, synchronize, then time repeated renders. Record resolution, Gaussian count, SH degree, precision, GPU model, driver, PyTorch CUDA build, repetitions and peak allocated/reserved memory.

The early diagnostic's single-camera FPS remains an integration measurement. `scripts/benchmark_render.py` is the formal path: it loads a completed workflow checkpoint read-only, visits every fixed held-out camera, performs per-camera warmups and synchronized repetitions, and records raw-camera distributions plus aggregate mean/p50/p90/p95 latency. It is accepted only through the shared-budget queue and refuses to overwrite evidence.

On the first formal scene, the separately scheduled 3k control used 220,795 Gaussians instead of 641,484, reduced PLY size from 151.4 MB to 52.1 MB, and reduced optimization time from 302.4 s to 118.7 s. LPIPS worsened from 0.3516 to 0.4633, so this remains a selectable quality/cost tradeoff.

The queued multi-camera gate passed on one otherwise idle A800 GPU. Each result covers all 8 fixed held-out cameras, 5 warmups per camera and 30 synchronized repeats per camera (240 timed samples):

| Checkpoint | Gaussians | Mean ms ↓ | p50 ms | p95 ms | FPS ↑ | Peak allocated MB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| COLMAP 7k | 641,484 | 2.607 | 2.604 | 3.771 | 383.6 | 707.4 |
| COLMAP 3k | 220,795 | **2.067** | **2.172** | **2.532** | **483.7** | **267.1** |
| VGGT + official BA 7k | 684,451 | 2.752 | 2.778 | 3.767 | 363.4 | 747.6 |

Relative to COLMAP 7k, the 3k checkpoint reduced mean render latency by 20.7%, increased reciprocal-throughput FPS by 26.1%, and reduced peak allocated memory by 62.2%. These are render-only resident-tensor measurements, not viewer end-to-end latency, network performance, or a custom-kernel result. Machine-readable distributions and checkpoint/config hashes are under `reports/performance/`.
