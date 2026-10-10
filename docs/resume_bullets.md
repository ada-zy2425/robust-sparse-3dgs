# Evidence-backed resume bullets

- Built and GPU-validated a pinned COLMAP/VGGT-to-gsplat reconstruction harness with train/hold-out isolation, immutable receipts, PLY export, genuine held-out rendering and a browser-tested localhost viewer with calibrated camera presets on NVIDIA A800 GPUs.
- Implemented a crash-aware multi-GPU experiment queue with atomic aggregate GPU-hour reservations, GPU UUID binding, dependency tracking, immutable task snapshots and failure classification; covered critical contracts with CPU tests.
- Diagnosed sparse registration, VGGT camera error and shape-changing resume failures from preserved evidence; on one development scene, an evidence-led 3k schedule cut optimization time 60.8% and PLY size 65.6% versus 7k while improving PSNR/SSIM, with the LPIPS regression explicitly retained.
