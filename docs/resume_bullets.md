# Evidence-backed resume bullets

- Built and GPU-validated a pinned COLMAP/VGGT-to-gsplat reconstruction harness with train/hold-out isolation, immutable receipts, PLY export, genuine held-out rendering and a browser-tested localhost viewer with calibrated camera presets on NVIDIA A800 GPUs.
- Implemented a crash-aware multi-GPU experiment queue with atomic aggregate GPU-hour reservations, GPU UUID binding, dependency tracking, immutable task snapshots and failure classification; covered critical contracts with CPU tests.
- Executed frozen three-development/three-final-scene sparse-view evaluations with every failed initialization retained; qualified upstream VGGT bundle adjustment completed 24/24 initialization on all six scenes, while paired final COLMAP arms reached 20/24, 21/24 and 4/24 and failed the fixed coverage gate.
- Benchmarked eight quality checkpoints over every held-out camera with synchronized latency distributions and memory receipts; separately measured a 3k/7k schedule tradeoff that cut optimization time 60.8% and PLY size 65.6% while explicitly retaining its LPIPS regression.
