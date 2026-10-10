# Evidence-backed resume bullets

- Built and GPU-validated a pinned COLMAP/VGGT-to-gsplat reconstruction harness with train/hold-out isolation, immutable receipts, PLY export, genuine held-out rendering and a browser-tested localhost viewer with calibrated camera presets on NVIDIA A800 GPUs.
- Implemented a crash-aware multi-GPU experiment queue with atomic aggregate GPU-hour reservations, GPU UUID binding, dependency tracking, immutable task snapshots and failure classification; covered critical contracts with CPU tests.
- Executed a frozen three-scene 6/12/24-view study with every failed initialization retained; diagnosed scene-dependent sparse registration and showed that upstream VGGT bundle adjustment restored 24-view initialization across all development scenes while producing mixed quality/cost outcomes.
- Benchmarked eight quality checkpoints over every held-out camera with synchronized latency distributions and memory receipts; separately measured a 3k/7k schedule tradeoff that cut optimization time 60.8% and PLY size 65.6% while explicitly retaining its LPIPS regression.
