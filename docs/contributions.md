# Contributions and provenance

Directly reused:

- COLMAP reconstruction tools.
- The pinned gsplat trainer, rasterizer, strategy and viewer.
- The pinned VGGT model/demo and LightGlue dependency.
- Standard PSNR, SSIM, LPIPS and Sim(3) alignment techniques.

Implemented in this repository:

- Train-only immutable staging, calibrated held-out evaluation and split integrity checks.
- Exact-source AST adapter contract for bounded gsplat training.
- Full workflow checkpoints, strict resume qualification and preserved failure evidence.
- First-divergence instrumentation across checkpoint binding, scheduler/strategy/RNG state, sampling, forward, backward, optimizer update and densification decisions; production resume remains disabled because the fixed shape-changing qualification still fails.
- A persistent multi-GPU experiment queue with a single aggregate budget, GPU UUID binding, crash recovery and immutable task snapshots.
- Milestone evaluation, PLY/report export and deployment guardrails.
- Calibrated viewer camera presets plus a real browser/WebGL interaction check and explicit shutdown evidence.
- A bounded read-only multi-camera latency benchmark with per-camera and aggregate distribution reporting.
- A frozen-final protocol guard that rejects final-scene run IDs not declared before scoring.
- A same-reconstruction BA-operation harness that snapshots before the pinned upstream BA call, reloads non-aliasing pre/post copies, checks snapshot immutability and records geometry, timing and downstream failure/success evidence.
- Exact-path initialization diagnostics plus a versioned dtype/scene-scale-relative positive-scale guard, immediate finite-state reporting and every-step forward/backward/strategy probes. This is conventional numerical robustness, not a new reconstruction algorithm.

The main reliability result is a controlled system diagnosis: a frozen three-scene matrix exposed initializer-specific failures, and VGGT's complete official `use_ba` branch restored 24-view initialization across all development scenes. That switch changes tracking, point construction/filtering and camera-model handling in addition to calling bundle adjustment; it is not a one-factor BA result. A separately frozen three-scene final matrix then completed 3/3 official-branch arms while all three paired COLMAP arms produced partial registrations and failed the full-coverage gate. Quality and performance remained scene dependent, so this is evidence for a deployment policy and upstream integration—not a new BA algorithm.

The matched operation control is narrower. BA reduced mean reprojection error from 10.925 to 2.491 px on bonsai and 10.500 to 2.457 px on room. An exact replay corrected the historical attribution: the old step-100 failures were already non-finite at initialization because duplicate-point clusters produced zero KNN scales and `log(0)`. I compared deduplication with a scale-equivariant positive floor, selected the latter because it changed only 4/8 degenerate scales and no points or colors, and applied it identically to pre/post arms. Under seed 42, all four 7k runs completed; post minus pre was +1.606/+2.166 dB PSNR, +0.0507/+0.0806 SSIM and -0.1377/-0.0895 LPIPS on bonsai/room. Seed 43 retained the directions and only tests training randomness because it reuses the snapshots. This estimates the joint camera/point BA operation on two development snapshots and provides no basis for separate camera or point effects.

The measured engineering tradeoff is an early-stopping control: it reduced optimization time and artifact size, while quality moved in different directions (PSNR/SSIM improved and LPIPS regressed). It is not a quality-preserving speedup and is backed by one scene only. No custom CUDA kernel or SOTA claim is made.
