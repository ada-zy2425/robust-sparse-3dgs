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

The main reliability result is a controlled system diagnosis: a frozen three-scene matrix exposed initializer-specific failures, and VGGT's complete official `use_ba` branch restored 24-view initialization across all development scenes. That switch changes tracking, point construction/filtering and camera-model handling in addition to calling bundle adjustment; it is not a one-factor BA result. A separately frozen three-scene final matrix then completed 3/3 official-branch arms while all three paired COLMAP arms produced partial registrations and failed the full-coverage gate. Quality and performance remained scene dependent, so this is evidence for a deployment policy and upstream integration—not a new BA algorithm.

The matched operation control is narrower. BA reduced mean reprojection error from 10.925 to 2.491 px on bonsai and 10.500 to 2.457 px on room. The pre-BA snapshots became non-finite at training step 100 for seeds 42 and 43, whereas all four post-BA runs completed. This is evidence that the joint camera/point BA operation changed trainability for these snapshots, but there are zero valid paired quality endpoints and no basis for attributing separate camera or point effects.

The measured engineering tradeoff is an early-stopping control: it reduced optimization time and artifact size, while quality moved in different directions (PSNR/SSIM improved and LPIPS regressed). It is not a quality-preserving speedup and is backed by one scene only. No custom CUDA kernel or SOTA claim is made.
