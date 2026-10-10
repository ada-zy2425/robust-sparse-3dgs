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
- A persistent multi-GPU experiment queue with a single aggregate budget, GPU UUID binding, crash recovery and immutable task snapshots.
- Milestone evaluation, PLY/report export and deployment guardrails.
- Calibrated viewer camera presets plus a real browser/WebGL interaction check and explicit shutdown evidence.
- A bounded read-only multi-camera latency benchmark with per-camera and aggregate distribution reporting.
- A frozen-final protocol guard that rejects final-scene run IDs not declared before scoring.

The main reliability result is a controlled diagnosis: a frozen three-scene matrix exposed initializer-specific failures, and changing one factor to official upstream VGGT bundle adjustment restored 24-view initialization across all development scenes. A separately frozen three-scene final matrix then completed 3/3 official-BA arms while all three paired COLMAP arms failed full coverage. Quality and performance remained scene dependent, so this is evidence for a deployment policy and upstream integration—not a new BA algorithm.

The measured engineering tradeoff is an early-stopping control: it reduced optimization time and artifact size while improving PSNR/SSIM, with a disclosed LPIPS regression. It is schedule selection backed by one scene, not a universal optimization claim. No custom CUDA kernel or SOTA claim is made.
