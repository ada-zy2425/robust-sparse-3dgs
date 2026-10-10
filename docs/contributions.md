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

The first measured engineering improvement is an early-stopping quality/cost control: it reduced optimization time and artifact size while improving PSNR/SSIM, with a disclosed LPIPS regression. This is schedule selection backed by one development scene, not a novel algorithm or cross-scene claim. No custom CUDA kernel or SOTA claim is made.
