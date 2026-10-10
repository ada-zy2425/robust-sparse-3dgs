# Acceptance gates

These gates separate a working engineering system from complete experimental evidence and from a validated improvement.

## ENGINEERING_READY — PASS (single-node A800 deployment)

- Immutable train-only staging, initialization, 7k training, genuine held-out scoring and PLY export completed.
- COLMAP and no-BA VGGT environments completed paired runs without replacing the system CUDA/PyTorch stack.
- The aggregate-budget queue dispatched independent GPU tasks, retained failed registration/dependency runs and enforced one reasoned retry.
- A localhost-only viewer was checked in a real WebGL browser: a calibrated default view rendered and a deterministic orbit drag changed the canvas; all viewer, browser and tunnel processes were then stopped.
- Exact-source trainer patching and CPU contracts pass. Shape-changing resume is deliberately excluded because qualification failed.

## EVIDENCE_READY — NOT YET

The current scored denominator is one development scene plus preserved failed sparse selections from that capture. The frozen requirement is at least three development and three final scenes spanning object-centric, indoor and outdoor categories, with nested 6/12/24 inputs and at least ten held-out images where available. Official data acquisition and additional scene execution remain required.

## IMPROVEMENT_VALIDATED — PRELIMINARY, NOT CROSS-SCENE

On one development scene, the independently declared 3k schedule reduced optimization time and artifact size and improved PSNR/SSIM relative to 7k, while LPIPS regressed. The all-held-out-camera benchmark also measured 20.7% lower mean render latency, 26.1% higher FPS and 62.2% lower peak allocated memory. This validates a first-scene quality/cost tradeoff only. Cross-scene confirmation remains the claim gate.

VGGT bundle adjustment is a controlled upstream diagnostic, not a project algorithm. Any later filtering mechanism must beat point-count-matched random and confidence controls from one cached point pool before it can change this status.
