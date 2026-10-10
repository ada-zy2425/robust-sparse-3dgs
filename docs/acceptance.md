# Acceptance gates

These gates separate a working engineering system from complete experimental evidence and from a validated improvement.

## ENGINEERING_READY — PASS (single-node A800 deployment)

- Immutable train-only staging, initialization, 7k training, genuine held-out scoring and PLY export completed.
- COLMAP and no-BA VGGT environments completed paired runs without replacing the system CUDA/PyTorch stack.
- The aggregate-budget queue dispatched independent GPU tasks, retained failed registration/dependency runs and enforced one reasoned retry.
- A localhost-only viewer was checked in a real WebGL browser: a calibrated default view rendered and a deterministic orbit drag changed the canvas; all viewer, browser and tunnel processes were then stopped.
- Exact-source trainer patching and CPU contracts pass. Shape-changing resume is deliberately excluded because qualification failed.

## EVIDENCE_READY — PASS (scoped six-scene evidence)

- Development covers object-centric bonsai, indoor room and outdoor garden with immutable nested 6/12/24 inputs, 24–39 held-out images, all 18 COLMAP/no-BA identities and three one-factor official-BA controls. Thirteen initialization failures remain in the denominator.
- The final protocol and six run IDs were frozen before scores. Category-matched counter, kitchen and bicycle use 24 fixed views and 25–35 held-out images. Official BA completed 3/3; COLMAP registered only 20/24, 21/24 and 4/24 and failed 0/3 without retries.
- Reports contain 1k/3k/7k quality, registration, alignment, stage/end-to-end cost, memory, PLY hashes and complete failure rows. Formal performance spans every held-out camera for eight development checkpoints; an independent garden repeat differed by 0.096% in mean latency.
- A real browser/WebGL viewer interaction and a separate 39.97-second, three-scene held-out comparison video were inspected. RGB, video and PLYs remain private because redistribution terms are unresolved.

This is one seed per main arm and no geometry ground truth. It supports a portfolio-grade engineering result and scoped reliability conclusion, not strong statistical generalization.

## IMPROVEMENT_VALIDATED — PASS FOR RELIABILITY INTEGRATION, NOT ALGORITHM NOVELTY

The evidence-led intervention changed one factor from no-BA VGGT to official upstream bundle adjustment after camera residuals implicated pose quality. It produced full-coverage 24-view initialization on all three development scenes and all three untouched final scenes. On development, it improved all held-out metrics over no-BA on bonsai and room; garden gained reliability and LPIPS but lost PSNR/SSIM and became larger/slower versus COLMAP. On final, all COLMAP comparators failed coverage, so no conditional quality-win claim is made.

The validated contribution is the qualified upstream-BA integration and deployment policy, backed by frozen splits, complete failure accounting and external final scenes. BA itself is established upstream work. The 3k/7k schedule result remains a separate one-scene quality/cost tradeoff, and no new filtering algorithm, robust resume, custom CUDA kernel or SOTA claim is accepted.
