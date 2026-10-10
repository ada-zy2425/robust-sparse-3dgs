# Acceptance gates

These gates separate a working engineering system from complete experimental evidence and from a validated improvement.

## ENGINEERING_READY — PASS (single-node A800 deployment)

- Immutable train-only staging, initialization, 7k training, genuine held-out scoring and PLY export completed.
- COLMAP and no-BA VGGT environments completed paired runs without replacing the system CUDA/PyTorch stack.
- The aggregate-budget queue dispatched independent GPU tasks, retained failed registration/dependency runs and enforced one reasoned retry.
- A localhost-only viewer was checked in a real WebGL browser: a calibrated default view rendered and a deterministic orbit drag changed the canvas; all viewer, browser and tunnel processes were then stopped.
- Exact-source trainer patching and CPU contracts pass. Shape-changing resume is deliberately excluded because qualification failed; the first-divergence diagnosis is documented separately and does not relax the qualification tolerance.
- A source-only archive of commit `c569b12` was extracted into a new directory and passed 41/41 tests, the pinned trainer source contract, real development-manifest hash verification and a queued 39-camera GPU render benchmark while reusing only the declared dependency/data/checkpoint caches.

## EVIDENCE_READY — PASS (scoped six-scene evidence)

- Development covers object-centric bonsai, indoor room and outdoor garden with immutable nested 6/12/24 inputs, 24–39 held-out images, all 18 COLMAP/feed-forward identities and three official BA-branch system comparisons. Thirteen initialization failures remain in the denominator.
- The final protocol and six run IDs were frozen before scores. Category-matched counter, kitchen and bicycle use 24 fixed views and 25–35 held-out images. The official BA branch completed 3/3; COLMAP registered only 20/24, 21/24 and 4/24 and failed 0/3 without retries.
- Reports contain 1k/3k/7k quality, registration, alignment, stage/end-to-end cost, memory, PLY hashes and complete failure rows. Formal performance spans every held-out camera for eight development checkpoints; an independent garden repeat differed by 0.096% in mean latency.
- A real browser/WebGL viewer interaction and a separate 39.97-second, three-scene held-out comparison video were inspected. RGB, video and PLYs remain private because redistribution terms are unresolved.

This is one seed per main arm and no geometry ground truth. It supports a portfolio-grade engineering result and scoped reliability conclusion, not strong statistical generalization.

## IMPROVEMENT_VALIDATED — PASS FOR RELIABILITY INTEGRATION, NOT ALGORITHM NOVELTY

The evidence-led intervention integrated and qualified VGGT's complete official `use_ba` initialization branch after camera residuals implicated pose quality. That branch changes the tracker, point construction/filtering and camera-model path as well as executing bundle adjustment, so the historical comparison is not a one-factor BA ablation. It produced full-coverage 24-view initialization on all three development scenes and all three untouched final scenes. On development, it improved all held-out metrics over the feed-forward branch on bonsai and room; garden gained reliability and LPIPS but lost PSNR/SSIM and became larger/slower versus COLMAP. On final, all COLMAP comparators produced only partial registrations and failed the fixed full-coverage gate, so no conditional quality-win claim is made.

The validated contribution is the qualified upstream branch integration and deployment policy, backed by frozen splits, registration ratios, complete failure accounting and external final scenes. BA itself is established upstream work. The new same-reconstruction control isolated the joint `pycolmap.bundle_adjustment` operation: on bonsai and room it reduced mean reprojection error from 10.925 to 2.491 px and from 10.500 to 2.457 px, and both post-BA arms trained successfully under seeds 42 and 43. Every matched pre-BA arm became non-finite at step 100, so no protocol-valid paired PSNR/SSIM/LPIPS, model-size or render-cost delta exists. This validates a trainability effect for these two snapshots, not an image-quality gain, and still cannot split camera from point optimization. The 3k/7k schedule result remains a separate one-scene quality/cost tradeoff, and no new filtering algorithm, robust resume, custom CUDA kernel or SOTA claim is accepted.

## Claim inventory

- Engineering delivery: complete for the pinned single-node workflow, finite queue, true held-out evaluation, PLY export, measured renderer and private localhost viewer.
- Upstream initialization reliability integration: verified for the complete official `use_ba` branch, with full-coverage success and registration ratios reported separately.
- Isolated BA operation contribution: trainability effect verified on the same-reconstruction bonsai/room controls across two downstream seeds; paired image-quality/model/render deltas are unavailable because every pre-BA arm failed the fixed finite-state gate at step 100. The result cannot divide camera from point optimization.
- Pose-perturbation specialization: not completed in this correction round.
- Independently new algorithmic robustness mechanism: not completed; no algorithm-gain claim is accepted.
- Shape-changing resume: diagnosed but not qualified; formal results continue to use uninterrupted runs.
