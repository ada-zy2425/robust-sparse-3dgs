# Architecture

The workflow has four small layers:

1. **Immutable inputs** — manifests hash RGB and calibration, enforce disjoint train/hold-out sets, and materialize only training RGB for the initializer.
2. **Pinned adapters** — COLMAP or VGGT produces a COLMAP model; the adapter verifies full training-view coverage and patches only the exact pinned gsplat training function.
3. **Bounded execution** — every stage has an immutable fingerprint, timeout, retry cap, expected artifacts and GPU-hour receipt. A persistent queue makes one atomic reservation across concurrent single-GPU experiments.
4. **Independent evaluation** — training cameras fit Sim(3); held-out calibrated cameras render only after training. Metrics, PLY, checkpoints, timing and peak memory are written to the run directory.

The private viewer loads a workflow checkpoint and binds to localhost. It is a presentation surface, not an evaluation substitute.
