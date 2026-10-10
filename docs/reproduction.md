# Reproduction

1. Clone the exact revisions in `upstream.lock.json` into separate project-local checkouts.
2. Create separate gsplat and VGGT environments; do not replace system CUDA/PyTorch.
3. Run `python -m unittest discover -s tests -v` and verify `patch_train_function` against the pinned trainer.
4. Export calibration metadata, create an immutable manifest, then generate a run config with `python -m bgs configure`.
5. Initialize the global queue with a finite aggregate GPU-hour and disk budget. Add immutable task specs that name physical GPU UUIDs.
6. Run one coordinator. Reconnection uses `python -m bgs.queue status --state "$QUEUE_STATE"` before any submission.

Dataset RGB, model weights, checkpoints and machine-specific configs are intentionally excluded. The VGGT original research checkpoint must be obtained from its official source and checked against the recorded hash and license.
