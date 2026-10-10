# Reproduction

1. Clone the exact revisions in `upstream.lock.json` into separate project-local checkouts.
2. Create separate gsplat and VGGT environments; do not replace system CUDA/PyTorch.
3. Run `python -m unittest discover -s tests -v` and verify `patch_train_function` against the pinned trainer.
4. Export calibration metadata, create an immutable manifest, then generate a run config with `python -m bgs configure`.
5. Initialize the global queue with a finite aggregate GPU-hour and disk budget. Add immutable task specs that name physical GPU UUIDs.
6. Run one coordinator. Reconnection uses `python -m bgs.queue status --state "$QUEUE_STATE"` before any submission.

For a final partition, freeze the scene IDs and run IDs before scoring with `scripts/freeze_final_protocol.py`. Build manifests that reference that receipt, materialize the paired matrix with `scripts/build_final_matrix.py`, and submit those immutable task files to the existing aggregate-budget queue. After every predeclared task is terminal, use `scripts/collect_final_results.py`; it refuses incomplete matrices and includes failed runs in the output rows.

The development CSV and performance JSON under `reports/` are sanitized, machine-readable examples. Each latency receipt represents every held-out camera with five warmups and 30 synchronized repeats per camera, not a selected easy view.

Dataset RGB, derived comparison panels/video, model weights, checkpoints and machine-specific configs are intentionally excluded. The VGGT original research checkpoint must be obtained from its official source and checked against the recorded hash and non-commercial license. Keep viewers bound to localhost and stop the browser, tunnel and viewer after validation.
