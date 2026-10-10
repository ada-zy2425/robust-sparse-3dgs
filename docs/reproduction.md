# Reproduction

These commands are the exercised single-node workflow. Dataset RGB, model weights and checkpoints are not committed. Use a licensed local dataset copy and the two project-local environments; do not replace system CUDA or PyTorch.

## 1. Paths and environment check

```bash
export PROJECT_ROOT=/absolute/path/to/robust-sparse-3dgs
export DATA_ROOT=/absolute/path/to/licensed/scene
export OUTPUT_ROOT=/absolute/path/to/bgs-runs
export GSPLAT_PY=/absolute/path/to/envs/gsplat/bin/python
export VGGT_PY=/absolute/path/to/envs/vggt/bin/python
export PYTHONPATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
df -h "$OUTPUT_ROOT"
"$GSPLAT_PY" -c 'import torch,gsplat; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),gsplat.__version__)'
"$VGGT_PY" -c 'import torch,pycolmap; print(torch.__version__,torch.version.cuda,pycolmap.__version__)'
"$GSPLAT_PY" -m unittest discover -s tests -v
"$GSPLAT_PY" -c 'from pathlib import Path; from bgs.train_adapter import patch_train_function; patch_train_function(Path("third_party/gsplat/examples/simple_trainer.py").read_text()); print("source contract OK")'
```

Copy the selected idle device's index and UUID from that same `nvidia-smi` output into `GPU_INDEX` and `GPU_UUID`; every queued task binds both values and refuses a changed mapping.

`upstream.lock.json` and `vggt_runtime.lock.json` contain the qualified source and dependency hashes. For a new checkout, populate the pinned source directories with `python scripts/setup_upstream.py --destination third_party`; obtain the VGGT checkpoint from its official source and verify the recorded hash/license. The gsplat and VGGT environments remain separate.

## 2. Prepare a calibrated development scene

The example assumes undistorted RGB and a matching `PINHOLE`/`SIMPLE_PINHOLE` reference model. The full-capture model supplies evaluation calibration metadata only; its points do not initialize training.

```bash
mkdir -p "$OUTPUT_ROOT/manifests" "$OUTPUT_ROOT/references" "$OUTPUT_ROOT/work"

"$GSPLAT_PY" -m bgs reference \
  --model "$DATA_ROOT/sparse/0" \
  --out "$OUTPUT_ROOT/references/scene.json" \
  --source "dataset release/version; full-capture SfM cameras; not geometry GT"

"$GSPLAT_PY" -m bgs prepare \
  --images "$DATA_ROOT/images" \
  --reference "$OUTPUT_ROOT/references/scene.json" \
  --out "$OUTPUT_ROOT/manifests/scene_dev_v24.json" \
  --train-views 24 --test-every 8 --scene-id scene_dev --partition development
```

Inspect the immutable manifest before launching: training and hold-out names must be disjoint, and every recorded image hash must still match. Held-out RGB must not appear in the staged initializer directory.

## 3. Run one scene

Choose an actually idle physical GPU from the environment check. The following is a COLMAP integration run; `--gpu-hours` and `--timeout` are finite campaign bounds, not expected costs.

```bash
"$GSPLAT_PY" -m bgs configure \
  --root "$OUTPUT_ROOT/work/scene_colmap_v1" \
  --manifest "$OUTPUT_ROOT/manifests/scene_dev_v24.json" \
  --run-id scene_dev_v24_colmap_s42_7k_v1 \
  --initializer colmap --gpu 0 --gpu-hours 1.5 \
  --steps 7000 --max-edge 1024 --timeout 5000 \
  --splat-numerics-policy dtype_scene_scale_floor_v1 \
  --gsplat-root "$PROJECT_ROOT/third_party/gsplat" \
  --out "$OUTPUT_ROOT/scene_dev_v24_colmap_s42_7k_v1.json"

"$GSPLAT_PY" -m bgs run --config "$OUTPUT_ROOT/scene_dev_v24_colmap_s42_7k_v1.json"
```

The command exits nonzero on partial registration, too few points, timeout, non-finite tensors, identity mismatch or budget exhaustion. `training/initialization_numerics.json` records the selected policy, scale floor, changed-point count and immediate post-initialization finite check. The floor is dtype- and scene-scale-relative; it is not `nan_to_num`, and a normal point cloud is unchanged. Do not remove a partial registration from the denominator: report both the full-coverage success rate and its registered/requested ratio.

## 4. Bounded batch execution

Create one aggregate ledger, add immutable task JSON files, then run one coordinator. Task files bind a physical GPU index and UUID, reserve disk and GPU hours atomically, and retain their own logs and receipts.

```bash
export QUEUE_STATE="$OUTPUT_ROOT/work/formal_stage_v1/queue.json"

"$GSPLAT_PY" -m bgs.queue init \
  --state "$QUEUE_STATE" --project-root "$PROJECT_ROOT" \
  --budget-gpu-hours 24 --max-parallel 4 \
  --disk-budget-gib 200 --min-free-gib 100

"$GSPLAT_PY" -m bgs.queue add --state "$QUEUE_STATE" --task /absolute/path/to/task.json
"$GSPLAT_PY" -m bgs.queue status --state "$QUEUE_STATE"
"$GSPLAT_PY" -m bgs.queue run --state "$QUEUE_STATE" --poll-s 1 --max-runtime-s 21600
```

On reconnection, run `status` first and inspect the recorded PID/token before starting a coordinator. A failed task is not retried implicitly; the same identity permits at most one explicit, reasoned retry. A changed config or source requires a new run ID.

The matched BA-operation development ablation is materialized with:

```bash
# Fill these from the same idle-GPU check used to choose indices 0--3.
export GPU0_UUID=GPU-REPLACE_FROM_NVIDIA_SMI
export GPU1_UUID=GPU-REPLACE_FROM_NVIDIA_SMI
export GPU2_UUID=GPU-REPLACE_FROM_NVIDIA_SMI
export GPU3_UUID=GPU-REPLACE_FROM_NVIDIA_SMI
"$GSPLAT_PY" scripts/build_ba_operation_ablation.py \
  --root "$PROJECT_ROOT" --revision floor_v1 --seed 42 \
  --reuse-pair-revision v1 --pair-seed 42 \
  --splat-numerics-policy dtype_scene_scale_floor_v1 \
  --max-gpu-hours 1.25 \
  --gpu-uuids "$GPU0_UUID" "$GPU1_UUID" "$GPU2_UUID" "$GPU3_UUID"
```

This exercised command reuses the already immutable seed-42 pre/post reconstruction pair and applies the same numerical policy to both arms. To create a new pair, omit `--reuse-pair-revision` and give a new revision; the builder first persists one official BA-branch reconstruction before `pycolmap.bundle_adjustment`, reloads isolated pre/post copies, applies BA only to the post copy, and uses identical downstream gsplat settings. The original feed-forward/no-BA branch is a different initialization pipeline and is reported separately.

The exercised training-randomness repeat used a new immutable identity and explicit remaining-budget bounds: `--revision floor_v3 --seed 43 --reuse-pair-revision v1 --pair-seed 42 --splat-numerics-policy dtype_scene_scale_floor_v1 --max-gpu-hours 0.7 --timeout-s 2400`. The unused `floor_v2` materialization was rejected before queue addition because its 4200-second timeout exceeded that reservation; it consumed zero GPU and was not overwritten.

## 5. Evaluation and performance

Every successful `bgs run` writes `summary.json`, `training/metrics.json`, per-view comparison canvases, milestone metrics, `model.ply`, checkpoint/progress receipts and `report.html`. Validate that the metric image names exactly equal the manifest hold-out names.

```bash
"$GSPLAT_PY" scripts/benchmark_render.py \
  --config "$OUTPUT_ROOT/scene_dev_v24_colmap_s42_7k_v1.json" \
  --out "$OUTPUT_ROOT/scene_dev_v24_colmap_s42_7k_v1.render.json" \
  --warmup 5 --repeats 30 --max-cameras 64
```

The benchmark is render-only: device-resident inputs, CUDA synchronization, every fixed hold-out camera, five warmups and 30 repeats. Report initialization, optimization, evaluation and end-to-end time separately from render latency. Report PLY bytes, Gaussian count and peak allocated memory with image resolution, GPU model and software revisions.

For a final partition, first freeze scene IDs, run IDs, code/config hashes and evaluation rules with `scripts/freeze_final_protocol.py`. Build only the declared matrix with `scripts/build_final_matrix.py`; `scripts/collect_final_results.py` refuses incomplete matrices and retains failed rows. Never use final-scene scores to change parameters or select endpoints.

## 6. Local-only viewer

```bash
"$GSPLAT_PY" -m bgs view --config "$OUTPUT_ROOT/scene_dev_v24_colmap_s42_7k_v1.json"
```

The viewer binds to `127.0.0.1`. If the GPU machine is remote, create an SSH local port forward from a second terminal, inspect a calibrated training preset and a genuine held-out camera preset, orbit the model, then stop both viewer and tunnel explicitly. HTTP 200 alone is not an interaction check.

## 7. Clean-source validation boundary

A clean source directory can reuse read-only dependency and dataset caches by pointing configs to their absolute paths; no RGB, weights or checkpoint is copied into the repository. The exercised sequence is:

```bash
export CLEAN_ROOT=/absolute/path/to/empty/clean-checkout
export REAL_MANIFEST=/absolute/path/to/an/existing/development-manifest.json
export COMPLETED_CONFIG=/absolute/path/to/an/existing/completed-development-config.json

mkdir -p "$CLEAN_ROOT"
git archive --format=tar HEAD | tar -xf - -C "$CLEAN_ROOT"
ln -s "$PROJECT_ROOT/third_party" "$CLEAN_ROOT/third_party"
cd "$CLEAN_ROOT"
PYTHONPATH="$CLEAN_ROOT" "$GSPLAT_PY" -m unittest discover -s tests -v
PYTHONPATH="$CLEAN_ROOT" "$GSPLAT_PY" -c \
  'from pathlib import Path; from bgs.train_adapter import patch_train_function; patch_train_function(Path("third_party/gsplat/examples/simple_trainer.py").read_text()); print("source contract OK")'
PYTHONPATH="$CLEAN_ROOT" "$GSPLAT_PY" -c \
  'import sys; from bgs.core import read_json; from bgs.data import verify_inputs; verify_inputs(read_json(sys.argv[1])); print("manifest OK")' \
  "$REAL_MANIFEST"

PYTHONPATH="$CLEAN_ROOT" "$GSPLAT_PY" scripts/build_render_benchmark_task.py \
  --project-root "$PROJECT_ROOT" --source-root "$CLEAN_ROOT" \
  --config "$COMPLETED_CONFIG" --gpu "$GPU_INDEX:$GPU_UUID" \
  --out-dir "$OUTPUT_ROOT/clean-validation-tasks" --revision 99
"$GSPLAT_PY" -m bgs.queue add --state "$QUEUE_STATE" \
  --task "$OUTPUT_ROOT/clean-validation-tasks/bench_RUN_ID_mc_v99.json"
"$GSPLAT_PY" -m bgs.queue run --state "$QUEUE_STATE" --poll-s 1 --max-runtime-s 1800
```

The clean tree needs access to the pinned `third_party/gsplat` checkout for the source-contract line; in the verified deployment it was linked read-only from the qualified dependency cache. Record the clean commit, commands and exit codes. This validates code/document reproducibility while keeping historical run identities immutable; it does not pretend that an existing checkpoint was retrained.

This sequence was exercised from clean commit `c569b12`: 41/41 CPU tests, the pinned trainer contract and a real 24-view room manifest passed. The queued read-only GPU check then rendered all 39 hold-out cameras for 1,170 synchronized samples from the clean source tree at 1.837 ms mean / 2.230 ms p95, consuming 0.005094 GPU-hours. The sanitized command/exit/hash receipt is [clean reproduction validation v1](../reports/clean_reproduction_validation_v1.json).

The numerical-fix delivery was independently re-exercised from clean commit `3a4025e`: 47/47 tests, compileall, the pinned trainer contract and the real room manifest passed. A queued read-only render then loaded the new 668,402-Gaussian room pre-BA checkpoint from the clean source tree and rendered all 39 holdouts / 1,170 samples at 2.018 ms mean / 2.479 ms p95 for 0.005442 GPU-hours. This is checkpoint render reproduction, not from-scratch training reproduction; see [clean reproduction validation floor v2](../reports/clean_reproduction_validation_floor_v2.json).

The sanitized development CSV and performance JSON under `reports/` are examples of the expected receipts. Mip-NeRF 360 RGB, derived comparison panels/video, PLYs and VGGT weights remain private because redistribution/commercial terms are not established.
