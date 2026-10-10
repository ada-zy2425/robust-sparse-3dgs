#!/usr/bin/env python3
"""Build historical official-BA branch comparisons from feed-forward configs.

The upstream switch changes several initialization operations and is not an
isolated BA ablation. Existing generated identities and receipts stay intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bgs.core import WorkflowError, atomic_json, file_hash, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--base-config", action="append", required=True)
    parser.add_argument("--gpu", action="append", required=True, help="INDEX:GPU-UUID")
    parser.add_argument("--dinov2-root", required=True)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--max-gpu-hours", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=int, default=5400)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    if len(args.base_config) != 3 or len(args.gpu) != 3:
        raise WorkflowError("Official use_ba branch matrix requires exactly three base configs and GPUs")
    if args.revision < 1:
        raise WorkflowError("revision must be positive")
    if (args.timeout_s + 905) / 3600 > args.max_gpu_hours:
        raise WorkflowError("Budget must cover initialization allowance and training cleanup")
    mappings = []
    for value in args.gpu:
        index, separator, gpu_uuid = value.partition(":")
        if not separator or not index.isdigit() or not gpu_uuid.startswith("GPU-"):
            raise WorkflowError(f"Invalid GPU mapping: {value}")
        mappings.append((int(index), gpu_uuid))

    dinov2_root = Path(args.dinov2_root).resolve()
    tag = f"mipnerf360_ba_v{args.revision}"
    config_dir = root / "configs" / tag
    task_dir = root / "remote_tasks" / tag
    if config_dir.exists() or task_dir.exists():
        raise WorkflowError("Refusing to overwrite immutable use_ba branch matrix")
    config_dir.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    created = []
    for base_value, (gpu_index, gpu_uuid) in zip(args.base_config, mappings):
        base_path = Path(base_value).resolve()
        base = read_json(base_path)
        manifest = read_json(base["manifest"])
        if (
            base.get("initializer") != "vggt"
            or base.get("vggt_ba") is not False
            or base.get("steps") != 7000
            or base.get("seed") != 42
            or base.get("max_edge") != 1024
            or len(manifest.get("train", [])) != 24
        ):
            raise WorkflowError(f"Base is not a fixed 24-view no-BA formal config: {base_path}")
        source_suffix = "_vggt_s42_7k_v3"
        if not base["run_id"].endswith(source_suffix):
            raise WorkflowError(f"Unexpected base identity: {base['run_id']}")
        run_id = base["run_id"][:-len(source_suffix)] + f"_vggt_ba_s42_7k_v{args.revision}"
        campaign = root / "work" / tag / "campaigns" / run_id
        config = dict(base)
        config.update(
            run_id=run_id,
            root=str(campaign),
            gpu=str(gpu_index),
            gpu_hours=args.max_gpu_hours,
            timeout_s=args.timeout_s,
            vggt_ba=True,
            dinov2_root=str(dinov2_root),
            hypothesis="The complete official use_ba initialization branch improves reliability at fixed inputs and training schedule.",
            independent_contribution="Controlled diagnosis and offline qualification; bundle adjustment is established upstream work.",
            control_of=base["run_id"],
            control_of_sha256=file_hash(base_path),
        )
        config_path = config_dir / f"{run_id}.json"
        atomic_json(config_path, config)
        run = campaign / "runs" / run_id
        expected = [run / "summary.json", run / "input" / "init_receipt.json", run / "training" / "model.ply"]
        expected += [run / "training" / "milestones" / f"step_{step:08d}" / "metrics.json" for step in (1000, 3000, 7000)]
        task = {
            "task_id": run_id,
            "kind": "bgs_run",
            "argv": [str(root / "envs" / "gsplat" / "bin" / "python"), "-m", "bgs", "run", "--config", str(config_path)],
            "cwd": str(root),
            "env": {
                "PYTHONPATH": str(root),
                "TORCH_HOME": str(root / "cache" / "torch"),
                "TMPDIR": str(root / "tmp" / tag / run_id),
            },
            "dependencies": [],
            "config_path": str(config_path),
            "config_sha256": file_hash(config_path),
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "gpu_count": 1,
            "max_gpu_hours": args.max_gpu_hours,
            "timeout_s": args.timeout_s,
            "disk_reservation_bytes": 10 * 2 ** 30,
            "expected_outputs": [str(path) for path in expected],
            "priority": 50,
        }
        task_path = task_dir / f"{run_id}.json"
        atomic_json(task_path, task)
        created.append({"run_id": run_id, "base_config": str(base_path), "base_sha256": file_hash(base_path),
                        "config": str(config_path), "task": str(task_path)})
    atomic_json(task_dir / "matrix_receipt.json", {
        "schema": 1,
        "revision": args.revision,
        "factor": "complete official feed-forward branch -> complete official use_ba branch",
        "fixed": ["manifest", "24 views", "7000 steps", "max_edge=1024", "seed=42", "renderer", "GPU family"],
        "inactive_in_ba_branch": ["conf_thres_value", "100000-point random truncation"],
        "scope": "official upstream branch-level reliability comparison; not an isolated BA operation or a new algorithm",
        "created": created,
    })
    print(task_dir / "matrix_receipt.json")


if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
