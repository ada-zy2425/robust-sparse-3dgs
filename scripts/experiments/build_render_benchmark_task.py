#!/usr/bin/env python3
"""Build one immutable, dependency-gated formal render benchmark task."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from bgs.core import WorkflowError, atomic_json, file_hash, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-root", help="Clean source tree to execute; defaults to project root")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", required=True, help="INDEX:GPU-UUID")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--max-gpu-hours", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    source_root = Path(args.source_root).resolve() if args.source_root else root
    benchmark_script=source_root / "scripts" / "evaluation" / "benchmark_render.py"
    if not benchmark_script.is_file():
        raise WorkflowError("Source root does not contain scripts/evaluation/benchmark_render.py")
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    index, separator, gpu_uuid = args.gpu.partition(":")
    if not separator or not index.isdigit() or not gpu_uuid.startswith("GPU-"):
        raise WorkflowError("--gpu must be INDEX:GPU-UUID")
    if args.revision < 1 or args.timeout_s / 3600 > args.max_gpu_hours:
        raise WorkflowError("Invalid benchmark revision or budget")

    run_id = config["run_id"]
    task_id = f"bench_{run_id}_mc_v{args.revision}"
    out_dir = Path(args.out_dir).resolve()
    task_path = out_dir / f"{task_id}.json"
    result_path = root / "work" / "performance_mipnerf360_dev_v1" / f"{task_id}.json"
    if task_path.exists() or result_path.exists():
        raise WorkflowError(f"Refusing to overwrite benchmark identity: {task_id}")
    task = {
        "task_id": task_id,
        "kind": "render_benchmark",
        "argv": [
            str(root / "envs" / "gsplat" / "bin" / "python"),
            str(benchmark_script),
            "--config", str(config_path),
            "--out", str(result_path),
            "--warmup", "5", "--repeats", "30", "--max-cameras", "64",
        ],
        "cwd": str(source_root),
        "env": {
            "PYTHONPATH": str(source_root),
            "TORCH_HOME": str(root / "cache" / "torch"),
            "TMPDIR": str(root / "tmp" / "performance_mipnerf360_dev_v1" / task_id),
        },
        "dependencies": [run_id],
        "config_path": str(config_path),
        "config_sha256": file_hash(config_path),
        "gpu_index": int(index),
        "gpu_uuid": gpu_uuid,
        "gpu_count": 1,
        "max_gpu_hours": args.max_gpu_hours,
        "timeout_s": args.timeout_s,
        "disk_reservation_bytes": 2 ** 30,
        "expected_outputs": [str(result_path)],
        "priority": 100,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(task_path, task)
    print(task_path)


if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
