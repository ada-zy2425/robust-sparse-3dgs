#!/usr/bin/env python3
"""Build the predeclared paired COLMAP/official-BA final matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

from bgs.core import WorkflowError, atomic_json, file_hash, read_json
from bgs.data import validate_manifest, verify_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--scene", action="append", required=True,
                        help="scene_id:24-view-final-manifest")
    parser.add_argument("--gpu", action="append", required=True, help="INDEX:GPU-UUID")
    parser.add_argument("--dinov2-root", required=True)
    parser.add_argument("--max-gpu-hours", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=int, default=5400)
    parser.add_argument("--initialization-reserve-s", type=int, default=900)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    protocol_path = Path(args.protocol).resolve()
    protocol = read_json(protocol_path)
    if protocol.get("schema") != 1 or protocol.get("status") != "FROZEN_BEFORE_SCORING":
        raise WorkflowError("Final protocol is not frozen")
    if (args.initialization_reserve_s + args.timeout_s + 5) / 3600 > args.max_gpu_hours:
        raise WorkflowError("Initialization plus training cleanup exceeds task reservation")

    gpus = []
    for item in args.gpu:
        index, separator, uuid = item.partition(":")
        if not separator or not index.isdigit() or not uuid.startswith("GPU-"):
            raise WorkflowError(f"Invalid GPU mapping: {item}")
        gpus.append((int(index), uuid))
    if not 1 <= len(gpus) <= 4 or len({uuid for _, uuid in gpus}) != len(gpus):
        raise WorkflowError("Provide one to four unique GPU mappings")

    manifests = {}
    for value in args.scene:
        scene_id, separator, manifest_value = value.partition(":")
        if not separator or scene_id in manifests:
            raise WorkflowError(f"Invalid or duplicate scene mapping: {value}")
        manifest_path = Path(manifest_value).resolve()
        manifest = validate_manifest(read_json(manifest_path))
        verify_inputs(manifest)
        if (manifest["partition"] != "final" or manifest["scene_id"] != scene_id
                or len(manifest["train"]) != 24
                or Path(manifest["final_protocol"]).resolve() != protocol_path):
            raise WorkflowError(f"Final manifest contract mismatch: {manifest_path}")
        manifests[scene_id] = manifest_path
    if set(manifests) != set(protocol.get("scene_ids", [])):
        raise WorkflowError("Scene mappings differ from frozen protocol")

    matrix_id = protocol["matrix_id"]
    config_dir = root / "configs" / matrix_id
    task_dir = root / "remote_tasks" / matrix_id
    if config_dir.exists() or task_dir.exists():
        raise WorkflowError("Refusing to overwrite immutable final matrix")
    dinov2_root = Path(args.dinov2_root).resolve()
    expected_ids = set(protocol.get("run_ids", []))
    planned = []
    assignment = 0
    for scene_id in protocol["scene_ids"]:
        for initializer, ba in (("colmap", False), ("vggt", True)):
            arm = "vggt_ba" if ba else "colmap"
            run_id = f"{scene_id}_final_v24_{arm}_s42_7k_v1"
            if run_id not in expected_ids:
                raise WorkflowError(f"Run was not predeclared: {run_id}")
            gpu_index, gpu_uuid = gpus[assignment % len(gpus)]
            assignment += 1
            planned.append((scene_id, initializer, ba, run_id, gpu_index, gpu_uuid))
    if {row[3] for row in planned} != expected_ids:
        raise WorkflowError("Frozen run matrix contains unsupported or missing entries")

    config_dir.mkdir(parents=True)
    task_dir.mkdir(parents=True)
    created = []
    for scene_id, initializer, ba, run_id, gpu_index, gpu_uuid in planned:
        manifest_path = manifests[scene_id]
        campaign = root / "work" / matrix_id / "campaigns" / run_id
        spec = {
            "run_id": run_id, "root": str(campaign), "manifest": str(manifest_path),
            "gsplat_root": str(root / "third_party" / "gsplat"),
            "python": str(root / "envs" / "gsplat" / "bin" / "python"),
            "initializer": initializer, "gpu": str(gpu_index), "gpu_hours": args.max_gpu_hours,
            "steps": 7000, "max_edge": 1024, "seed": 42, "noise_seed": 1729,
            "checkpoint_every": 100, "diagnostic_steps": [1000, 3000, 7000],
            "max_gaussians": 2000000, "timeout_s": args.timeout_s,
            "rotation_degrees": 0.0, "translation_fraction": 0.0, "pose_opt": False,
            "cache_images": False,
            "hypothesis": "Official BA improves 24-view reliability and held-out quality on untouched final scenes.",
            "independent_contribution": "Frozen paired evaluation and reliability diagnosis; BA is established upstream work.",
            "final_protocol": str(protocol_path), "final_protocol_sha256": file_hash(protocol_path),
        }
        if initializer == "colmap":
            spec.update(colmap_binary=str(root / "envs" / "gsplat" / "bin" / "colmap"),
                        colmap_use_gpu=False)
        else:
            spec.update(vggt_root=str(root / "third_party" / "vggt"),
                        vggt_python=str(root / "envs" / "vggt" / "bin" / "python3.10"),
                        vggt_ba=True, vggt_confidence=5.0, vggt_max_points=100000,
                        vggt_point_sampling="upstream_seeded_random_limit",
                        dinov2_root=str(dinov2_root))
        config_path = config_dir / f"{run_id}.json"
        atomic_json(config_path, spec)
        run = campaign / "runs" / run_id
        outputs = [run / "summary.json", run / "input" / "init_receipt.json", run / "training" / "model.ply"]
        outputs += [run / "training" / "milestones" / f"step_{step:08d}" / "metrics.json"
                    for step in (1000, 3000, 7000)]
        task = {
            "task_id": run_id, "kind": "bgs_run",
            "argv": [str(root / "envs" / "gsplat" / "bin" / "python"), "-m", "bgs", "run",
                     "--config", str(config_path)],
            "cwd": str(root),
            "env": {"PYTHONPATH": str(root), "TORCH_HOME": str(root / "cache" / "torch"),
                    "TMPDIR": str(root / "tmp" / matrix_id / run_id)},
            "dependencies": [], "config_path": str(config_path), "config_sha256": file_hash(config_path),
            "gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "gpu_count": 1,
            "max_gpu_hours": args.max_gpu_hours, "timeout_s": args.timeout_s,
            "disk_reservation_bytes": 10 * 2 ** 30, "expected_outputs": [str(path) for path in outputs],
            "priority": 200,
        }
        task_path = task_dir / f"{run_id}.json"
        atomic_json(task_path, task)
        created.append({"run_id": run_id, "config": str(config_path), "task": str(task_path)})
    receipt = {
        "schema": 1, "status": "MATERIALIZED_FROM_FROZEN_PROTOCOL",
        "protocol": str(protocol_path), "protocol_sha256": file_hash(protocol_path),
        "initialization_reserve_s": args.initialization_reserve_s, "tasks": created,
    }
    atomic_json(task_dir / "matrix_receipt.json", receipt)
    print(task_dir / "matrix_receipt.json")


if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
