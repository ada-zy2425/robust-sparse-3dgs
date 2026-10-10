#!/usr/bin/env python3
"""Materialize immutable COLMAP/VGGT development tasks from a frozen registry."""
from __future__ import annotations

import argparse
from pathlib import Path

from bgs.core import WorkflowError, atomic_json, file_hash, read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",required=True)
    p.add_argument("--registry",required=True)
    p.add_argument("--gpu",action="append",required=True,help="INDEX:GPU-UUID; repeat up to four times")
    p.add_argument("--steps",type=int,default=7000)
    p.add_argument("--max-gpu-hours",type=float,default=1.5)
    p.add_argument("--timeout-s",type=int,default=5400)
    args=p.parse_args()
    root=Path(args.project_root).resolve();registry_path=Path(args.registry).resolve()
    registry=read_json(registry_path)
    if registry.get("schema")!=1 or registry.get("status")!="FROZEN_DEVELOPMENT":
        raise WorkflowError("Registry must be schema 1 and FROZEN_DEVELOPMENT")
    scenes=registry.get("development",[])
    if len(scenes)<3:raise WorkflowError("Need at least three frozen development scenes")
    gpus=[]
    for item in args.gpu:
        index,sep,uuid=item.partition(":")
        if not sep or not index.isdigit() or not uuid.startswith("GPU-"):
            raise WorkflowError(f"Invalid --gpu mapping: {item}")
        gpus.append((int(index),uuid))
    if not 1<=len(gpus)<=4 or len({u for _,u in gpus})!=len(gpus):
        raise WorkflowError("Provide one to four unique GPU UUID mappings")
    if args.steps!=7000:
        raise WorkflowError("Formal v1 matrix fixes a predeclared 7000-step total schedule")
    if args.timeout_s/3600>args.max_gpu_hours:
        raise WorkflowError("timeout exceeds task GPU-hour reservation")

    configs=root/"configs"/"mipnerf360_dev_v1"
    tasks=root/"remote_tasks"/"mipnerf360_dev_v1"
    configs.mkdir(parents=True,exist_ok=True);tasks.mkdir(parents=True,exist_ok=True)
    created=[];assignment=0
    for scene in scenes:
        scene_id=scene["scene_id"]
        manifests=scene.get("manifests",{})
        for views in (6,12,24):
            manifest=Path(manifests.get(str(views),"")).resolve()
            if not manifest.is_file():raise WorkflowError(f"Missing {scene_id} {views}-view manifest")
            record=read_json(manifest)
            if record.get("scene_id")!=scene_id or record.get("partition")!="development" or len(record.get("train",[]))!=views:
                raise WorkflowError(f"Manifest contract mismatch: {manifest}")
            for initializer in ("colmap","vggt"):
                run_id=f"{scene_id}_v{views:02d}_{initializer}_s42_7k_v1"
                config_path=configs/f"{run_id}.json";task_path=tasks/f"{run_id}.json"
                if config_path.exists() or task_path.exists():
                    raise WorkflowError(f"Refusing to overwrite immutable matrix entry: {run_id}")
                gpu_index,gpu_uuid=gpus[assignment%len(gpus)];assignment+=1
                campaign=root/"work"/"mipnerf360_dev_v1"/"campaigns"/run_id
                spec={"run_id":run_id,"root":str(campaign),"manifest":str(manifest),
                      "gsplat_root":str(root/"third_party"/"gsplat"),
                      "python":str(root/"envs"/"gsplat"/"bin"/"python"),
                      "initializer":initializer,"gpu":str(gpu_index),"gpu_hours":args.max_gpu_hours,
                      "steps":7000,"max_edge":1024,"seed":42,"noise_seed":1729,
                      "checkpoint_every":100,"diagnostic_steps":[1000,3000,7000],
                      "max_gaussians":2000000,"timeout_s":args.timeout_s,
                      "rotation_degrees":0.0,"translation_fraction":0.0,"pose_opt":False,
                      "cache_images":False,
                      "hypothesis":"Paired sparse-view baseline at fixed inputs, hardware, resolution, schedule and seed.",
                      "independent_contribution":"Reproducible integration and held-out evaluation; upstream initialization and gsplat are not new algorithms."}
                if initializer=="colmap":
                    spec.update(colmap_binary=str(root/"envs"/"gsplat"/"bin"/"colmap"),colmap_use_gpu=False)
                else:
                    spec.update(vggt_root=str(root/"third_party"/"vggt"),
                                vggt_python=str(root/"envs"/"vggt"/"bin"/"python3.10"),
                                vggt_ba=False,vggt_confidence=5.0,vggt_max_points=100000,
                                vggt_point_sampling="upstream_seeded_random_limit")
                atomic_json(config_path,spec)
                run=campaign/"runs"/run_id
                expected=[run/"summary.json",run/"input"/"init_receipt.json",run/"training"/"model.ply"]
                expected += [run/"training"/"milestones"/f"step_{step:08d}"/"metrics.json" for step in (1000,3000,7000)]
                task={"task_id":run_id,"kind":"bgs_run",
                      "argv":[str(root/"envs"/"gsplat"/"bin"/"python"),"-m","bgs","run","--config",str(config_path)],
                      "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                      "TMPDIR":str(root/"tmp"/"mipnerf360_dev_v1"/run_id)},"dependencies":[],
                      "config_path":str(config_path),"config_sha256":file_hash(config_path),
                      "gpu_index":gpu_index,"gpu_uuid":gpu_uuid,"gpu_count":1,
                      "max_gpu_hours":args.max_gpu_hours,"timeout_s":args.timeout_s,
                      "disk_reservation_bytes":10*2**30,"expected_outputs":[str(x) for x in expected],
                      "priority":20+views+(0 if initializer=="colmap" else 1)}
                atomic_json(task_path,task);created.append({"run_id":run_id,"config":str(config_path),"task":str(task_path)})
    receipt={"schema":1,"registry":str(registry_path),"registry_sha256":file_hash(registry_path),
             "steps":7000,"seed":42,"max_edge":1024,"tasks":created}
    receipt_path=tasks/"matrix_receipt.json"
    if receipt_path.exists():raise WorkflowError("Matrix receipt already exists")
    atomic_json(receipt_path,receipt)
    print(receipt_path)


if __name__=="__main__":
    try:main()
    except WorkflowError as e:raise SystemExit(f"BLOCKED: {e}")
