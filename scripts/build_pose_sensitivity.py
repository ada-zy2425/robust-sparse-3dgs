#!/usr/bin/env python3
"""Freeze the two-scene controlled pose-sensitivity matrix."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from bgs.core import WorkflowError,atomic_json,file_hash,read_json
from bgs.splat_numerics import FLOOR_POLICY


CONDITIONS=(
    ("r05",.5,0.),
    ("r10",1.,0.),
    ("t005",0.,.005),
    ("t010",0.,.01),
)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--revision",default="v1")
    p.add_argument("--gpu-uuids",nargs=4,required=True,metavar=("GPU0","GPU1","GPU2","GPU3"))
    p.add_argument("--round-start-charged",type=float,required=True)
    p.add_argument("--round-cap-gpu-hours",type=float,default=5.)
    p.add_argument("--max-gpu-hours",type=float,default=.6)
    p.add_argument("--timeout-s",type=int,default=2100)
    args=p.parse_args()
    if any(not x.startswith("GPU-") for x in args.gpu_uuids):raise WorkflowError("Explicit GPU UUIDs required")
    if args.timeout_s<=0 or args.timeout_s/3600>args.max_gpu_hours:raise WorkflowError("Timeout must fit per-task budget")
    if len(CONDITIONS)*2*args.max_gpu_hours>args.round_cap_gpu_hours+1e-12:
        raise WorkflowError("Worst-case task reservations exceed round cap")
    root=Path(args.root).resolve();protocol=root/"protocols"/"pose_sensitivity_v1.json"
    if not protocol.is_file():raise WorkflowError("Frozen pose_sensitivity_v1 protocol missing")
    queue_path=root/"work"/"formal_stage_v1"/"queue.json";queue=read_json(queue_path)
    if abs(queue["charged_gpu_hours"]-args.round_start_charged)>1e-9 or queue["reserved_gpu_hours"]!=0:
        raise WorkflowError("Queue charge/reservation changed since round budget audit")
    if any(x["status"] in ("PENDING","RESERVED","RUNNING") for x in queue["tasks"].values()):
        raise WorkflowError("Queue is not idle")
    if queue["max_parallel"]>4:raise WorkflowError("Queue parallelism exceeds protocol limit")
    worst_reservation=len(CONDITIONS)*2*args.max_gpu_hours
    if queue["charged_gpu_hours"]+worst_reservation>min(queue["budget_gpu_hours"],args.round_start_charged+args.round_cap_gpu_hours)+1e-12:
        raise WorkflowError("Frozen matrix does not fit both the total ledger and round cap")

    stage=root/"work"/f"pose_sensitivity_{args.revision}"
    config_dir=root/"configs"/f"pose_sensitivity_{args.revision}"
    task_dir=root/"remote_tasks"/f"pose_sensitivity_{args.revision}"
    if stage.exists() or config_dir.exists() or task_dir.exists():raise WorkflowError("Refusing to overwrite pose-sensitivity identity")
    config_dir.mkdir(parents=True);task_dir.mkdir(parents=True);rows=[];zero_references={}
    for scene_index,scene in enumerate(("bonsai","room")):
        zero_id=f"m360_{scene}_dev_v24_vggt_ba_post_s42_7k_floor_v1"
        zero_config_path=root/"configs"/"ba_operation_ablation_floor_v1"/f"{zero_id}.json"
        base=read_json(zero_config_path);zero_run=Path(base["root"])/"runs"/zero_id
        required={"steps":7000,"max_edge":1024,"seed":42,"noise_seed":1729,"rotation_degrees":0.,
                  "translation_fraction":0.,"pose_opt":False,"splat_numerics_policy":FLOOR_POLICY,
                  "vggt_ba_ablation_arm":"post_ba"}
        mismatch={k:(base.get(k),v) for k,v in required.items() if base.get(k)!=v}
        if mismatch:raise WorkflowError(f"Zero-reference contract mismatch for {scene}: {mismatch}")
        zero_outputs={
            "experiment":zero_run/"experiment.json","init_receipt":zero_run/"input"/"init_receipt.json",
            "alignment":zero_run/"training"/"alignment.json","metrics":zero_run/"training"/"metrics.json",
            "model":zero_run/"training"/"model.ply","complete":zero_run/"training"/"complete.json",
        }
        if any(not p.is_file() for p in zero_outputs.values()):raise WorkflowError(f"Incomplete zero reference for {scene}")
        experiment=read_json(zero_outputs["experiment"]);init=read_json(zero_outputs["init_receipt"])
        zero_references[scene]={"run_id":zero_id,"config":str(zero_config_path),
            "config_sha256":file_hash(zero_config_path),"outputs":{k:str(v) for k,v in zero_outputs.items()},
            "output_sha256":{k:file_hash(v) for k,v in zero_outputs.items()},
            "fingerprint_hash":experiment["fingerprint_hash"],"source_hashes":experiment["source"],
            "initialization_files":init["files"],"pair_root":base["vggt_ba_pair_root"]}
        for condition_index,(condition,rotation,translation) in enumerate(CONDITIONS):
            gpu=condition_index;run_id=f"m360_{scene}_dev_v24_vggt_ba_post_s42_7k_pose_{condition}_{args.revision}"
            config={**base,"run_id":run_id,"root":str(stage/"campaigns"/run_id),"gpu":str(gpu),
                "gpu_hours":args.max_gpu_hours,"timeout_s":args.timeout_s,
                "rotation_degrees":rotation,"translation_fraction":translation,
                "vggt_ba_pair_generate":False,"pose_sensitivity_protocol":"pose_sensitivity_v1",
                "pose_sensitivity_condition":condition,"zero_reference_run_id":zero_id,
                "zero_reference_config_sha256":file_hash(zero_config_path),
                "hypothesis":"Added training-pose error changes genuine held-out rendering quality at fixed initialization and schedule.",
                "independent_contribution":"Controlled sensitivity integration and measurement; not a new pose or BA algorithm.",
                "scope":"Development-only estimated-pose perturbation; no geometry ground truth or broad robustness claim."}
            config.pop("control_of",None);config.pop("control_of_sha256",None)
            config_path=config_dir/f"{run_id}.json";atomic_json(config_path,config)
            run=Path(config["root"])/"runs"/run_id
            expected=[run/"experiment.json",run/"summary.json",run/"report.html",run/"input"/"init_receipt.json",
                run/"training"/"alignment.json",run/"training"/"initialization_numerics.json",
                run/"training"/"metrics.json",run/"training"/"model.ply",run/"training"/"complete.json"]
            task={"task_id":run_id,"kind":"bgs_run",
                "argv":[config["python"],"-m","bgs","run","--config",str(config_path)],
                "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                    "TMPDIR":str(root/"tmp"/f"pose_sensitivity_{args.revision}"/run_id)},
                "dependencies":[],"config_path":str(config_path),"config_sha256":file_hash(config_path),
                "gpu_index":gpu,"gpu_uuid":args.gpu_uuids[gpu],"gpu_count":1,
                "max_gpu_hours":args.max_gpu_hours,"timeout_s":args.timeout_s,
                "disk_reservation_bytes":4*2**30,"expected_outputs":[str(x) for x in expected],
                "priority":30+scene_index}
            task_path=task_dir/f"{run_id}.task.json";atomic_json(task_path,task)
            rows.append({"scene":scene,"condition":condition,"rotation_degrees":rotation,
                "translation_fraction":translation,"config":str(config_path),"config_sha256":file_hash(config_path),
                "task":str(task_path),"task_sha256":file_hash(task_path),"gpu_index":gpu})
    atomic_json(task_dir/"matrix_receipt.json",{"schema":1,"revision":args.revision,
        "protocol":str(protocol),"protocol_sha256":file_hash(protocol),"created_before_scores":True,
        "queue":str(queue_path),"round_start_charged_gpu_hours":args.round_start_charged,
        "total_ledger_cap_gpu_hours":queue["budget_gpu_hours"],"total_ledger_remaining_at_freeze":queue["budget_gpu_hours"]-args.round_start_charged,
        "round_cap_gpu_hours":args.round_cap_gpu_hours,"per_task_max_gpu_hours":args.max_gpu_hours,
        "per_task_timeout_s":args.timeout_s,"worst_case_reservation_gpu_hours":worst_reservation,
        "max_parallel":4,"zero_references":zero_references,"new_runs":rows,
        "limits":["development scenes only","no final_v1 access","one training seed and one noise-direction set",
                  "controlled errors added to estimated poses, not known absolute pose error","no geometry ground truth"]})
    print(task_dir/"matrix_receipt.json")


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as error:raise SystemExit(f"BLOCKED: {error}")
