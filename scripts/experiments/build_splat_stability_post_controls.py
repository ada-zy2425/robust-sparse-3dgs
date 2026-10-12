#!/usr/bin/env python3
"""Freeze post-BA short controls for the selected scale-floor policy."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from bgs.core import WorkflowError,atomic_json,file_hash,read_json
from bgs.splat_numerics import FLOOR_POLICY


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--revision",default="v1")
    p.add_argument("--gpu-uuids",nargs=2,required=True);args=p.parse_args()
    if any(not x.startswith("GPU-") for x in args.gpu_uuids):raise WorkflowError("Explicit GPU UUIDs required")
    root=Path(args.root).resolve();config_dir=root/"configs"/f"splat_stability_post_{args.revision}"
    task_dir=root/"remote_tasks"/f"splat_stability_post_{args.revision}"
    out_dir=root/"work"/f"splat_stability_post_{args.revision}"
    if config_dir.exists() or task_dir.exists() or out_dir.exists():raise WorkflowError("Refusing to overwrite post controls")
    config_dir.mkdir(parents=True);task_dir.mkdir(parents=True);rows=[]
    for gpu,scene in enumerate(("bonsai","room")):
        source_id=f"m360_{scene}_dev_v24_vggt_ba_post_s42_7k_v1"
        source_path=root/"configs"/"ba_operation_ablation_v1"/f"{source_id}.json";source=read_json(source_path)
        run_id=f"m360_{scene}_post_s42_floor_stability801_{args.revision}"
        old_run=Path(source["root"])/"runs"/source_id
        config={**source,"run_id":run_id,"root":str(out_dir/"campaigns"/run_id),"steps":801,
            "checkpoint_every":801,"diagnostic_steps":[],"splat_numerics_policy":FLOOR_POLICY,
            "diagnostic_training_input":str(old_run/"input"),"expected_initial_points":22192 if scene=="bonsai" else 36226,
            "source_config":str(source_path),"source_config_sha256":file_hash(source_path),
            "hypothesis":"The selected floor policy is a no-op on the finite post-BA snapshot and remains finite through an actual structural update.",
            "scope":"development-only short matched-policy control; no quality claim"}
        config_path=config_dir/f"{run_id}.json";atomic_json(config_path,config)
        output=out_dir/f"{run_id}.json";task_id=f"probe_{run_id}"
        task={"task_id":task_id,"kind":"initialization_diagnostic",
            "argv":[str(root/"envs"/"gsplat"/"bin"/"python"),str(root/"scripts"/"diagnostics" / "run_splat_stability_probe.py"),
                    "--config",str(config_path),"--out",str(output)],
            "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                "TMPDIR":str(root/"tmp"/f"splat_stability_post_{args.revision}"/task_id)},
            "dependencies":[],"config_path":str(config_path),"config_sha256":file_hash(config_path),
            "gpu_index":gpu,"gpu_uuid":args.gpu_uuids[gpu],"gpu_count":1,"max_gpu_hours":.2,
            "timeout_s":720,"disk_reservation_bytes":3*2**30,"expected_outputs":[str(output)],"priority":4}
        task_path=task_dir/f"{task_id}.task.json";atomic_json(task_path,task)
        rows.append({"scene":scene,"policy":FLOOR_POLICY,"config":str(config_path),
                     "config_sha256":file_hash(config_path),"task":str(task_path)})
    atomic_json(task_dir/"matrix_receipt.json",{"schema":1,"revision":args.revision,"seed":42,
        "steps":801,"arms":"post_ba","selected_policy":FLOOR_POLICY,
        "selection_evidence":"splat_stability_candidates_v1","rows":rows})
    print(task_dir/"matrix_receipt.json")


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as error:raise SystemExit(f"BLOCKED: {error}")
