#!/usr/bin/env python3
"""Build four bounded read-only pre/post BA initialization audits."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from bgs.core import WorkflowError,atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--revision",default="v1")
    p.add_argument("--gpu-uuids",nargs=4,required=True,metavar=("GPU0","GPU1","GPU2","GPU3"));args=p.parse_args()
    if any(not x.startswith("GPU-") for x in args.gpu_uuids):raise WorkflowError("Explicit GPU UUIDs required")
    root=Path(args.root).resolve();task_dir=root/"remote_tasks"/f"splat_initialization_diagnostic_{args.revision}"
    out_dir=root/"work"/f"splat_initialization_diagnostic_{args.revision}"
    if task_dir.exists() or out_dir.exists():raise WorkflowError("Refusing to overwrite initialization diagnostic")
    task_dir.mkdir(parents=True);rows=[]
    assignments=[("bonsai","pre",0),("bonsai","post",1),("room","pre",2),("room","post",3)]
    for scene,arm,gpu in assignments:
        run_id=f"m360_{scene}_dev_v24_vggt_ba_{arm}_s42_7k_v1"
        config=root/"configs"/"ba_operation_ablation_v1"/f"{run_id}.json"
        task_id=f"diag_init_{scene}_{arm}_s42_{args.revision}"
        output=out_dir/f"{task_id}.json"
        task={"task_id":task_id,"kind":"initialization_diagnostic",
            "argv":[str(root/"envs"/"gsplat"/"bin"/"python"),str(root/"scripts"/"diagnose_splat_initialization.py"),
                    "--config",str(config),"--out",str(output)],
            "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                    "TMPDIR":str(root/"tmp"/f"splat_initialization_diagnostic_{args.revision}"/task_id)},
            "dependencies":[],"config_path":str(config),"config_sha256":file_hash(config),
            "gpu_index":gpu,"gpu_uuid":args.gpu_uuids[gpu],"gpu_count":1,"max_gpu_hours":.05,
            "timeout_s":180,"disk_reservation_bytes":2**30,"expected_outputs":[str(output)],"priority":3}
        path=task_dir/f"{task_id}.task.json";atomic_json(path,task);rows.append(str(path))
    atomic_json(task_dir/"matrix_receipt.json",{"schema":1,"revision":args.revision,
        "scope":"read-only exact point-to-splat initialization numerics","tasks":rows})
    print("\n".join(rows))


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as error:raise SystemExit(f"BLOCKED: {error}")
