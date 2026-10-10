from pathlib import Path
import argparse
import json
import os
import sys

from .core import WorkflowError,atomic_json,read_json
from .splat_numerics import POLICIES,UPSTREAM_POLICY


def diagnostic_steps(total_steps):
    return [step for step in (1000,3000,7000) if step<=total_steps]


def main():
    p=argparse.ArgumentParser(description="B industrial 3DGS workflow")
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("inspect"); q.add_argument("--root",required=True); q.add_argument("--out")
    q=sub.add_parser("reference"); q.add_argument("--model",required=True); q.add_argument("--out",required=True); q.add_argument("--source",required=True)
    q=sub.add_parser("prepare"); q.add_argument("--images",required=True); q.add_argument("--reference",required=True)
    q.add_argument("--out",required=True); q.add_argument("--train-views",type=int,default=6); q.add_argument("--test-every",type=int,default=8); q.add_argument("--scene-id",default="scene0")
    q.add_argument("--partition",choices=["development","final"],default="development"); q.add_argument("--final-protocol")
    q=sub.add_parser("configure"); q.add_argument("--root",required=True); q.add_argument("--manifest",required=True)
    q.add_argument("--run-id",required=True); q.add_argument("--initializer",choices=["colmap","vggt"],required=True)
    q.add_argument("--gpu",required=True); q.add_argument("--gpu-hours",type=float,required=True)
    q.add_argument("--steps",type=int,default=200); q.add_argument("--max-edge",type=int,default=1024)
    q.add_argument("--timeout",type=int,default=1200); q.add_argument("--out",required=True)
    q.add_argument("--gsplat-root",required=True); q.add_argument("--vggt-root"); q.add_argument("--vggt-python")
    q.add_argument("--splat-numerics-policy",choices=sorted(POLICIES),default=UPSTREAM_POLICY)
    q=sub.add_parser("run"); q.add_argument("--config",required=True); q.add_argument("--retry-failed",action="store_true"); q.add_argument("--recover-stale",action="store_true")
    q=sub.add_parser("loop"); q.add_argument("--configs",nargs="+",required=True); q.add_argument("--max-experiments",type=int,default=2)
    q=sub.add_parser("worker"); q.add_argument("--config",required=True); q.add_argument("--phase",choices=["initialize","train","viewer"],required=True)
    q=sub.add_parser("view"); q.add_argument("--config",required=True)
    q=sub.add_parser("status"); q.add_argument("--root",required=True)
    args=p.parse_args()
    try:
        if args.command=="inspect":
            from .preflight import inspect
            result=inspect(args.root)
            if args.out: atomic_json(args.out,result)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=="reference":
            from .data import export_colmap_reference
            if Path(args.out).exists(): raise WorkflowError("Refusing to overwrite reference cameras")
            export_colmap_reference(args.model,args.out,args.source)
            print(args.out)
        elif args.command=="prepare":
            from .data import prepare
            result=prepare(args.images,args.reference,args.out,args.train_views,args.test_every,args.scene_id,
                           args.partition,args.final_protocol)
            print(json.dumps({"train":result["train"],"holdout":result["holdout"],"manifest":args.out},indent=2))
        elif args.command=="configure":
            if Path(args.out).exists(): raise WorkflowError("Refusing to overwrite a config")
            spec={"run_id":args.run_id,"root":str(Path(args.root).resolve()),"manifest":str(Path(args.manifest).resolve()),
                  "gsplat_root":str(Path(args.gsplat_root).resolve()),"python":sys.executable,
                  "initializer":args.initializer,"gpu":args.gpu,"gpu_hours":args.gpu_hours,
                  "steps":args.steps,"max_edge":args.max_edge,"seed":42,"noise_seed":1729,
                  "checkpoint_every":100,"diagnostic_steps":diagnostic_steps(args.steps),
                  "max_gaussians":2000000,"timeout_s":args.timeout,
                  "rotation_degrees":0.,"translation_fraction":0.,"pose_opt":False,
                  "splat_numerics_policy":args.splat_numerics_policy,
                  "cache_images":False,"hypothesis":"Compare initialization at fixed inputs and training budget",
                  "independent_contribution":"workflow and evaluation integration; algorithm mechanism pending observations"}
            if args.initializer=="colmap":
                spec["colmap_binary"]=str(Path(sys.executable).with_name("colmap").resolve())
                spec["colmap_use_gpu"]=False
            if args.vggt_root: spec["vggt_root"]=str(Path(args.vggt_root).resolve())
            if args.vggt_python: spec["vggt_python"]=str(Path(args.vggt_python).resolve())
            if args.initializer=="vggt":
                if not args.vggt_root: raise WorkflowError("--vggt-root is required")
                spec.update(vggt_ba=False,vggt_confidence=5.0,vggt_max_points=100000,
                            vggt_point_sampling="upstream_seeded_random_limit")
            atomic_json(args.out,spec)
            from .pipeline import load_spec
            load_spec(args.out)
            print(args.out)
        elif args.command=="run":
            from .pipeline import run
            print(json.dumps(run(args.config,args.retry_failed,args.recover_stale),indent=2))
        elif args.command=="loop":
            from .pipeline import run
            if not 1<=args.max_experiments<=20 or len(args.configs)>args.max_experiments:
                raise WorkflowError("Explicit finite experiment cap exceeded")
            results=[]
            for config in args.configs:
                # Stops immediately on invalid data, timeout, insufficient budget
                # or an unresolved attempt. There is no hidden automatic retry.
                results.append(run(config))
            print(json.dumps({"results":results,"next":"Inspect paired evidence before designing the next round"},indent=2))
        elif args.command in ("worker","view"):
            from .pipeline import worker,load_spec
            spec=load_spec(args.config)
            os.environ["CUDA_VISIBLE_DEVICES"]=str(spec["gpu"])
            worker(args.config,"viewer" if args.command=="view" else args.phase)
        elif args.command=="status":
            ledger=Path(args.root)/"ledger.json"
            if not ledger.exists(): print("No jobs have been submitted.")
            else:
                data=read_json(ledger)
                print(json.dumps({"charged_gpu_hours":data["charged_gpu_hours"],
                   "stages":{k:{"status":v["status"],"attempts":len(v["attempts"]),"latest":v["attempts"][-1]} for k,v in data["stages"].items()}},indent=2))
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:
        print(f"BLOCKED: {e}",file=sys.stderr)
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
