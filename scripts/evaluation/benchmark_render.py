#!/usr/bin/env python3
"""Bounded multi-camera render-only latency benchmark for a completed run."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import time

from bgs.core import WorkflowError,atomic_json,file_hash,read_json
from bgs.data import validate_manifest
from bgs.pipeline import load_spec
from bgs.train_adapter import load_checkpoint,make_runner


def percentile(values,q):
    ordered=sorted(values)
    position=(len(ordered)-1)*q
    lower=math.floor(position);upper=math.ceil(position)
    if lower==upper:return ordered[lower]
    return ordered[lower]*(upper-position)+ordered[upper]*(position-lower)


def summary(values):
    if not values or not all(math.isfinite(x) and x>0 for x in values):
        raise WorkflowError("Benchmark produced invalid latency samples")
    return {"count":len(values),"mean":statistics.fmean(values),"stdev":statistics.pstdev(values),
            "min":min(values),"p50":percentile(values,.50),"p90":percentile(values,.90),
            "p95":percentile(values,.95),"max":max(values)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--warmup",type=int,default=5)
    parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--max-cameras",type=int,default=64)
    args=parser.parse_args()
    if not 0<=args.warmup<=20 or not 1<=args.repeats<=100 or not 1<=args.max_cameras<=100:
        raise WorkflowError("warmup/repeats/max-cameras exceed the bounded benchmark protocol")
    output=Path(args.out).resolve()
    if output.exists():raise WorkflowError("Refusing to overwrite benchmark output")
    spec=load_spec(args.config);manifest=validate_manifest(read_json(spec["manifest"]))
    run_dir=Path(spec["root"])/"runs"/spec["run_id"]
    training=run_dir/"training"
    if not (training/"complete.json").is_file():raise WorkflowError("Run is not complete")

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise WorkflowError("Select exactly one CUDA device")
    runner,_=make_runner(spec,manifest,run_dir/"input",training,read_only=True)
    if not load_checkpoint(runner) or runner._resume_step!=spec["steps"]:
        raise WorkflowError("Completed final checkpoint is unavailable")
    cameras=min(len(runner.valset),args.max_cameras)
    per_camera=[];all_samples=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index in range(cameras):
            data=runner.valset[index]
            image=data["image"];height,width=image.shape[:2]
            kwargs={"camtoworlds":data["camtoworld"][None].to(runner.device),
                    "Ks":data["K"][None].to(runner.device),"width":width,"height":height,
                    "sh_degree":runner.cfg.sh_degree,"near_plane":runner.cfg.near_plane,
                    "far_plane":runner.cfg.far_plane,"frame_idcs":None,
                    "camera_idcs":torch.tensor([0],device=runner.device),"exposure":None}
            for _ in range(args.warmup):runner.stage.render(runner.scene.id,**kwargs)
            torch.cuda.synchronize();samples=[]
            for _ in range(args.repeats):
                torch.cuda.synchronize();started=time.perf_counter()
                runner.stage.render(runner.scene.id,**kwargs)
                torch.cuda.synchronize();samples.append((time.perf_counter()-started)*1000.)
            stats=summary(samples);stats.update(name=data["name"],width=width,height=height)
            per_camera.append(stats);all_samples.extend(samples)
    latency=summary(all_samples)
    result={"schema":1,"status":"VALID","run_id":spec["run_id"],
            "protocol":"all held-out calibrated cameras; render-only host dispatch plus synchronized GPU; inputs resident on GPU",
            "warmup_per_camera":args.warmup,"repeats_per_camera":args.repeats,
            "camera_count":cameras,"sample_count":len(all_samples),"latency_ms":latency,
            "fps_from_mean_latency":1000./latency["mean"],"per_camera":per_camera,
            "num_gaussians":len(runner.splats["means"]),"sh_degree":runner.cfg.sh_degree,
            "dtype":str(runner.splats["means"].dtype),"gpu_name":torch.cuda.get_device_name(0),
            "compute_capability":list(torch.cuda.get_device_capability(0)),
            "torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "checkpoint_sha256":file_hash(runner._checkpoint),
            "config_sha256":file_hash(args.config)}
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(output)


if __name__=="__main__":
    raise SystemExit(main())
