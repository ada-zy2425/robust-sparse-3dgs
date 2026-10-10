#!/usr/bin/env python3
"""Trace every operation through the first actual structural update."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from bgs.core import WorkflowError,atomic_json,file_hash,read_json
from bgs.train_adapter import make_runner


def first_invalid(value,path=""):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        bad=torch.nonzero(~torch.isfinite(value.detach()),as_tuple=False)
        if len(bad):
            index=[int(x) for x in bad[0].cpu()];scalar=value.detach()[tuple(index)].item()
            return {"field":path,"index":index,"value":str(scalar)}
    elif isinstance(value,np.ndarray):
        bad=np.argwhere(~np.isfinite(value))
        if len(bad):
            index=[int(x) for x in bad[0]];return {"field":path,"index":index,"value":str(value[tuple(index)])}
    elif isinstance(value,Mapping):
        for key,item in value.items():
            found=first_invalid(item,f"{path}.{key}" if path else str(key))
            if found:return found
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):
            found=first_invalid(item,f"{path}[{index}]")
            if found:return found
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    args=p.parse_args();config_path=Path(args.config).resolve();out=Path(args.out).resolve()
    if out.exists():raise WorkflowError("Refusing to overwrite stability probe")
    spec=read_json(config_path);manifest=read_json(spec["manifest"]);training_input=Path(spec["diagnostic_training_input"])
    work=out.parent/(out.stem+".work");runner,_=make_runner(spec,manifest,training_input,work)
    initial_count=len(runner.splats["means"])
    events=[];first_problem=None;last_count=initial_count;max_step=-1
    operations=("sample","forward","pre_backward","backward","post_update","pre_strategy","post_strategy","after_step")
    def trace(label,step,local_vars):
        nonlocal first_problem,last_count,max_step
        max_step=max(max_step,int(step))
        selected={"splats":runner.splats}
        if label=="sample":selected.update({k:local_vars.get(k) for k in ("camtoworlds","Ks","pixels","image_ids")})
        elif label=="forward":selected.update({k:local_vars.get(k) for k in ("renders","alphas","info")})
        elif label=="pre_backward":selected.update({k:local_vars.get(k) for k in ("loss","l1loss","ssimloss","colors","pixels")})
        elif label=="backward":selected["gradients"]={k:v.grad for k,v in runner.splats.items()}
        elif label in ("post_update","pre_strategy","post_strategy","after_step"):
            selected["optimizer_state"]={k:v.state for k,v in runner.optimizers.items()}
            selected["strategy_state"]=runner.strategy_state
        problem=first_invalid(selected)
        if problem is not None:
            first_problem={"step":int(step),"operation":label,**problem}
            raise FloatingPointError(f"First traced non-finite value: {first_problem}")
        if label=="post_strategy":
            count=len(runner.splats["means"])
            if count!=last_count:
                events.append({"step":int(step),"before":last_count,"after":count})
            last_count=count
    runner._trace_point=trace
    started=time.perf_counter();error=None
    try:runner.train()
    except FloatingPointError as exc:error=str(exc)
    elapsed=time.perf_counter()-started
    crossed=bool(events and max_step>events[0]["step"])
    status="PASSED" if first_problem is None and crossed else ("NONFINITE" if first_problem else "NO_DENSIFICATION")
    atomic_json(out,{"schema":1,"status":status,"run_id":spec["run_id"],"scene":manifest["scene_id"],
        "arm":spec["vggt_ba_ablation_arm"],"policy":spec["splat_numerics_policy"],
        "config_sha256":file_hash(config_path),"source_config_sha256":spec["source_config_sha256"],
        "operations_checked_each_step":list(operations),"configured_steps":spec["steps"],"last_completed_trace_step":max_step,
        "source_snapshot_points":int(spec["expected_initial_points"]),"initial_gaussians":initial_count,
        "final_gaussians":len(runner.splats["means"]),
        "structural_events":events,"crossed_first_structural_event":crossed,"first_nonfinite":first_problem,
        "error":error,"elapsed_seconds":elapsed,
        "initialization_numerics_sha256":file_hash(work/"initialization_numerics.json")})
    print(out)


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as error:raise SystemExit(f"BLOCKED: {error}")
