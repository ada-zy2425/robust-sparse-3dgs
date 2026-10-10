#!/usr/bin/env python3
"""Read-only audit of the exact point-to-splat initialization path."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from bgs.core import WorkflowError,atomic_json,file_hash,read_json
from bgs.train_adapter import make_runner


QUANTILES=(0.,.001,.01,.05,.25,.5,.75,.95,.99,.999,1.)


def array_sha256(array):
    import numpy as np
    value=np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def finite_stats(value):
    import numpy as np
    a=np.asarray(value)
    finite=np.isfinite(a)
    values=a[finite]
    return {"shape":list(a.shape),"dtype":str(a.dtype),"total":int(a.size),
            "finite":int(finite.sum()),"nan":int(np.isnan(a).sum()),
            "positive_infinity":int(np.isposinf(a).sum()),
            "negative_infinity":int(np.isneginf(a).sum()),
            "finite_min":float(values.min()) if values.size else None,
            "finite_max":float(values.max()) if values.size else None,
            "sha256":array_sha256(a)}


def quantiles(value):
    import numpy as np
    a=np.asarray(value);a=a[np.isfinite(a)]
    return {str(q):float(np.quantile(a,q)) for q in QUANTILES} if a.size else {}


def exact_duplicate_stats(points,point_ids=None,colors=None,max_examples=20):
    import numpy as np
    p=np.asarray(points)
    unique,inverse,counts=np.unique(p,axis=0,return_inverse=True,return_counts=True)
    duplicate_groups=np.flatnonzero(counts>1)
    examples=[]
    for group in duplicate_groups[:max_examples]:
        indices=np.flatnonzero(inverse==group)
        row={"indices":[int(x) for x in indices],"size":int(len(indices)),
             "point":[float(x) for x in unique[group]]}
        if point_ids is not None:row["point3D_ids"]=[int(point_ids[x]) for x in indices]
        if colors is not None:
            group_colors=np.asarray(colors)[indices]
            row["colors"]=[[int(x) for x in color] for color in group_colors]
            row["unique_color_count"]=int(len(np.unique(group_colors,axis=0)))
        examples.append(row)
    return {"points":int(len(p)),"unique_points":int(len(unique)),
            "duplicate_rows":int((counts[counts>1]-1).sum()),
            "points_in_duplicate_clusters":int(counts[counts>1].sum()),
            "duplicate_cluster_count":int(len(duplicate_groups)),
            "maximum_cluster_size":int(counts.max()) if len(counts) else 0,
            "examples":examples}


def colmap_points(path):
    import numpy as np
    import pycolmap
    reconstruction=pycolmap.Reconstruction(str(path))
    ids=sorted(reconstruction.points3D)
    points=np.asarray([reconstruction.points3D[i].xyz for i in ids],dtype=np.float64).reshape(-1,3)
    colors=np.asarray([reconstruction.points3D[i].color for i in ids],dtype=np.uint8).reshape(-1,3)
    return ids,points,colors


def point_stage(points,point_ids=None,colors=None):
    import numpy as np
    a=np.asarray(points)
    return {"array":finite_stats(a),"coordinates":{"min":[float(x) for x in a.min(0)],
            "max":[float(x) for x in a.max(0)],"span":[float(x) for x in np.ptp(a,axis=0)]},
            "duplicates":exact_duplicate_stats(a,point_ids,colors)}


def parameter_stats(runner):
    import torch
    params={name:finite_stats(value.detach().cpu().numpy()) for name,value in runner.splats.items()}
    scales=torch.exp(runner.splats["scales"].detach()).cpu().numpy()
    params["exp_scales"]=finite_stats(scales)
    params["exp_scales"]["zero_count"]=int((scales==0).sum())
    params["quaternion_norms"]=finite_stats(torch.linalg.vector_norm(
        runner.splats["quats"].detach(),dim=-1).cpu().numpy())
    params["sigmoid_opacities"]=finite_stats(torch.sigmoid(
        runner.splats["opacities"].detach()).cpu().numpy())
    return params


def optimizer_stats(runner):
    rows={}
    for name,opt in runner.optimizers.items():
        group=opt.param_groups[0]
        rows[name]={"class":type(opt).__name__,"bound_to_current_parameter":
                    len(group["params"])==1 and group["params"][0] is runner.splats[name],
                    "learning_rate":float(group["lr"]),"eps":float(group["eps"]),
                    "betas":[float(x) for x in group["betas"]],
                    "state_parameter_count":len(opt.state),
                    "state_entries":[str(x) for x in next(iter(opt.state.values()),{}).keys()]}
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    args=p.parse_args();config_path=Path(args.config).resolve();out=Path(args.out).resolve()
    if out.exists():raise WorkflowError("Refusing to overwrite initialization diagnostic")
    spec={**read_json(config_path),"splat_numerics_allow_nonfinite_audit":True};manifest=read_json(spec["manifest"])
    if spec.get("vggt_ba_ablation_arm") not in ("pre_ba","post_ba"):
        raise WorkflowError("Diagnostic requires a matched BA-operation arm")
    run=Path(spec["root"])/"runs"/spec["run_id"]
    input_model=run/"input"/"sparse"/"0"
    pair_model=Path(spec["vggt_ba_pair_root"])/spec["vggt_ba_ablation_arm"]
    if not input_model.is_dir() or not pair_model.is_dir():raise WorkflowError("Missing immutable input model")
    pair_ids,pair_points,pair_colors=colmap_points(pair_model)
    input_ids,input_points,input_colors=colmap_points(input_model)
    if pair_ids!=input_ids or not (pair_points==input_points).all() or not (pair_colors==input_colors).all():
        raise WorkflowError("Run input is not the immutable BA-pair arm")
    work=out.parent/(out.stem+".work")
    runner,module=make_runner(spec,manifest,run/"input",work)
    processed=runner.parser.points
    import numpy as np
    import torch
    trainer_points=torch.from_numpy(processed).float()
    distances=module.knn(trainer_points,4).cpu().numpy()
    dist2_avg=np.mean(distances[:,1:]**2,axis=-1)
    dist_avg=np.sqrt(dist2_avg)
    init_scale=float(runner.cfg.init_scale)
    with np.errstate(divide="ignore",invalid="ignore"):
        scale_logs=np.log(dist_avg*init_scale)
    zero_indices=np.flatnonzero(dist2_avg==0)
    first_abnormal=None
    if len(zero_indices):
        index=int(zero_indices[0])
        first_abnormal={"phase":"initialization","operation":"log(sqrt(dist2_avg) * init_scale)",
            "field":"scales","point_index":index,"point3D_id":int(pair_ids[index]),
            "input":{"point":[float(x) for x in trainer_points[index]],
                     "neighbor_distances":[float(x) for x in distances[index]],
                     "dist2_avg":float(dist2_avg[index]),"init_scale":init_scale},
            "result":"negative_infinity"}
    report={"schema":1,"status":"INITIALIZATION_NONFINITE" if first_abnormal else "INITIALIZATION_FINITE",
        "run_id":spec["run_id"],"scene":manifest["scene_id"],"arm":spec["vggt_ba_ablation_arm"],
        "config_sha256":file_hash(config_path),"trainer_sha256":file_hash(Path(spec["gsplat_root"])/"examples"/"simple_trainer.py"),
        "pair_receipt_sha256":file_hash(Path(spec["vggt_ba_pair_root"])/"pair_receipt.json"),
        "source_identity":{"pair_model_files":{x.name:file_hash(x) for x in sorted(pair_model.iterdir()) if x.is_file()},
                           "run_input_files":{x.name:file_hash(x) for x in sorted(input_model.iterdir()) if x.is_file()},
                           "arrays_equal":True},
        "points":{"original_pair_float64":point_stage(pair_points,pair_ids,pair_colors),
                  "parser_processed":point_stage(processed,pair_ids,pair_colors),
                  "trainer_float32":point_stage(trainer_points.cpu().numpy(),pair_ids,pair_colors)},
        "knn":{"implementation":"pinned examples/utils.py sklearn NearestNeighbors euclidean",
            "distance_dtype":str(distances.dtype),"nearest_nonself_quantiles":quantiles(distances[:,1]),
            "all_three_nonself_quantiles":quantiles(distances[:,1:]),
            "zero_distance_counts_by_nonself_rank":[int((distances[:,i]==0).sum()) for i in (1,2,3)],
            "dist2_avg":finite_stats(dist2_avg),"dist2_avg_zero_count":int((dist2_avg==0).sum()),
            "dist_avg":finite_stats(dist_avg),"scale_logs":finite_stats(scale_logs),
            "zero_scale_point_indices":[int(x) for x in zero_indices[:100]],
            "zero_scale_point_count":int(len(zero_indices)),"init_scale":init_scale},
        "initial_parameters":parameter_stats(runner),"initial_optimizers":optimizer_stats(runner),
        "scene_scale":float(runner.scene_scale),"first_abnormal":first_abnormal,
        "interpretation":"Read-only exact-path audit before any forward, loss, backward, optimizer, or strategy operation."}
    atomic_json(out,report);print(out)


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as error:raise SystemExit(f"BLOCKED: {error}")
