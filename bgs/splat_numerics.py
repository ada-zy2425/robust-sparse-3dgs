"""Versioned, conventional numerical guards for point-to-splat initialization."""
from __future__ import annotations

import math

from .core import WorkflowError


UPSTREAM_POLICY="upstream_v1"
DEDUP_POLICY="exact_deduplicate_mean_color_v1"
FLOOR_POLICY="dtype_scene_scale_floor_v1"
POLICIES={UPSTREAM_POLICY,DEDUP_POLICY,FLOOR_POLICY}


def _validate_points(points):
    import numpy as np
    value=np.asarray(points)
    if value.ndim!=2 or value.shape[1]!=3 or len(value)<4 or not np.issubdtype(value.dtype,np.floating):
        raise WorkflowError("Initialization points must be a floating [N,3] array with N>=4")
    if not np.isfinite(value).all():raise WorkflowError("Initialization points must be finite")
    return value


def stable_exact_deduplicate(points,colors,errors=None,point_indices=None):
    """Keep first-coordinate order and mean colors within exact duplicate clusters."""
    import numpy as np
    value=_validate_points(points);rgb=np.asarray(colors)
    if rgb.shape!=(len(value),3) or not np.isfinite(rgb).all():
        raise WorkflowError("Point colors must be a finite [N,3] array")
    unique,first,inverse,counts=np.unique(value,axis=0,return_index=True,return_inverse=True,return_counts=True)
    order=np.argsort(first);old_group_to_new=np.empty(len(order),dtype=np.int64)
    old_group_to_new[order]=np.arange(len(order));old_to_new=old_group_to_new[inverse]
    selected=first[order];output_points=value[selected].copy();output_counts=counts[order]
    sums=np.zeros((len(order),3),dtype=np.float64);np.add.at(sums,old_to_new,rgb.astype(np.float64))
    output_colors=(sums/output_counts[:,None]).astype(np.float32)
    output_errors=None
    if errors is not None:
        err=np.asarray(errors)
        if err.shape!=(len(value),) or not np.isfinite(err).all():raise WorkflowError("Point errors must be finite [N]")
        error_sums=np.zeros(len(order),dtype=np.float64);np.add.at(error_sums,old_to_new,err.astype(np.float64))
        output_errors=(error_sums/output_counts).astype(np.float32)
    output_indices=None
    if point_indices is not None:
        output_indices={}
        for name,indices in point_indices.items():
            idx=np.asarray(indices)
            if idx.ndim!=1 or ((idx<0)|(idx>=len(value))).any():raise WorkflowError("Invalid point index mapping")
            output_indices[name]=np.unique(old_to_new[idx]).astype(np.int32)
    conflicting=0;max_change=0.
    for new_index,count in enumerate(output_counts):
        if count<=1:continue
        members=np.flatnonzero(old_to_new==new_index);cluster=rgb[members].astype(np.float64)
        conflicting+=int(len(np.unique(cluster,axis=0))>1)
        max_change=max(max_change,float(np.abs(cluster-output_colors[new_index]).max()))
    before_min,before_max=value.min(0),value.max(0)
    receipt={"policy":DEDUP_POLICY,"input_points":int(len(value)),"output_points":int(len(output_points)),
        "removed_points":int(len(value)-len(output_points)),
        "duplicate_cluster_count":int((output_counts>1).sum()),
        "points_in_duplicate_clusters":int(output_counts[output_counts>1].sum()),
        "conflicting_color_cluster_count":conflicting,"color_aggregation":"arithmetic mean in RGB byte space",
        "maximum_color_channel_change_from_cluster_member":max_change,
        "point_order":"first occurrence order","spatial_coverage_preserved":bool(
            np.array_equal(before_min,output_points.min(0)) and np.array_equal(before_max,output_points.max(0)))}
    return output_points,output_colors,output_errors,output_indices,old_to_new,receipt


def guarded_log_scales(points,distances,scene_scale,init_scale):
    """Apply a scale-equivariant lower bound derived from dtype precision."""
    import torch
    if not torch.is_tensor(points) or points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not points.is_floating_point():
        raise WorkflowError("Scale guard points must be a floating [N,3] tensor with N>=4")
    if not torch.isfinite(points).all():raise WorkflowError("Scale guard points must be finite")
    if not torch.is_tensor(distances) or distances.shape!=(len(points),4) or distances.dtype!=points.dtype:
        raise WorkflowError("Scale guard distances must be a same-dtype [N,4] tensor")
    if not torch.isfinite(distances).all() or (distances<0).any():raise WorkflowError("KNN distances must be finite and nonnegative")
    if not isinstance(scene_scale,(int,float)) or not math.isfinite(scene_scale) or scene_scale<=0:
        raise WorkflowError("scene_scale must be finite and positive")
    if not isinstance(init_scale,(int,float)) or not math.isfinite(init_scale) or init_scale<=0:
        raise WorkflowError("init_scale must be finite and positive")
    dist2_avg=(distances[:,1:]**2).mean(dim=-1);raw=torch.sqrt(dist2_avg)*float(init_scale)
    reference=max(float(points.abs().max()),float(scene_scale))
    floor=float(torch.finfo(points.dtype).eps)*reference*float(init_scale)
    guarded=torch.clamp_min(raw,floor);logs=torch.log(guarded).unsqueeze(-1).repeat(1,3)
    changed=raw<floor;positive=raw[raw>0]
    receipt={"policy":FLOOR_POLICY,"formula":"eps(dtype) * max(max_abs_coordinate, scene_scale) * init_scale",
        "dtype":str(points.dtype),"scene_scale":float(scene_scale),"coordinate_reference_scale":reference,
        "init_scale":float(init_scale),"positive_scale_floor":floor,
        "input_points":int(len(points)),"output_points":int(len(points)),"point_count_changed":False,
        "colors_changed":False,"spatial_coverage_changed":False,"zero_raw_scale_points":int((raw==0).sum()),
        "clamped_point_count":int(changed.sum()),"minimum_positive_raw_scale":float(positive.min()) if len(positive) else None}
    return logs,receipt


def first_nonfinite_splat(splats):
    """Return an exact first invalid parameter location, including zero physical scales."""
    import torch
    for name in ("means","scales","quats","opacities","sh0","shN"):
        if name not in splats:continue
        value=splats[name].detach();bad=torch.nonzero(~torch.isfinite(value),as_tuple=False)
        if len(bad):
            index=[int(x) for x in bad[0].cpu()]
            scalar=value[tuple(index)].item()
            return {"field":name,"index":index,"kind":"nonfinite","value":str(scalar)}
    physical=torch.exp(splats["scales"].detach());bad=torch.nonzero(
        (~torch.isfinite(physical))|(physical<=0),as_tuple=False)
    if len(bad):
        index=[int(x) for x in bad[0].cpu()]
        return {"field":"exp(scales)","index":index,"kind":"nonpositive_or_nonfinite",
                "value":str(physical[tuple(index)].item())}
    return None
