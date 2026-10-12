#!/usr/bin/env python3
"""CPU probe of the pinned Parser/Dataset camera path for a frozen pose matrix."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np

from bgs.core import WorkflowError,atomic_json,file_hash,read_json
from bgs.geometry import perturb_training_poses,rotation_error_degrees
from bgs.train_adapter import sized_tensor


def bytes_hash(value):
    array=value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--matrix",required=True)
    p.add_argument("--out",required=True);args=p.parse_args()
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    if out.exists():raise WorkflowError("Refusing to overwrite pose probe")
    matrix_path=Path(args.matrix).resolve();matrix=read_json(matrix_path)
    sys.path.insert(0,str(root/"third_party"/"gsplat"/"examples"));sys.path.insert(0,str(root/"third_party"/"gsplat"))
    from datasets.colmap import Dataset,Parser

    class AllSelectedDataset(Dataset):
        def __init__(self,parser,split="train",**kwargs):
            super().__init__(parser,split=split,**kwargs);self.indices=np.arange(len(parser.image_names))
        def __getitem__(self,item):
            data=super().__getitem__(item);data["image"],data["K"]=sized_tensor(data["image"],data["K"],1024);return data

    rows={}
    for scene,zero in matrix["zero_references"].items():
        config=read_json(zero["config"]);manifest=read_json(config["manifest"])
        zero_run=Path(config["root"])/"runs"/zero["run_id"];training_input=zero_run/"input"
        parser=Parser(data_dir=str(training_input),factor=1,normalize=True,test_every=8,load_exposure=False)
        dataset=AllSelectedDataset(parser,split="train",patch_size=None,load_depths=False)
        base=parser.camtoworlds.copy();points_hash=bytes_hash(parser.points)
        intrinsics_hash=hashlib.sha256("".join(bytes_hash(parser.Ks_dict[k]) for k in sorted(parser.Ks_dict)).encode()).hexdigest()
        paths=list(parser.image_paths);path_hash=hashlib.sha256("\n".join(paths).encode()).hexdigest()
        if len(dataset)!=24 or list(parser.image_names)!=read_json(zero["outputs"]["alignment"])["training_names"]:
            raise WorkflowError(f"Actual Parser order differs from frozen training order for {scene}")
        if set(parser.image_names)!=set(manifest["train"]):raise WorkflowError(f"Training denominator mismatch for {scene}")
        baseline=[]
        for i in range(len(dataset)):
            data=dataset[i];baseline.append({"image":bytes_hash(data["image"]),"K":bytes_hash(data["K"])})
            np.testing.assert_array_equal(data["camtoworld"].numpy(),base[i].astype(np.float32))
        centers=base[:,:3,3];radius=float(np.max(np.linalg.norm(centers-centers.mean(0),axis=1)))
        conditions={"zero":(0.,0.)}
        for row in matrix["new_runs"]:
            if row["scene"]==scene:conditions[row["condition"]]=(row["rotation_degrees"],row["translation_fraction"])
        results={};generated={}
        for name,(rotation,translation) in conditions.items():
            poses=perturb_training_poses(base,config["noise_seed"],rotation,translation);generated[name]=poses
            parser.camtoworlds=poses
            read_poses=[];rgb_and_k_equal=True
            for i in range(len(dataset)):
                data=dataset[i];read_poses.append(data["camtoworld"].numpy())
                rgb_and_k_equal &= baseline[i]=={"image":bytes_hash(data["image"]),"K":bytes_hash(data["K"])}
            read_poses=np.asarray(read_poses);expected=poses.astype(np.float32)
            rotation_errors=[rotation_error_degrees(poses[i,:3,:3],base[i,:3,:3]) for i in range(len(base))]
            shifts=np.linalg.norm(poses[:,:3,3]-base[:,:3,3],axis=1)
            orthogonality=max(float(np.max(np.abs(r@r.T-np.eye(3)))) for r in poses[:,:3,:3])
            determinant=max(abs(float(np.linalg.det(r))-1.) for r in poses[:,:3,:3])
            result={"rotation_degrees":rotation,"translation_fraction":translation,
                "zero_exact":bool(np.array_equal(poses,base)) if name=="zero" else None,
                "dataset_pose_exact_float32":bool(np.array_equal(read_poses,expected)),
                "rgb_and_intrinsics_exact":bool(rgb_and_k_equal),"anchor_exact":bool(np.array_equal(poses[0],base[0])),
                "non_anchor_rotation_degrees_min":min(rotation_errors[1:]),"non_anchor_rotation_degrees_max":max(rotation_errors[1:]),
                "non_anchor_translation_min":float(shifts[1:].min()),"non_anchor_translation_max":float(shifts[1:].max()),
                "max_rotation_orthogonality_error":orthogonality,"max_rotation_determinant_error":determinant}
            if not result["dataset_pose_exact_float32"] or not result["rgb_and_intrinsics_exact"] or not result["anchor_exact"]:
                raise WorkflowError(f"Actual Dataset perturbation contract failed for {scene}/{name}")
            results[name]=result
        for low,high in (("r05","r10"),("t005","t010")):
            if low.startswith("r"):
                errors=[np.max(np.abs((generated[low][i,:3,:3]@base[i,:3,:3].T)@(
                    generated[low][i,:3,:3]@base[i,:3,:3].T)-(generated[high][i,:3,:3]@base[i,:3,:3].T)))
                    for i in range(1,len(base))]
            else:
                errors=[np.max(np.abs(2*(generated[low][i,:3,3]-base[i,:3,3])-(generated[high][i,:3,3]-base[i,:3,3])))
                    for i in range(1,len(base))]
            if max(errors)>1e-12:raise WorkflowError(f"Noise directions differ across levels for {scene}/{low}/{high}")
        parser.camtoworlds=base
        rows[scene]={"training_views":len(dataset),"holdout_views":len(manifest["holdout"]),
            "anchor_index":0,"anchor_name":parser.image_names[0],"camera_radius":radius,
            "radius_definition":"max norm(center - arithmetic mean of 24 unperturbed centers)",
            "points_hash_before_after":points_hash,"intrinsics_hash_before_after":intrinsics_hash,
            "image_path_hash_before_after":path_hash,"initialization_files":zero["initialization_files"],
            "showcase_holdout":manifest["holdout"][0],"conditions":results}
    atomic_json(out,{"schema":1,"status":"PASS","matrix":str(matrix_path),"matrix_sha256":file_hash(matrix_path),
        "actual_path":"Pinned gsplat Parser and Dataset with adapter's all-selected and max_edge wrapper",
        "alignment_order":"The production adapter fits alignment before replacing parser.camtoworlds; this probe uses the persisted unperturbed alignment order.",
        "heldout_behavior":"Production HeldoutDataset is constructed from reference cameras and the pre-perturbation alignment, not parser.camtoworlds.",
        "scenes":rows})
    print(out)


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError,AssertionError) as error:raise SystemExit(f"BLOCKED: {error}")
