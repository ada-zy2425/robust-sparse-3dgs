"""Create isolated pre/post-BA COLMAP models from one pinned VGGT BA reconstruction."""
from pathlib import Path
import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

from .core import WorkflowError, atomic_json, file_hash
from .vggt_wrapper import local_hub_loader


BA_CALL = "pycolmap.bundle_adjustment(reconstruction, ba_options)"
BA_REPLACEMENT = "reconstruction = __bgs_isolated_ba_pair__(reconstruction, ba_options)"


def patch_demo_source(source):
    if source.count(BA_CALL) != 1:
        raise WorkflowError("Pinned VGGT BA call contract changed")
    return source.replace(BA_CALL, BA_REPLACEMENT)


def directory_hashes(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


def reconstruction_stats(reconstruction):
    import numpy as np
    errors=[float(point.error) for point in reconstruction.points3D.values()
            if hasattr(point,"error") and np.isfinite(point.error) and point.error>=0]
    return {"cameras":len(reconstruction.cameras),"registered_images":len(reconstruction.reg_image_ids()),
            "points3D":len(reconstruction.points3D),
            "observations":sum(len(point.track.elements) for point in reconstruction.points3D.values()),
            "cached_reprojection_error_valid":bool(errors),
            "cached_point_error_mean":float(np.mean(errors)) if errors else None,
            "cached_point_error_max":float(np.max(errors)) if errors else None}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--demo",required=True);parser.add_argument("--dinov2-root",required=True)
    parser.add_argument("--pair-root",required=True)
    args,passthrough=parser.parse_known_args(argv)
    demo=Path(args.demo).resolve();dinov2=Path(args.dinov2_root).resolve();pair_root=Path(args.pair_root).resolve()
    if pair_root.exists():raise WorkflowError("Refusing to overwrite an immutable BA pair")
    scene_arg=next((passthrough[i+1] for i,x in enumerate(passthrough[:-1]) if x=="--scene_dir"),None)
    if scene_arg is None:raise WorkflowError("BA pair requires --scene_dir")
    scene=Path(scene_arg).resolve();staging_receipt=scene.parent/"staging.json"
    if not staging_receipt.is_file():raise WorkflowError("Missing immutable train-only staging receipt")
    source=demo.read_text();patched=patch_demo_source(source)
    pair_root.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix=pair_root.name+".",dir=pair_root.parent))
    raw=work/"snapshot_pre_ba";raw.mkdir();state={}
    import pycolmap
    original_ba=pycolmap.bundle_adjustment

    def isolated_pair(reconstruction,options):
        reconstruction.write(str(raw))
        before_hashes=directory_hashes(raw)
        arm_a=pycolmap.Reconstruction(str(raw));arm_b=pycolmap.Reconstruction(str(raw))
        state["distinct_python_objects"]=id(arm_a)!=id(arm_b)
        state["pre_stats"]=reconstruction_stats(arm_a)
        started=time.perf_counter();original_ba(arm_b,options);state["ba_seconds"]=time.perf_counter()-started
        state["post_stats_raw"]=reconstruction_stats(arm_b)
        state["snapshot_unchanged_after_ba"]=directory_hashes(raw)==before_hashes
        state["arm_a"]=arm_a
        return arm_b

    import torch
    original_load=torch.hub.load;torch.hub.load=local_hub_loader(original_load,dinov2)
    old_argv=sys.argv
    namespace={"__name__":"__main__","__file__":str(demo),"__bgs_isolated_ba_pair__":isolated_pair}
    try:
        sys.argv=[str(demo),*passthrough]
        exec(compile(patched,str(demo),"exec"),namespace,namespace)
        if "arm_a" not in state:raise WorkflowError("Pinned demo did not reach the BA hook")
        image_paths=glob.glob(str(scene/"images"/"*"))
        _,coords=namespace["load_and_preprocess_images_square"](image_paths,1024)
        pre=namespace["rename_colmap_recons_and_rescale_camera"](
            state.pop("arm_a"),[os.path.basename(p) for p in image_paths],coords.cpu().numpy(),
            img_size=1024,shift_point2d_to_original_res=True,shared_camera=False)
        (work/"pre_ba").mkdir();pre.write(str(work/"pre_ba"))
        shutil.copytree(scene/"sparse",work/"post_ba")
        post=pycolmap.Reconstruction(str(work/"post_ba"))
        state["post_stats"]=reconstruction_stats(post)
        state["pre_final_stats"]=reconstruction_stats(pre)
        atomic_json(work/"pair_receipt.json",{
            "schema":1,"scope":"isolated contribution of the official pycolmap.bundle_adjustment call only",
            "demo":str(demo),"demo_sha256":file_hash(demo),
            "staging_receipt_sha256":file_hash(staging_receipt),
            "source_patch":{"exact_call":BA_CALL,"replacement":BA_REPLACEMENT},
            **state,"pre_ba_files":directory_hashes(work/"pre_ba"),
            "post_ba_files":directory_hashes(work/"post_ba"),
            "limits":"Does not separate camera optimization from point optimization."})
        os.replace(work,pair_root)
    except BaseException:
        shutil.rmtree(work,ignore_errors=True);raise
    finally:
        sys.argv=old_argv;torch.hub.load=original_load


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as error:raise SystemExit(f"BLOCKED: {error}")
