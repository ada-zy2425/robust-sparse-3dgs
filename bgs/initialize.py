"""Run public COLMAP/VGGT only on the staged training image directory."""
from pathlib import Path
import os
import shutil
import subprocess
import uuid

from .core import WorkflowError, read_json, atomic_json, file_hash


def _vggt_policy(spec):
    policy={"bundle_adjustment":bool(spec.get("vggt_ba",False)),
            "confidence_threshold":float(spec.get("vggt_confidence",5.0)),
            "max_points":int(spec.get("vggt_max_points",100000)),
            "point_sampling":spec.get("vggt_point_sampling","upstream_seeded_random_limit")}
    if policy["max_points"]!=100000 or policy["point_sampling"]!="upstream_seeded_random_limit":
        raise WorkflowError("Pinned VGGT demo fixes a seeded random 100000-point limit; changing it requires a new pinned adapter")
    return policy


def _verified_dinov2_root(spec):
    if "dinov2_root" not in spec:
        raise WorkflowError("VGGT bundle adjustment requires a pinned local dinov2_root")
    root=Path(spec["dinov2_root"])
    if not root.is_absolute():
        raise WorkflowError("dinov2_root must be absolute")
    root=root.resolve()
    runtime_lock=Path(__file__).resolve().parents[1]/"vggt_runtime.lock.json"
    pins=read_json(runtime_lock)["dinov2"]
    expected={"hubconf.py":pins["hubconf_sha256"],
              "dinov2/hub/backbones.py":pins["backbones_sha256"]}
    for relative,digest in expected.items():
        path=root/relative
        if not path.is_file() or file_hash(path)!=digest:
            raise WorkflowError(f"DINOv2 source differs from pinned file: {relative}")
    return root


def _verify_ba_cache(runtime_lock=None):
    lock_path=Path(runtime_lock) if runtime_lock else Path(__file__).resolve().parents[1]/"vggt_runtime.lock.json"
    pins=read_json(lock_path)
    torch_home=os.environ.get("TORCH_HOME")
    if not torch_home or not Path(torch_home).is_absolute():
        raise WorkflowError("VGGT bundle adjustment requires an absolute TORCH_HOME")
    checkpoints=Path(torch_home)/"hub"/"checkpoints"
    required={
        "vggsfm_v2_tracker.pt":pins["vggsfm_tracker"]["sha256"],
        "dinov2_vitb14_reg4_pretrain.pth":pins["dinov2"]["checkpoint_sha256"],
        "aliked-n16.pth":pins["aliked"]["sha256"],
        "superpoint_v1.pth":pins["superpoint"]["sha256"],
    }
    for name,digest in required.items():
        path=checkpoints/name
        if not path.is_file() or file_hash(path)!=digest:
            raise WorkflowError(f"VGGT BA cache missing or differs from runtime lock: {name}")
    return checkpoints


def _colmap_commands(spec, database, images, models):
    """Build explicit commands so the submitted config fixes CPU/GPU behavior."""
    exe=spec.get("colmap_binary","colmap")
    use_gpu="1" if spec.get("colmap_use_gpu",False) else "0"
    return [
      [exe,"feature_extractor","--database_path",str(database),"--image_path",str(images),
       "--ImageReader.camera_model","PINHOLE","--FeatureExtraction.use_gpu",use_gpu],
      [exe,"exhaustive_matcher","--database_path",str(database),
       "--FeatureMatching.use_gpu",use_gpu],
      [exe,"mapper","--database_path",str(database),"--image_path",str(images),"--output_path",str(models)]]


def initialize(spec,manifest,prepared):
    import pycolmap
    prepared=Path(prepared).resolve()
    pins=read_json(Path(__file__).resolve().parents[1]/"upstream.lock.json")
    actual={p.relative_to(prepared/"images").as_posix() for p in (prepared/"images").rglob("*") if p.is_file()}
    if actual!=set(manifest["train"]): raise WorkflowError("Initializer input is not exactly the training split")
    policy=None
    if spec["initializer"]=="colmap":
        policy={"feature_and_matching_gpu":bool(spec.get("colmap_use_gpu",False)),
                "matcher":"exhaustive","camera_model":"PINHOLE"}
        attempt=prepared/("colmap_attempt_"+uuid.uuid4().hex[:10]); attempt.mkdir()
        database=attempt/"database.db"; models=attempt/"models"; models.mkdir()
        commands=_colmap_commands(spec,database,prepared/"images",models)
        for command in commands:
            print("COLMAP:",command,flush=True)
            subprocess.run(command,check=True)
        candidates=[]
        for model in sorted(models.iterdir()):
            if model.is_dir():
                reconstruction=pycolmap.Reconstruction(str(model))
                candidates.append((len(reconstruction.reg_image_ids()),model))
        if not candidates: raise WorkflowError("COLMAP produced no registered model")
        source=max(candidates,key=lambda pair:pair[0])[1]
        source_model=source
    elif spec["initializer"]=="vggt":
        policy=_vggt_policy(spec)
        root=Path(spec["vggt_root"]).resolve()
        if file_hash(root/"demo_colmap.py")!=pins["vggt"]["files"]["demo_colmap.py"]:
            raise WorkflowError("VGGT demo differs from pinned source")
        if any(Path(n).parent!=Path(".") for n in manifest["train"]):
            raise WorkflowError("Pinned VGGT demo expects flat image names; explicitly preprocess names and references first")
        attempt=prepared/("vggt_attempt_"+uuid.uuid4().hex[:10]); attempt.mkdir()
        (attempt/"images").symlink_to(prepared/"images",target_is_directory=True)
        demo_args=["--scene_dir",str(attempt),"--seed",str(spec["seed"]),
                   "--conf_thres_value",str(policy["confidence_threshold"])]
        if policy["bundle_adjustment"]:
            dinov2_root=_verified_dinov2_root(spec)
            _verify_ba_cache()
            command=[spec.get("vggt_python",spec["python"]),"-m","bgs.vggt_wrapper",
                     "--demo",str(root/"demo_colmap.py"),"--dinov2-root",str(dinov2_root),
                     *demo_args,"--use_ba"]
        else:
            command=[spec.get("vggt_python",spec["python"]),str(root/"demo_colmap.py"),*demo_args]
        repo_root=str(Path(__file__).resolve().parents[1])
        env=os.environ.copy()
        env["PYTHONPATH"]=repo_root+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.run(command,cwd=root,env=env,check=True)
        source_model=attempt/"sparse"
    else:
        raise WorkflowError("Unknown initializer")
    model=pycolmap.Reconstruction(str(source_model))
    registered={model.images[i].name for i in model.reg_image_ids()}
    if registered!=set(manifest["train"]):
        raise WorkflowError(f"Registration coverage {len(registered)}/{len(manifest['train'])}; missing={set(manifest['train'])-registered}")
    if len(model.points3D)<4: raise WorkflowError("Too few initialization points")
    sparse=prepared/"sparse"
    if sparse.exists():
        # Preserve incomplete output from an interrupted attempt; never mix it
        # into a successful rerun or delete historical evidence.
        sparse.rename(prepared/("orphan_sparse_"+uuid.uuid4().hex[:10]))
    dest=sparse/"0"; dest.parent.mkdir()
    shutil.copytree(source_model,dest)
    files={p.name:file_hash(p) for p in dest.iterdir() if p.is_file() and p.suffix in (".bin",".txt")}
    atomic_json(prepared/"init_receipt.json",{"initializer":spec["initializer"],"train_only":True,
        "registered_names":sorted(registered),"num_points":len(model.points3D),"model":str(dest),"files":files,
        "policy":policy,
        "scope":"official initialization baseline; no new reliability mechanism yet"})
