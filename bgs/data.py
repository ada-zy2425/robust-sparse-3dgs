"""Immutable input splits and training-only image staging."""
from __future__ import annotations
import math
from pathlib import Path, PurePosixPath
import shutil

from .core import WorkflowError, atomic_json, read_json, file_hash, digest

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _camera_model_name(camera):
    """Return a stable COLMAP model name across pycolmap API versions."""
    kind = getattr(camera, "model_name", None)
    if kind is None:
        kind = getattr(camera, "model", None)
    kind = kind() if callable(kind) else kind
    kind = getattr(kind, "name", kind)
    if kind is None:
        raise WorkflowError("COLMAP camera does not expose its model name")
    # Some enum bindings stringify as ``CameraModelId.PINHOLE``.
    return str(kind).rsplit(".", 1)[-1]


def validate_camera(camera, name):
    import numpy as np
    K = np.asarray(camera["K"], dtype=float)
    pose = np.asarray(camera["c2w"], dtype=float)
    if K.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(pose).all():
        raise WorkflowError(f"Invalid camera matrix: {name}")
    R = pose[:3, :3]
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-4)):
        raise WorkflowError(f"Require PINHOLE K and a rigid right-handed c2w pose: {name}")
    if any(not isinstance(camera[k], int) or camera[k] <= 0 for k in ("width", "height")):
        raise WorkflowError(f"Invalid image dimensions: {name}")


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ("..", ".") for x in p.parts) or "\\" in name:
        raise WorkflowError(f"Unsafe relative image name: {name!r}")
    return name


def split_names(names, n_train, test_every=8):
    names = sorted(names)
    if len(names) != len(set(names)) or test_every < 2 or n_train < 3:
        raise WorkflowError("Need unique images, >=3 train views and test_every>=2")
    holdout = [n for i, n in enumerate(names) if i % test_every == 0]
    pool = [n for i, n in enumerate(names) if i % test_every != 0]
    if len(pool) < n_train or not holdout:
        raise WorkflowError("Insufficient images for the requested split")
    # Nested order: endpoints then the largest filename-index gaps. No RGB or
    # held-out metrics enter view selection. Suitable starting point for ordered captures.
    selected = [0]
    while len(selected) < n_train:
        remaining = [i for i in range(len(pool)) if i not in selected]
        selected.append(max(remaining, key=lambda i: (min(abs(i-j) for j in selected), -i)))
    return [pool[i] for i in selected], holdout


def validate_manifest(m):
    if m.get("schema") != 1 or m.get("partition") not in ("development","final"):
        raise WorkflowError("Manifest partition must be development or final")
    if m["partition"]=="final":
        freeze=m.get("final_freeze",{})
        required=("matrix_id","frozen_at","protocol_sha256")
        if freeze.get("status")!="FROZEN_BEFORE_SCORING" or any(not freeze.get(k) for k in required):
            raise WorkflowError("Final manifests require a frozen pre-scoring protocol receipt")
    train, holdout = m.get("train", []), m.get("holdout", [])
    if len(train) < 3 or not holdout or len(train) != len(set(train)) or len(holdout) != len(set(holdout)):
        raise WorkflowError("Invalid/duplicate/empty split")
    if set(train) & set(holdout):
        raise WorkflowError("Train/holdout overlap")
    for n in train + holdout:
        safe_name(n)
        if n not in m["images"]:
            raise WorkflowError(f"Missing manifest entry: {n}")
    return m


def prepare(images, reference, output, n_train=6, test_every=8, scene_id="scene0",
            partition="development", final_protocol=None):
    from PIL import Image
    images, reference, output = Path(images).resolve(), Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise WorkflowError("Refusing to overwrite a manifest; choose a new output")
    refs = read_json(reference)
    if refs.get("schema") != 1 or not refs.get("source"):
        raise WorkflowError("Reference JSON needs schema=1 and an explicit source description")
    names = sorted(p.relative_to(images).as_posix() for p in images.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    train, holdout = split_names(names, n_train, test_every)
    rows = {}
    for name in train + holdout:
        safe_name(name)
        camera = refs["cameras"].get(name)
        if camera is None:
            raise WorkflowError(f"No reference camera for {name}")
        validate_camera(camera, name)
        with Image.open(images / name) as im:
            width, height = im.size
            if im.mode != "RGB":
                raise WorkflowError(f"Require explicit RGB preprocessing (no silent alpha/grayscale handling): {name}")
        if [width, height] != [camera["width"], camera["height"]]:
            raise WorkflowError(f"Image/camera dimensions differ: {name}; resize intrinsics consistently")
        rows[name] = {"sha256": file_hash(images/name), "width": width, "height": height}
    if partition not in ("development","final"):
        raise WorkflowError("partition must be development or final")
    m = {"schema": 1, "scene_id": scene_id, "partition": partition,
         "image_root": str(images), "reference": str(reference),
         "reference_sha256": file_hash(reference), "reference_source": refs["source"],
         "train": train, "holdout": holdout, "images": rows,
         "selection": "nested_farthest_filename_index", "test_every": test_every}
    if partition=="final":
        if final_protocol is None:raise WorkflowError("Final preparation requires --final-protocol")
        protocol_path=Path(final_protocol).resolve();protocol=read_json(protocol_path)
        if (protocol.get("schema")!=1 or protocol.get("status")!="FROZEN_BEFORE_SCORING"
                or scene_id not in protocol.get("scene_ids",[]) or not protocol.get("run_ids")):
            raise WorkflowError("Final protocol must predeclare this scene and a finite run matrix")
        m["final_protocol"]=str(protocol_path)
        m["final_freeze"]={"status":protocol["status"],"matrix_id":protocol["matrix_id"],
                           "frozen_at":protocol["frozen_at"],"protocol_sha256":file_hash(protocol_path)}
    validate_manifest(m)
    atomic_json(output, m)
    return m


def verify_inputs(m):
    validate_manifest(m)
    if file_hash(m["reference"]) != m["reference_sha256"]:
        raise WorkflowError("Reference cameras changed")
    if m["partition"]=="final" and file_hash(m["final_protocol"])!=m["final_freeze"]["protocol_sha256"]:
        raise WorkflowError("Frozen final protocol changed")
    root = Path(m["image_root"])
    for name in m["train"] + m["holdout"]:
        if file_hash(root/name) != m["images"][name]["sha256"]:
            raise WorkflowError(f"Input image changed: {name}")


def validate_final_run(m,run_id):
    """Reject any final-scene run not named before the first final score."""
    if m["partition"]!="final":return
    if file_hash(m["final_protocol"])!=m["final_freeze"]["protocol_sha256"]:
        raise WorkflowError("Frozen final protocol changed")
    protocol=read_json(m["final_protocol"])
    if run_id not in protocol.get("run_ids",[]):
        raise WorkflowError(f"Run {run_id} was not predeclared in the frozen final matrix")


def stage_training(m, destination):
    destination = Path(destination).resolve()
    imgdir = destination / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    allowed = set(m["train"])
    existing = {p.relative_to(imgdir).as_posix() for p in imgdir.rglob("*") if p.is_file()}
    if existing - allowed:
        raise WorkflowError(f"Unexpected images in initialization directory: {existing-allowed}")
    for name in m["train"]:
        src, dst = (Path(m["image_root"])/name).resolve(), imgdir/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or not dst.is_file() or file_hash(dst) != m["images"][name]["sha256"]:
                raise WorkflowError(f"Existing staging image differs from the manifest: {dst}")
        else:
            # COLMAP 3.13 canonicalizes symlink targets into image names, which
            # can escape the train-only staging tree and break name contracts.
            # A private copy keeps model names stable and input isolation clear.
            shutil.copy2(src,dst)
    atomic_json(destination/"staging.json", {"manifest_hash": digest(m), "names": m["train"],
                "train_only": True, "image_root": m["image_root"], "materialization":"copy"})


def export_colmap_reference(model, output, source):
    """Exports calibrated evaluation reference only; never initializes GS points."""
    import numpy as np
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(model))
    records = {}
    for image in reconstruction.images.values():
        transform = image.cam_from_world
        transform = transform() if callable(transform) else transform
        mat = np.eye(4)
        mat[:3] = np.asarray(transform.matrix())
        camera = reconstruction.cameras[image.camera_id]
        kind = _camera_model_name(camera)
        if kind not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise WorkflowError("Use a COLMAP-undistorted PINHOLE reference and matching images; distorted evaluation is not silently approximated")
        records[safe_name(image.name)] = {"K": np.asarray(camera.calibration_matrix()).tolist(),
                 "c2w": np.linalg.inv(mat).tolist(), "width": int(camera.width), "height": int(camera.height)}
        validate_camera(records[image.name], image.name)
    if not records:
        raise WorkflowError("No registered reference cameras")
    atomic_json(output, {"schema": 1, "source": source, "cameras": records})
