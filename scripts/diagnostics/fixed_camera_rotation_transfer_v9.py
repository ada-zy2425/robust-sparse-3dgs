#!/usr/bin/env python3
"""Garden v9 A/C fixed-camera runner around the unchanged pinned trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


class ContractError(RuntimeError):
    pass


ARMS = {"A_original_perturbed_fixed", "C_ba_rotation_input_center_fixed"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pose_sha256(value) -> str:
    a = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object: {path}")
    return value


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def array_state_sha256(mapping) -> str:
    h = hashlib.sha256()
    for name in sorted(mapping):
        value = mapping[name]
        array = value.detach().cpu().contiguous().numpy() if hasattr(value, "detach") else np.asarray(value)
        array = np.ascontiguousarray(array)
        h.update(name.encode() + b"\0" + str(array.dtype).encode() + b"\0")
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def ordered_intrinsics(parser) -> np.ndarray:
    if not hasattr(parser, "camera_ids") or not hasattr(parser, "Ks_dict"):
        raise ContractError("Pinned Parser intrinsics contract changed")
    values = [np.asarray(parser.Ks_dict[camera_id], dtype=np.float64) for camera_id in parser.camera_ids]
    result = np.stack(values)
    if result.shape != (len(parser.image_names), 3, 3) or not np.isfinite(result).all():
        raise ContractError("Invalid ordered intrinsics")
    return result


def validate_poses(names, poses, tol):
    if poses.shape != (len(names), 4, 4) or not np.isfinite(poses).all() or len(names) != len(set(names)):
        raise ContractError("Invalid camera override array")
    rotations = poses[:, :3, :3]
    checks = {
        "homogeneous_row_max_abs_error": float(np.max(np.abs(poses[:, 3, :] - np.array([0., 0., 0., 1.])))),
        "rotation_orthogonality_max_abs_error": float(np.max(np.abs(rotations @ np.swapaxes(rotations, 1, 2) - np.eye(3)))),
        "rotation_determinant_max_abs_error": float(np.max(np.abs(np.linalg.det(rotations) - 1.))),
        "c2w_w2c_roundtrip_max_abs_error": float(np.max(np.abs(np.linalg.inv(np.linalg.inv(poses)) - poses))),
    }
    limits = {
        "homogeneous_row_max_abs_error": tol["homogeneous_row_max_abs"],
        "rotation_orthogonality_max_abs_error": tol["rotation_orthogonality_max_abs"],
        "rotation_determinant_max_abs_error": tol["rotation_determinant_max_abs"],
        "c2w_w2c_roundtrip_max_abs_error": tol["c2w_w2c_roundtrip_max_abs"],
    }
    for key, value in checks.items():
        if value > limits[key]:
            raise ContractError(f"{key}={value} exceeds {limits[key]}")
    return checks


def static_validate(path: Path):
    path = path.resolve(); cfg = read(path)
    if cfg.get("schema") != 1 or cfg.get("protocol_id") != "rotation_transfer_cross_scene_confirmation_v9":
        raise ContractError("Wrong v9 task identity")
    if cfg.get("arm") not in ARMS or cfg.get("status") != "PREPARED_NOT_STARTED":
        raise ContractError("Wrong arm or pre-submission status")
    if cfg.get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise ContractError("Runner hash mismatch")
    spec = cfg["spec"]
    required = {"steps": 7000, "max_edge": 1024, "seed": 42, "pose_opt": False,
                "densification_policy": "default", "cache_images": False,
                "rotation_degrees": 1.0, "translation_fraction": 0.0,
                "splat_numerics_policy": "dtype_scene_scale_floor_v1"}
    if any(spec.get(k) != v for k, v in required.items()):
        raise ContractError("Frozen training spec changed")
    if spec.get("noise_seed") != cfg.get("perturbation_seed"):
        raise ContractError("Perturbation seed mismatch")
    project = Path(cfg["project_root"]).resolve()
    manifest_path = Path(spec["manifest"]).resolve(); manifest = read(manifest_path)
    if sha256(manifest_path) != cfg["identities"]["manifest_sha256"] or manifest.get("partition") != "development":
        raise ContractError("Manifest identity mismatch")
    if len(manifest["train"]) != 24 or len(manifest["holdout"]) != 24:
        raise ContractError("Frozen Garden denominator changed")
    cpu_path = Path(cfg["cpu_admission"]["path"]).resolve()
    if sha256(cpu_path) != cfg["cpu_admission"]["sha256"] or read(cpu_path).get("gpu_admission_pass") is not True:
        raise ContractError("Seed CPU admission is absent or failed")
    source = Path(cfg["source_input"]["path"]).resolve()
    for name, expected in (("init_receipt.json", cfg["source_input"]["init_receipt_sha256"]),
                           ("staging.json", cfg["source_input"]["staging_sha256"])):
        if sha256(source / name) != expected:
            raise ContractError(f"Source receipt mismatch: {name}")
    init = read(source / "init_receipt.json")
    if init.get("train_only") is not True or set(init.get("registered_names", [])) != set(manifest["train"]):
        raise ContractError("Source input is not complete train-only Garden")
    model = source / "sparse" / "0"
    for name, expected in cfg["source_input"]["model_file_sha256"].items():
        if sha256(model / name) != expected:
            raise ContractError(f"Source model changed: {name}")
    for name in manifest["train"]:
        if sha256(source / "images" / name) != manifest["images"][name]["sha256"]:
            raise ContractError(f"Training image changed: {name}")
    for relative, expected in cfg["identities"]["core_source_sha256"].items():
        if sha256(project / relative) != expected:
            raise ContractError(f"Core source changed: {relative}")
    if sha256(project / "upstream.lock.json") != cfg["identities"]["upstream_lock_sha256"]:
        raise ContractError("Upstream lock changed")
    override_path = Path(cfg["camera_override"]["path"]).resolve()
    if sha256(override_path) != cfg["camera_override"]["sha256"]:
        raise ContractError("Camera override identity mismatch")
    override = read(override_path); names = [str(x) for x in override["names"]]
    poses = np.asarray(override["poses_c2w"], dtype=np.float64)
    if override.get("arm") != cfg["arm"] or override.get("seed") != cfg["perturbation_seed"]:
        raise ContractError("Override arm/seed mismatch")
    if override.get("convention") != "OpenCV c2w" or override.get("application") != "replace ordered training c2w once after normal Runner construction":
        raise ContractError("Camera convention/application mismatch")
    if pose_sha256(poses) != override["pose_sha256"] or set(names) != set(manifest["train"]):
        raise ContractError("Override pose or names mismatch")
    numeric = validate_poses(names, poses, cfg["numeric_tolerance"])
    if override.get("source_solver_usable") is not True:
        raise ContractError("Source standard BA is not usable")
    if override.get("anchor_equals_A_max_abs", 1.0) > cfg["numeric_tolerance"]["anchor_c2w_max_abs"]:
        raise ContractError("Camera 0 anchor differs")
    return {"schema": 1, "status": "PASS_STATIC_CPU_GATE", "task_id": cfg["task_id"],
            "arm": cfg["arm"], "seed": cfg["perturbation_seed"], "config_sha256": sha256(path),
            "manifest_train_count": 24, "source_point_count": init["num_points"],
            "override_pose_sha256": override["pose_sha256"], "numeric": numeric,
            "camera_frozen": True, "solver_usable": True, "finite": True}


def apply_once(runner, cfg, receipt_path):
    if getattr(runner, "_v9_camera_override_application_count", 0) != 0 or bool(runner.cfg.pose_opt):
        raise ContractError("Camera override repeated or pose optimizer enabled")
    override = read(Path(cfg["camera_override"]["path"]))
    names = [str(x) for x in override["names"]]
    if names != [str(x) for x in runner.parser.image_names]:
        raise ContractError("Ordered camera names differ")
    poses = np.asarray(override["poses_c2w"], dtype=np.float64)
    generated = np.ascontiguousarray(np.asarray(runner.parser.camtoworlds, dtype=np.float64))
    if pose_sha256(generated) != override["A_pose_sha256"]:
        raise ContractError("Runtime perturbation differs from preregistered A")
    splat_before = array_state_sha256(runner.splats)
    intr_before = pose_sha256(ordered_intrinsics(runner.parser))
    valset_id = id(runner.valset)
    runner.parser.camtoworlds = poses.copy()
    runner._perturbed_training_poses = poses.copy()
    runner._v9_camera_override_application_count = 1
    splat_after = array_state_sha256(runner.splats); intr_after = pose_sha256(ordered_intrinsics(runner.parser))
    if splat_before != splat_after or intr_before != intr_after or id(runner.valset) != valset_id:
        raise ContractError("Non-camera state changed during override")
    receipt = {"schema": 1, "status": "PASS_RUNTIME_WIRING_GATE", "task_id": cfg["task_id"],
               "arm": cfg["arm"], "seed": cfg["perturbation_seed"], "application_count": 1,
               "names": names, "generated_perturbed_pose_sha256": pose_sha256(generated),
               "applied_pose_sha256": pose_sha256(poses),
               "anchor_max_abs_change": float(np.max(np.abs(poses[0] - generated[0]))),
               "gaussian_state_sha256_before": splat_before, "gaussian_state_sha256_after": splat_after,
               "gaussian_state_unchanged_during_override": True, "gaussian_count": int(len(runner.splats["means"])),
               "intrinsics_sha256_before": intr_before, "intrinsics_sha256_after": intr_after,
               "intrinsics_unchanged": True, "heldout_dataset_identity_unchanged": True,
               "pose_optimizer_enabled": False,
               "source_model_file_sha256": cfg["source_input"]["model_file_sha256"]}
    write(receipt_path, receipt)
    return receipt


def execute(config_path: Path):
    static = static_validate(config_path); cfg = read(config_path.resolve())
    project = Path(cfg["project_root"]).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from bgs.data import validate_manifest
    from bgs.core import read_json
    import bgs.train_adapter as adapter
    manifest = validate_manifest(read_json(cfg["spec"]["manifest"]))
    output = Path(cfg["output_dir"]).resolve(); output.mkdir(parents=True, exist_ok=True)
    experiment = output / "v9_experiment.json"
    identity = {"schema": 1, "task_config": cfg, "task_config_sha256": sha256(config_path.resolve()), "static_gate": static}
    if experiment.exists() and read(experiment) != identity:
        raise ContractError("Output identity conflict")
    write(experiment, identity)
    original = adapter.make_runner; runtime = {}
    def wrapped(spec, manifest_arg, training_dir, output_arg, viewer=False, read_only=False):
        runner, module = original(spec, manifest_arg, training_dir, output_arg, viewer=viewer, read_only=read_only)
        apply_once(runner, cfg, Path(output_arg) / "camera_override_receipt.json")
        runtime["runner"] = runner
        return runner, module
    adapter.make_runner = wrapped
    try:
        adapter.execute(cfg["spec"], manifest, Path(cfg["source_input"]["path"]), output, viewer=False)
        runner = runtime.get("runner")
        if runner is None:
            raise ContractError("Runner was not captured")
        override = read(Path(cfg["camera_override"]["path"]))
        terminal = np.ascontiguousarray(np.asarray(runner.parser.camtoworlds, dtype=np.float64))
        expected = np.asarray(override["poses_c2w"], dtype=np.float64)
        if pose_sha256(terminal) != override["pose_sha256"] or getattr(runner, "_v9_camera_override_application_count", 0) != 1:
            raise ContractError("Fixed camera terminal contract failed")
        write(output / "terminal_camera_freeze_receipt.json", {
            "schema": 1, "status": "PASS_TERMINAL_CAMERA_FREEZE_GATE", "task_id": cfg["task_id"],
            "arm": cfg["arm"], "seed": cfg["perturbation_seed"], "application_count": 1,
            "expected_pose_sha256": override["pose_sha256"], "terminal_raw_parser_pose_sha256": pose_sha256(terminal),
            "terminal_camera_max_abs_change": float(np.max(np.abs(terminal - expected))),
            "pose_optimizer_enabled": bool(runner.cfg.pose_opt)})
    finally:
        adapter.make_runner = original


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True, type=Path)
    p.add_argument("--validate-only", action="store_true"); p.add_argument("--receipt", type=Path)
    args = p.parse_args()
    if args.validate_only:
        result = static_validate(args.config)
        if args.receipt:
            if args.receipt.exists():
                raise ContractError("Refusing to overwrite validation receipt")
            write(args.receipt, result)
        print(json.dumps(result, sort_keys=True)); return 0
    execute(args.config); return 0


if __name__ == "__main__":
    raise SystemExit(main())
