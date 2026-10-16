#!/usr/bin/env python3
"""Build C cameras: standard-BA c2w rotations plus A world-space centers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pose_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def read_override(path: Path) -> tuple[dict, list[str], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [str(name) for name in payload["names"]]
    poses = np.asarray(payload["poses_c2w"], dtype=np.float64)
    if len(names) != len(set(names)) or poses.shape != (len(names), 4, 4):
        raise ValueError(f"invalid camera override: {path}")
    if not np.isfinite(poses).all():
        raise ValueError(f"non-finite camera override: {path}")
    return payload, names, poses


def validate_c2w(poses: np.ndarray, tolerance: float = 1e-10) -> None:
    rotations = poses[:, :3, :3]
    if np.max(np.abs(poses[:, 3] - np.asarray([0.0, 0.0, 0.0, 1.0]))) > tolerance:
        raise ValueError("invalid homogeneous row")
    if np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3))) > tolerance:
        raise ValueError("rotation is not orthogonal")
    if np.max(np.abs(np.linalg.det(rotations) - 1.0)) > tolerance:
        raise ValueError("rotation determinant is not one")


def build_rotation_only(a_names: list[str], a: np.ndarray, b_names: list[str], b: np.ndarray) -> np.ndarray:
    if a_names != b_names:
        raise ValueError("A/BA ordered camera identities differ")
    validate_c2w(a)
    validate_c2w(b)
    c = a.copy()
    c[:, :3, :3] = b[:, :3, :3]
    validate_c2w(c)
    return c


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="A perturbed OpenCV c2w override JSON")
    parser.add_argument("--ba", required=True, type=Path, help="standard-BA OpenCV c2w override JSON")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    _, a_names, a = read_override(args.a)
    ba_payload, b_names, b = read_override(args.ba)
    c = build_rotation_only(a_names, a, b_names, b)
    result = {
        "schema": 1,
        "arm": "C_ba_rotation_input_center_fixed",
        "seed": args.seed,
        "names": a_names,
        "poses_c2w": c.tolist(),
        "convention": "OpenCV c2w",
        "application": "replace ordered training c2w once after normal Runner construction",
        "definition": "C[:3,:3] = standard-BA rotation; C[:3,3] = A world-space camera center",
        "reference_pose_used": False,
        "A_pose_sha256": pose_sha256(a),
        "B_pose_sha256": pose_sha256(b),
        "pose_sha256": pose_sha256(c),
        "C_rotation_equals_B_max_abs": float(np.max(np.abs(c[:, :3, :3] - b[:, :3, :3]))),
        "C_center_equals_A_max_abs": float(np.max(np.abs(c[:, :3, 3] - a[:, :3, 3]))),
        "anchor_equals_A_max_abs": float(np.max(np.abs(c[0] - a[0]))),
        "source_solver_usable": bool(ba_payload.get("source_solver_usable", False)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("pose_sha256", "C_rotation_equals_B_max_abs", "C_center_equals_A_max_abs", "anchor_equals_A_max_abs")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
