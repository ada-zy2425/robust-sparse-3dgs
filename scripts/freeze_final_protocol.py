#!/usr/bin/env python3
"""Freeze the finite final-scene matrix before any final manifest is scored."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from bgs.core import WorkflowError, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--matrix-id", default="mipnerf360_final_v1")
    parser.add_argument("--scene", action="append", required=True)
    args = parser.parse_args()

    output = Path(args.out).resolve()
    scenes = list(args.scene)
    if output.exists():
        raise WorkflowError(f"Refusing to overwrite frozen final protocol: {output}")
    if len(scenes) < 3 or len(scenes) != len(set(scenes)):
        raise WorkflowError("Final protocol requires at least three unique scenes")

    run_ids = []
    for scene in scenes:
        if not scene or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in scene):
            raise WorkflowError(f"Unsafe scene id: {scene!r}")
        run_ids.extend([
            f"{scene}_final_v24_colmap_s42_7k_v1",
            f"{scene}_final_v24_vggt_ba_s42_7k_v1",
        ])
    atomic_json(output, {
        "schema": 1,
        "status": "FROZEN_BEFORE_SCORING",
        "matrix_id": args.matrix_id,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scene_ids": scenes,
        "run_ids": run_ids,
        "partition": "final",
        "selection": "unseen scenes selected by category before any final RGB score",
        "fixed": {
            "train_views": 24,
            "holdout_rule": "every eighth ordered image",
            "steps": 7000,
            "diagnostic_steps": [1000, 3000, 7000],
            "max_edge": 1024,
            "seed": 42,
            "hardware_family": "NVIDIA A800-SXM4-80GB",
            "arms": ["COLMAP", "VGGT with official upstream bundle adjustment"],
        },
        "decision_rule": "report every predeclared run and failure; do not tune from final scores",
        "claim_scope": "external confirmation of a development-set diagnosis; BA is established upstream work",
    })
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
