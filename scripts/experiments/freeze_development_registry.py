#!/usr/bin/env python3
"""Freeze a development-scene registry after checking nested split contracts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from bgs.core import WorkflowError, atomic_json, file_hash, read_json
from bgs.data import validate_manifest, verify_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--archive-bytes", required=True, type=int)
    parser.add_argument("--archive-md5", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--license-note", required=True)
    parser.add_argument(
        "--scene", action="append", required=True,
        help="scene_id:category:reference:manifest6:manifest12:manifest24",
    )
    args = parser.parse_args()

    output = Path(args.out).resolve()
    if output.exists():
        raise WorkflowError(f"Refusing to overwrite frozen registry: {output}")
    scenes = []
    seen = set()
    for value in args.scene:
        fields = value.split(":", 5)
        if len(fields) != 6:
            raise WorkflowError(f"Invalid --scene record: {value}")
        scene_id, category, reference, manifest6, manifest12, manifest24 = fields
        if scene_id in seen:
            raise WorkflowError(f"Duplicate scene id: {scene_id}")
        seen.add(scene_id)
        reference_path = Path(reference).resolve()
        manifests = {
            "6": Path(manifest6).resolve(),
            "12": Path(manifest12).resolve(),
            "24": Path(manifest24).resolve(),
        }
        records = {}
        for count, path in manifests.items():
            record = validate_manifest(read_json(path))
            verify_inputs(record)
            if (
                record["scene_id"] != scene_id
                or record["partition"] != "development"
                or len(record["train"]) != int(count)
                or Path(record["reference"]).resolve() != reference_path
            ):
                raise WorkflowError(f"Manifest contract mismatch: {path}")
            records[count] = record
        if not (
            set(records["6"]["train"]) < set(records["12"]["train"])
            < set(records["24"]["train"])
        ):
            raise WorkflowError(f"Train splits are not strictly nested: {scene_id}")
        if not (records["6"]["holdout"] == records["12"]["holdout"] == records["24"]["holdout"]):
            raise WorkflowError(f"Holdout sets differ across view counts: {scene_id}")
        scenes.append({
            "scene_id": scene_id,
            "category": category,
            "reference": str(reference_path),
            "reference_sha256": file_hash(reference_path),
            "manifests": {count: str(path) for count, path in manifests.items()},
            "manifest_sha256": {count: file_hash(path) for count, path in manifests.items()},
            "holdout_count": len(records["6"]["holdout"]),
        })

    registry = {
        "schema": 1,
        "status": "FROZEN_DEVELOPMENT",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": args.source_url,
            "archive_bytes": args.archive_bytes,
            "archive_md5": args.archive_md5,
            "archive_sha256": args.archive_sha256,
            "license_note": args.license_note,
            "rgb_distribution": "private_only",
        },
        "selection": "predeclared category coverage; no held-out RGB or score used",
        "development": scenes,
    }
    atomic_json(output, registry)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
