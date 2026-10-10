#!/usr/bin/env python3
"""Validate a COLMAP-undistorted scene before immutable split creation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import pycolmap

from bgs.core import WorkflowError, atomic_json
from bgs.data import EXTENSIONS, _camera_model_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-edge", type=int, required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    image_root = Path(args.images).resolve()
    model_root = Path(args.model).resolve()
    files = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    reconstruction = pycolmap.Reconstruction(str(model_root))
    registered = {image.name: image for image in reconstruction.images.values()}
    names = {path.relative_to(image_root).as_posix() for path in files}
    if not files or names != set(registered):
        raise WorkflowError(
            f"Image/model name mismatch: files={len(files)} registered={len(registered)}"
        )

    models = set()
    max_width = max_height = 0
    for path in files:
        name = path.relative_to(image_root).as_posix()
        camera = reconstruction.cameras[registered[name].camera_id]
        model = _camera_model_name(camera)
        models.add(model)
        if model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise WorkflowError(f"Distorted camera model remains: {name}: {model}")
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise WorkflowError(f"Non-RGB image: {name}: {image.mode}")
            width, height = image.size
        if (width, height) != (int(camera.width), int(camera.height)):
            raise WorkflowError(f"Image/camera dimensions differ: {name}")
        if max(width, height) > args.max_edge:
            raise WorkflowError(f"Image exceeds max edge: {name}: {width}x{height}")
        max_width, max_height = max(max_width, width), max(max_height, height)

    receipt = {
        "schema": 1,
        "status": "PASSED",
        "images": len(files),
        "registered_images": len(registered),
        "camera_models": sorted(models),
        "maximum_width": max_width,
        "maximum_height": max_height,
        "max_edge_limit": args.max_edge,
        "images_root": str(image_root),
        "model_root": str(model_root),
    }
    if args.out:
        output = Path(args.out)
        if output.exists():
            raise WorkflowError(f"Refusing to overwrite validation receipt: {output}")
        atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    try:
        main()
    except WorkflowError as error:
        raise SystemExit(f"BLOCKED: {error}")
