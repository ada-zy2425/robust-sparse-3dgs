"""Run the pinned VGGT demo while resolving pinned DINOv2 code locally.

VGGT's optional bundle-adjustment path asks torch.hub for
``facebookresearch/dinov2`` even when the source and weights are already
cached.  The compute nodes are intentionally not assumed to have outbound
network access, so this wrapper redirects only that exact repository lookup.
"""
from pathlib import Path
import argparse
import runpy
import sys


DINOV2_HUB_REPOSITORY = "facebookresearch/dinov2"


def local_hub_loader(original_load, dinov2_root):
    root = str(Path(dinov2_root).resolve())

    def load(repo_or_dir, *args, **kwargs):
        if repo_or_dir == DINOV2_HUB_REPOSITORY:
            kwargs["source"] = "local"
            return original_load(root, *args, **kwargs)
        return original_load(repo_or_dir, *args, **kwargs)

    return load


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True)
    parser.add_argument("--dinov2-root", required=True)
    args, passthrough = parser.parse_known_args(argv)
    demo = Path(args.demo).resolve()
    dinov2_root = Path(args.dinov2_root).resolve()
    if not demo.is_file():
        raise SystemExit(f"VGGT demo does not exist: {demo}")
    if not (dinov2_root / "hubconf.py").is_file():
        raise SystemExit(f"DINOv2 hubconf does not exist: {dinov2_root}")

    import torch

    original_load = torch.hub.load
    torch.hub.load = local_hub_loader(original_load, dinov2_root)
    old_argv = sys.argv
    try:
        sys.argv = [str(demo), *passthrough]
        runpy.run_path(str(demo), run_name="__main__")
    finally:
        sys.argv = old_argv
        torch.hub.load = original_load


if __name__ == "__main__":
    main()
