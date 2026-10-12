#!/usr/bin/env python3
"""Collect a complete formal matrix, including failures, into JSON and CSV."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from bgs.core import WorkflowError, atomic_json, read_json


def maybe_json(path: Path):
    return read_json(path) if path.is_file() else {}


def mean_metrics(record):
    mean = record.get("mean", {}) if record else {}
    return mean.get("psnr"), mean.get("ssim"), mean.get("lpips")


def list_mean(record, key):
    values = record.get(key, [])
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--revision", required=True, type=int)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    state_path = Path(args.state).resolve()
    state = read_json(state_path)
    suffix = f"_v{args.revision}"
    tasks = {
        task_id: task for task_id, task in state.get("tasks", {}).items()
        if task_id.startswith("m360_") and task_id.endswith(suffix)
    }
    if len(tasks) != 18:
        raise WorkflowError(f"Expected 18 formal matrix tasks, found {len(tasks)}")

    rows = []
    for task_id, task in sorted(tasks.items()):
        config = read_json(task["config_path"])
        manifest = read_json(config["manifest"])
        run = Path(config["root"]) / "runs" / task_id
        init = maybe_json(run / "input" / "init_receipt.json")
        summary = maybe_json(run / "summary.json")
        metrics = maybe_json(run / "training" / "metrics.json")
        alignment = maybe_json(run / "training" / "alignment.json")
        final_psnr, final_ssim, final_lpips = mean_metrics(metrics)
        attempts = task.get("attempts", [])
        latest = attempts[-1] if attempts else {}
        row = {
            "run_id": task_id,
            "scene": manifest["scene_id"],
            "train_views": len(manifest["train"]),
            "holdout_views": len(manifest["holdout"]),
            "initializer": config["initializer"],
            "bundle_adjustment": bool(config.get("vggt_ba", False)),
            "steps": config["steps"],
            "status": task["status"],
            "failure_category": latest.get("category"),
            "attempts": len(attempts),
            "registered_views": len(init.get("registered_names", [])),
            "initial_points": init.get("num_points"),
            "psnr": final_psnr,
            "ssim": final_ssim,
            "lpips": final_lpips,
            "num_gaussians": metrics.get("num_gaussians"),
            "queue_gpu_hours": sum(float(item.get("gpu_hours", 0)) for item in attempts),
            "run_gpu_hours": summary.get("gpu_hours_known_attempts"),
            "end_to_end_s": summary.get("end_to_end_seconds_known_attempts"),
            "optimization_s": metrics.get("optimization_seconds_this_attempt"),
            "model_bytes": metrics.get("model_bytes", summary.get("model_bytes")),
            "render_fps_integration": metrics.get("render_fps", summary.get("render_fps")),
            "peak_allocated_bytes": metrics.get("peak_allocated_bytes", summary.get("peak_allocated_bytes")),
            "center_error_mean": list_mean(alignment, "center_residuals"),
            "orientation_error_degrees_mean": list_mean(alignment, "orientation_errors_degrees"),
        }
        for step in (1000, 3000, 7000):
            milestone = maybe_json(run / "training" / "milestones" / f"step_{step:08d}" / "metrics.json")
            psnr, ssim, lpips = mean_metrics(milestone)
            row.update({f"psnr_{step}": psnr, f"ssim_{step}": ssim, f"lpips_{step}": lpips})
        rows.append(row)

    result = {
        "schema": 1,
        "queue_state": str(state_path),
        "revision": args.revision,
        "budget_gpu_hours": state["budget_gpu_hours"],
        "charged_gpu_hours_at_collection": state["charged_gpu_hours"],
        "rows": rows,
    }
    json_out = Path(args.json_out).resolve()
    csv_out = Path(args.csv_out).resolve()
    for output in (json_out, csv_out):
        if output.exists():
            raise WorkflowError(f"Refusing to overwrite result artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(json_out, result)
    with csv_out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_out), "csv": str(csv_out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
