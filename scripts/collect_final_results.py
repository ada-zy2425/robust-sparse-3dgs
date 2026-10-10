#!/usr/bin/env python3
"""Collect the complete frozen final matrix, retaining terminal failures."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from bgs.core import WorkflowError, atomic_json, file_hash, read_json


def maybe_json(path: Path) -> dict:
    return read_json(path) if path.is_file() else {}


def list_mean(record: dict, key: str):
    values = record.get(key, [])
    return statistics.fmean(values) if values else None


def registration_failure(run: Path) -> tuple[int | None, int | None, str | None]:
    log = run / "logs" / "initialize.01.log"
    if not log.is_file():
        return None, None, None
    blocked = [line.strip() for line in log.read_text(errors="replace").splitlines()
               if line.startswith("BLOCKED:")]
    reason = blocked[-1] if blocked else None
    match = re.search(r"Registration coverage (\d+)/(\d+)", reason or "")
    return ((int(match.group(1)), int(match.group(2)), reason) if match
            else (None, None, reason))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    state = read_json(Path(args.state).resolve())
    protocol_path = Path(args.protocol).resolve()
    protocol = read_json(protocol_path)
    expected = list(protocol.get("run_ids", []))
    if protocol.get("status") != "FROZEN_BEFORE_SCORING" or len(expected) < 6:
        raise WorkflowError("Invalid frozen final protocol")
    tasks = state.get("tasks", {})
    missing = sorted(set(expected) - set(tasks))
    if missing:
        raise WorkflowError(f"Final queue is missing tasks: {missing}")
    incomplete = sorted(run_id for run_id in expected
                        if tasks[run_id].get("status") not in {"SUCCEEDED", "FAILED"})
    if incomplete:
        raise WorkflowError(f"Final tasks are not terminal: {incomplete}")

    rows = []
    for run_id in expected:
        task = tasks[run_id]
        config = read_json(Path(task["config_path"]))
        manifest = read_json(Path(config["manifest"]))
        run = Path(config["root"]) / "runs" / run_id
        init = maybe_json(run / "input" / "init_receipt.json")
        summary = maybe_json(run / "summary.json")
        metrics = maybe_json(run / "training" / "metrics.json")
        alignment = maybe_json(run / "training" / "alignment.json")
        attempts = task.get("attempts", [])
        latest = attempts[-1] if attempts else {}
        registered, requested, failure_reason = registration_failure(run)
        registered = len(init.get("registered_names", [])) if init else registered
        requested = len(manifest["train"]) if init else requested
        mean = metrics.get("mean", {})
        model = run / "training" / "model.ply"
        row = {
            "run_id": run_id,
            "scene": manifest["scene_id"],
            "category": manifest.get("category"),
            "initializer": "vggt_official_ba" if config.get("vggt_ba") else config["initializer"],
            "train_views": len(manifest["train"]),
            "holdout_views": len(manifest["holdout"]),
            "status": task["status"],
            "attempts": len(attempts),
            "failure_category": latest.get("category"),
            "failure_reason": failure_reason,
            "registered_views": registered,
            "requested_views": requested,
            "initial_points": init.get("num_points"),
            "psnr": mean.get("psnr"),
            "ssim": mean.get("ssim"),
            "lpips": mean.get("lpips"),
            "num_gaussians": metrics.get("num_gaussians"),
            "optimization_s": metrics.get("optimization_seconds_this_attempt"),
            "end_to_end_s": summary.get("end_to_end_seconds_known_attempts"),
            "queue_gpu_hours": sum(float(attempt.get("gpu_hours", 0)) for attempt in attempts),
            "peak_allocated_bytes": metrics.get("peak_allocated_bytes", summary.get("peak_allocated_bytes")),
            "model_bytes": model.stat().st_size if model.is_file() else None,
            "model_sha256": file_hash(model) if model.is_file() else None,
            "metrics_sha256": file_hash(run / "training" / "metrics.json") if metrics else None,
            "center_error_mean": list_mean(alignment, "center_residuals"),
            "orientation_error_degrees_mean": list_mean(alignment, "orientation_errors_degrees"),
            "config_sha256": task.get("config_sha256"),
            "identity_hash": latest.get("identity_hash"),
        }
        for step in (1000, 3000, 7000):
            milestone = maybe_json(run / "training" / "milestones" / f"step_{step:08d}" / "metrics.json")
            milestone_mean = milestone.get("mean", {})
            row.update({f"psnr_{step}": milestone_mean.get("psnr"),
                        f"ssim_{step}": milestone_mean.get("ssim"),
                        f"lpips_{step}": milestone_mean.get("lpips")})
        rows.append(row)

    result = {
        "schema": 1,
        "protocol_sha256": file_hash(protocol_path),
        "budget_gpu_hours": state["budget_gpu_hours"],
        "charged_gpu_hours_at_collection": state["charged_gpu_hours"],
        "rows": rows,
    }
    json_out, csv_out = Path(args.json_out).resolve(), Path(args.csv_out).resolve()
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
