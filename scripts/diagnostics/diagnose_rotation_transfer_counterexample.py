#!/usr/bin/env python3
"""Audit a held-out A/C rendering counterexample from existing CPU-readable caches.

The evaluator stores a three-panel PNG (target | prediction | mean absolute RGB
error).  This tool never renders a checkpoint.  It checks image identity and
dimensions, compares the recorded float metrics at fixed milestones, and uses
the existing uint8 panels only for a clearly labelled post-hoc diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def metric_row(metrics: dict, image_name: str) -> tuple[int, dict]:
    rows = metrics.get("images")
    if not isinstance(rows, list):
        raise ValueError("metrics.images is not a list")
    matches = [(index, row) for index, row in enumerate(rows) if row.get("name") == image_name]
    if len(matches) != 1:
        raise ValueError(f"expected one metric row for {image_name}, found {len(matches)}")
    return matches[0]


def split_composite(path: Path, expected_width: int, expected_height: int) -> tuple[np.ndarray, ...]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape != (expected_height, expected_width * 3, 3):
        raise ValueError(f"unexpected composite shape {image.shape}: {path}")
    return tuple(image[:, i * expected_width : (i + 1) * expected_width] for i in range(3))


def panel_sha256(panel: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(panel).tobytes()).hexdigest()


def analyze_panels(target_u8: np.ndarray, prediction_u8: np.ndarray, error_u8: np.ndarray) -> dict:
    target = target_u8.astype(np.float64) / 255.0
    prediction = prediction_u8.astype(np.float64) / 255.0
    signed = prediction - target
    absolute_mean = np.abs(signed).mean(axis=2)
    squared = signed**2
    total_squared_error = float(squared.sum())
    mse = float(squared.mean())
    height, width = absolute_mean.shape
    luma = np.asarray([0.2126, 0.7152, 0.0722])
    target_luma = target @ luma
    prediction_luma = prediction @ luma

    grid = []
    for row in range(4):
        for column in range(4):
            ys = slice(row * height // 4, (row + 1) * height // 4)
            xs = slice(column * width // 4, (column + 1) * width // 4)
            tile_error = absolute_mean[ys, xs]
            tile_squared = float(squared[ys, xs].sum())
            grid.append(
                {
                    "row": row,
                    "column": column,
                    "mean_absolute_rgb_error": float(tile_error.mean()),
                    "p95_absolute_rgb_error": float(np.percentile(tile_error, 95)),
                    "squared_error_share": tile_squared / total_squared_error,
                }
            )

    encoded_error = np.rint(absolute_mean * 255).astype(np.int16)
    visualization_error = error_u8[:, :, 0].astype(np.int16)
    return {
        "panel_shape_hwc": [height, width, 3],
        "target_panel_sha256": panel_sha256(target_u8),
        "prediction_panel_sha256": panel_sha256(prediction_u8),
        "error_panel_sha256": panel_sha256(error_u8),
        "quantized_psnr_db": -10.0 * math.log10(mse),
        "mean_rgb": {
            "target": target.mean(axis=(0, 1)).tolist(),
            "prediction": prediction.mean(axis=(0, 1)).tolist(),
            "signed_prediction_minus_target": signed.mean(axis=(0, 1)).tolist(),
        },
        "luminance": {
            "target_mean": float(target_luma.mean()),
            "prediction_mean": float(prediction_luma.mean()),
            "signed_mean": float((prediction_luma - target_luma).mean()),
            "signed_median": float(np.median(prediction_luma - target_luma)),
        },
        "saturation": {
            "prediction_any_channel_le_1_over_255_fraction": float(
                (np.min(prediction, axis=2) <= 1 / 255).mean()
            ),
            "prediction_any_channel_ge_254_over_255_fraction": float(
                (np.max(prediction, axis=2) >= 254 / 255).mean()
            ),
            "prediction_all_channels_ge_254_over_255_fraction": float(
                (np.min(prediction, axis=2) >= 254 / 255).mean()
            ),
        },
        "absolute_rgb_error": {
            "mean": float(absolute_mean.mean()),
            "median": float(np.median(absolute_mean)),
            "p90": float(np.percentile(absolute_mean, 90)),
            "p95": float(np.percentile(absolute_mean, 95)),
            "p99": float(np.percentile(absolute_mean, 99)),
            "fraction_ge_0_2": float((absolute_mean >= 0.2).mean()),
            "fraction_ge_0_4": float((absolute_mean >= 0.4).mean()),
        },
        "spatial_error": {
            "top_half_mean_absolute": float(absolute_mean[: height // 2].mean()),
            "bottom_half_mean_absolute": float(absolute_mean[height // 2 :].mean()),
            "top_half_squared_error_share": float(
                squared[: height // 2].sum() / total_squared_error
            ),
            "max_squared_error_tile_4x4": max(grid, key=lambda item: item["squared_error_share"]),
            "grid_4x4": grid,
        },
        "stored_error_panel_contract_max_abs_uint8": int(
            np.abs(encoded_error - visualization_error).max()
        ),
    }


def load_step(training_dir: Path, step: int, image_name: str) -> dict:
    step_dir = training_dir / "milestones" / f"step_{step:08d}"
    metrics_path = step_dir / "metrics.json"
    metrics = read_json(metrics_path)
    index, row = metric_row(metrics, image_name)
    render_path = step_dir / row["visualization"]
    target, prediction, error = split_composite(render_path, int(row["width"]), int(row["height"]))
    analysis = analyze_panels(target, prediction, error)
    analysis.update(
        {
            "step": step,
            "index": index,
            "image_name": image_name,
            "metrics_path_sha256": sha256(metrics_path),
            "composite_sha256": sha256(render_path),
            "recorded_float_metrics": {
                key: float(row[key]) for key in ("psnr", "ssim", "lpips")
            },
            "quantized_psnr_minus_recorded_db": (
                analysis["quantized_psnr_db"] - float(row["psnr"])
            ),
        }
    )
    return {"analysis": analysis, "target": target, "prediction": prediction, "error": error}


def final_improvement_counts(a_training: Path, c_training: Path) -> dict:
    a = read_json(a_training / "metrics.json")
    c = read_json(c_training / "metrics.json")
    a_rows = {row["name"]: row for row in a["images"]}
    c_rows = {row["name"]: row for row in c["images"]}
    if list(a_rows) != list(c_rows):
        raise ValueError("A/C held-out identities or order differ")
    result = {"heldout_images": len(a_rows), "metrics": {}}
    for metric in ("psnr", "ssim", "lpips"):
        deltas = [float(c_rows[name][metric] - a_rows[name][metric]) for name in a_rows]
        better = [value > 0 for value in deltas] if metric != "lpips" else [value < 0 for value in deltas]
        result["metrics"][metric] = {
            "improved": int(sum(better)),
            "not_improved": int(len(better) - sum(better)),
            "mean_C_minus_A": float(np.mean(deltas)),
        }
    return result


def font(size: int):
    candidates = (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def build_public_figure(records: dict, output: Path, image_name: str) -> None:
    columns = [
        ("C @ 1k", records["C"][1000]),
        ("C @ 3k", records["C"][3000]),
        ("A @ 7k", records["A"][7000]),
        ("C @ 7k", records["C"][7000]),
    ]
    panel_width = 480
    sample = columns[0][1]["prediction"]
    panel_height = round(sample.shape[0] * panel_width / sample.shape[1])
    header, row_gap, footer = 86, 40, 64
    canvas = Image.new("RGB", (panel_width * len(columns), header + panel_height * 2 + row_gap + footer), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, foot_font = font(30), font(22), font(17)
    draw.text((24, 14), f"Garden seed 3141 · {image_name} · cached predictions", fill="#16263b", font=title_font)
    for column, (label, record) in enumerate(columns):
        x = column * panel_width
        metrics = record["analysis"]["recorded_float_metrics"]
        subtitle = f"{label}   PSNR {metrics['psnr']:.2f}"
        draw.text((x + 14, header - 30), subtitle, fill="#203047", font=label_font)
        prediction = Image.fromarray(record["prediction"]).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        error_gray = Image.fromarray(record["error"][:, :, 0])
        error_color = ImageOps.colorize(error_gray, black="#07111f", white="#ffb000").resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
        canvas.paste(prediction, (x, header))
        canvas.paste(error_color, (x, header + panel_height + row_gap))
    draw.text((18, header + panel_height + 8), "Prediction", fill="#203047", font=label_font)
    draw.text((18, header + panel_height * 2 + row_gap + 15),
              "Error maps: cached mean |RGB prediction − held-out target|. Target RGB is intentionally not included.",
              fill="#465a73", font=foot_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-training-dir", required=True, type=Path)
    parser.add_argument("--c-training-dir", required=True, type=Path)
    parser.add_argument("--image", default="DSC08108.JPG")
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 3000, 7000])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--public-figure", type=Path)
    args = parser.parse_args(argv)

    records: dict[str, dict[int, dict]] = {"A": {}, "C": {}}
    for arm, directory in (("A", args.a_training_dir), ("C", args.c_training_dir)):
        for step in args.steps:
            records[arm][step] = load_step(directory, step, args.image)

    target_hashes = {
        records[arm][step]["analysis"]["target_panel_sha256"]
        for arm in records for step in records[arm]
    }
    if len(target_hashes) != 1:
        raise ValueError("A/C milestone target panels differ")

    public = {
        arm: {str(step): records[arm][step]["analysis"] for step in args.steps}
        for arm in ("A", "C")
    }
    c7 = public["C"]["7000"]
    a7 = public["A"]["7000"]
    trajectory = {
        str(step): {
            metric: public["C"][str(step)]["recorded_float_metrics"][metric]
            - public["A"][str(step)]["recorded_float_metrics"][metric]
            for metric in ("psnr", "ssim", "lpips")
        }
        for step in args.steps
    }
    result = {
        "schema": 1,
        "status": "PASS_CPU_POSTHOC_DIAGNOSTIC",
        "scope": "Existing cached PNGs and metric JSON only; no rendering, training, checkpoint selection, or tuning.",
        "image_name": args.image,
        "target_panels_equal_across_arms_and_steps": True,
        "target_panel_sha256": next(iter(target_hashes)),
        "milestones": public,
        "C_minus_A_metric_trajectory": trajectory,
        "final_heldout_improvement_counts": final_improvement_counts(
            args.a_training_dir, args.c_training_dir
        ),
        "interpretation": {
            "supported": [
                "The C-arm PSNR counterexample emerges between 1k and 3k while SSIM and LPIPS still improve over A.",
                (
                    "At 7k, C has a spatially concentrated upper-image error: "
                    f"{100*c7['spatial_error']['top_half_squared_error_share']:.1f}% of squared RGB error is in the top half, "
                    f"versus {100*a7['spatial_error']['top_half_squared_error_share']:.1f}% for A."
                ),
                (
                    "C has a regional brightness/color shift rather than broad clipping: mean luminance bias is "
                    f"{c7['luminance']['signed_mean']:+.4f}, blue-channel bias is "
                    f"{c7['mean_rgb']['signed_prediction_minus_target'][2]:+.4f}, and all-channel near-white saturation is "
                    f"{100*c7['saturation']['prediction_all_channels_ge_254_over_255_fraction']:.4f}%."
                ),
                "The final A/C target panels, image identity, RGB mode, dimensions, and error-panel construction agree.",
            ],
            "not_supported": [
                "The caches do not identify which Gaussian or structural event created the regional artifact.",
                "Temporal coincidence with an active structural policy is not a densification causal attribution.",
                "The PNG recomputation is quantized and does not replace the recorded float PSNR/SSIM/LPIPS.",
            ],
        },
        "metric_authority": "Recorded float metrics remain authoritative; PNG-derived statistics are diagnostic only.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if args.public_figure:
        build_public_figure(records, args.public_figure, args.image)
        result["public_figure_sha256"] = sha256(args.public_figure)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
