#!/usr/bin/env python3
"""Collect the frozen development-scene pose-sensitivity experiment."""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from bgs.core import WorkflowError,atomic_json,digest,file_hash,read_json


TERMINAL={"SUCCEEDED","FAILED","TIMED_OUT","FAILED_MISSING_OUTPUT","FAILED_CONFIG_TAMPER",
          "FAILED_LOST_WORKER","FAILED_LAUNCH","BLOCKED_DEPENDENCY"}
METRICS=("psnr","ssim","lpips")


def relative(root,path):
    return str(Path(path).resolve().relative_to(root))


def task_gpu_hours(task):
    return sum(float(x.get("gpu_hours",0.)) for x in task.get("attempts",[]))


def stage_times(config,run_id):
    ledger=read_json(Path(config["root"])/"ledger.json")
    stages={key.rsplit("/",1)[-1]:value for key,value in ledger["stages"].items() if key.startswith(run_id+"/")}
    return {name:sum(float(a.get("elapsed_s",0.)) for a in value["attempts"]) for name,value in stages.items()}


def svg_chart(path,title,modality,rows):
    width,height=1080,360;left,right,top,bottom=62,20,42,55;panel_w=(width-left-right)/3
    colors={"bonsai":"#1677ff","room":"#d46b08"};labels={"psnr":"PSNR (dB)","ssim":"SSIM","lpips":"LPIPS"}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',f'<text x="{width/2}" y="24" text-anchor="middle" font-family="system-ui" font-size="17">{html.escape(title)}</text>']
    for panel,metric in enumerate(METRICS):
        x0=left+panel*panel_w;x1=x0+panel_w-26;y0=top;y1=height-bottom
        values=[r[metric] for r in rows if r["modality"]==modality and r[metric] is not None]
        if not values:continue
        lo,hi=min(values),max(values);pad=max((hi-lo)*.12,1e-6);lo-=pad;hi+=pad
        xvals=sorted(set(r["level"] for r in rows if r["modality"]==modality and r[metric] is not None));xmin,xmax=min(xvals),max(xvals)
        xp=lambda x:x0+(x-xmin)/max(xmax-xmin,1e-12)*(x1-x0)
        yp=lambda y:y1-(y-lo)/(hi-lo)*(y1-y0)
        parts+=['<g stroke="#d9d9d9" stroke-width="1">',f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>',f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>','</g>',
            f'<text x="{(x0+x1)/2}" y="{height-12}" text-anchor="middle" font-family="system-ui" font-size="12">{("rotation degrees" if modality=="rotation" else "translation / camera radius")}</text>',
            f'<text x="{x0}" y="{y0-8}" font-family="system-ui" font-size="13">{labels[metric]}</text>',
            f'<text x="{x0-5}" y="{y0+4}" text-anchor="end" font-family="monospace" font-size="10">{hi:.4g}</text>',
            f'<text x="{x0-5}" y="{y1+4}" text-anchor="end" font-family="monospace" font-size="10">{lo:.4g}</text>']
        for x in xvals:parts.append(f'<text x="{xp(x)}" y="{y1+17}" text-anchor="middle" font-family="monospace" font-size="10">{x:g}</text>')
        for scene in ("bonsai","room"):
            points=sorted((r["level"],r[metric]) for r in rows if r["scene"]==scene and r["modality"]==modality and r[metric] is not None)
            coords=" ".join(f"{xp(x):.2f},{yp(y):.2f}" for x,y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[scene]}" stroke-width="2.5"/>')
            for x,y in points:parts.append(f'<circle cx="{xp(x):.2f}" cy="{yp(y):.2f}" r="4" fill="{colors[scene]}"/>')
    parts+=['<g font-family="system-ui" font-size="12">','<circle cx="850" cy="23" r="4" fill="#1677ff"/><text x="860" y="27">bonsai</text>',
        '<circle cx="930" cy="23" r="4" fill="#d46b08"/><text x="940" y="27">room</text>','</g>','</svg>']
    path.write_text("\n".join(parts)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--matrix",required=True)
    p.add_argument("--probe",required=True);p.add_argument("--out-dir",required=True);args=p.parse_args()
    root=Path(args.root).resolve();matrix_path=Path(args.matrix).resolve();matrix=read_json(matrix_path)
    probe_path=Path(args.probe).resolve();probe=read_json(probe_path)
    if probe.get("status")!="PASS" or probe.get("matrix_sha256")!=file_hash(matrix_path):raise WorkflowError("Pose probe is missing or stale")
    out=Path(args.out_dir).resolve()
    if out.exists():raise WorkflowError("Refusing to overwrite pose-sensitivity report")
    queue=read_json(root/"work"/"formal_stage_v1"/"queue.json")
    protocol=read_json(matrix["protocol"]);new_by={(x["scene"],x["condition"]):x for x in matrix["new_runs"]}
    rows=[]
    for scene in ("bonsai","room"):
        zero=matrix["zero_references"][scene]
        conditions=[("zero",0.,0.)]+[(x["id"],x["rotation_degrees"],x["translation_fraction"]) for x in protocol["conditions"] if x["id"]!="zero"]
        for condition,rotation,translation in conditions:
            if condition=="zero":
                run_id=zero["run_id"];config_path=Path(zero["config"]);status="SUCCEEDED";task=queue["tasks"][run_id]
            else:
                frozen=new_by[scene,condition];config_path=Path(frozen["config"]);run_id=read_json(config_path)["run_id"]
                task=queue["tasks"].get(run_id)
                if task is None or task["status"] not in TERMINAL:raise WorkflowError(f"Incomplete task: {run_id}")
                status=task["status"]
            config=read_json(config_path);run=Path(config["root"])/"runs"/run_id
            valid=status=="SUCCEEDED" and (run/"training"/"metrics.json").is_file()
            metrics=read_json(run/"training"/"metrics.json") if valid else None
            experiment=read_json(run/"experiment.json") if (run/"experiment.json").is_file() else None
            init=read_json(run/"input"/"init_receipt.json") if (run/"input"/"init_receipt.json").is_file() else None
            alignment=run/"training"/"alignment.json";times=stage_times(config,run_id) if valid else {}
            fixed_name=protocol["showcase_holdouts"][scene]
            fixed=next((x for x in metrics["images"] if x["name"]==fixed_name),None) if valid else None
            if valid and fixed is None:raise WorkflowError(f"Missing preregistered holdout: {scene}/{fixed_name}")
            init_match=bool(init and init.get("files")==zero["initialization_files"])
            alignment_match=alignment.is_file() and file_hash(alignment)==zero["output_sha256"]["alignment"]
            source=experiment["source"] if experiment else {};zero_source=zero["source_hashes"]
            source_differences=sorted(k for k in set(source)|set(zero_source) if source.get(k)!=zero_source.get(k))
            # The zero run predates a later validation-only deployment sync: __main__ gained
            # a configure flag, pipeline gained policy validation/one additional expected
            # output, and three macOS AppleDouble files were removed. None is imported by
            # train/evaluate. All remaining bgs modules, including data, geometry,
            # initialization, numerical policy, metrics and train adapter, must match.
            allowed_wrapper_differences={"bgs/__main__.py","bgs/pipeline.py","bgs/.___main__.py",
                "bgs/._initialize.py","bgs/._train_adapter.py"}
            behavior_keys=(set(source)|set(zero_source))-allowed_wrapper_differences
            behavior_source_match=bool(experiment and all(source.get(k)==zero_source.get(k) for k in behavior_keys))
            source_difference_allowed=set(source_differences)<=allowed_wrapper_differences
            if valid and not (init_match and alignment_match and behavior_source_match and source_difference_allowed):
                raise WorkflowError(f"Matched-control identity failed: {run_id}")
            row={"scene":scene,"condition":condition,"run_id":run_id,"status":status,
                "rotation_degrees":rotation,"translation_fraction":translation,
                "zero_reused":condition=="zero","config_sha256":file_hash(config_path),
                "source_tree_digest":digest(experiment["source"]) if experiment else None,
                "source_tree_exact_match":not source_differences,
                "training_evaluation_source_match":behavior_source_match,
                "allowed_wrapper_source_differences":"|".join(source_differences),
                "initialization_files_match_zero":init_match,"alignment_sha256_match_zero":alignment_match,
                "pose_optimization":config.get("pose_opt"),"training_seed":config.get("seed"),"noise_seed":config.get("noise_seed"),
                "psnr":metrics["mean"]["psnr"] if valid else None,"ssim":metrics["mean"]["ssim"] if valid else None,
                "lpips":metrics["mean"]["lpips"] if valid else None,"num_gaussians":metrics["num_gaussians"] if valid else None,
                "model_bytes":metrics["model_bytes"] if valid else None,"training_seconds":metrics["training_seconds_this_attempt"] if valid else None,
                "optimization_seconds":metrics["optimization_seconds_this_attempt"] if valid else None,
                "initialize_seconds":times.get("initialize"),"train_evaluate_seconds":times.get("train_evaluate"),
                "queue_gpu_hours":task_gpu_hours(task),"peak_allocated_bytes":metrics["peak_allocated_bytes"] if valid else None,
                "render_fps_single_camera":metrics["render_fps"] if valid else None,
                "fixed_holdout":fixed_name,"fixed_holdout_psnr":fixed["psnr"] if fixed else None,
                "fixed_holdout_ssim":fixed["ssim"] if fixed else None,"fixed_holdout_lpips":fixed["lpips"] if fixed else None,
                "fixed_holdout_panel":relative(root,run/"training"/fixed["visualization"]) if fixed else None,
                "metrics_sha256":file_hash(run/"training"/"metrics.json") if valid else None,
                "model_sha256":file_hash(run/"training"/"model.ply") if valid else None,
                "failure_log_sha256":"|".join(file_hash(x["log"]) for x in task.get("attempts",[]) if Path(x.get("log","")).is_file() and x.get("status")!="SUCCEEDED")}
            rows.append(row)
    zero_by={x["scene"]:x for x in rows if x["condition"]=="zero"};deltas={}
    chart_rows=[]
    for row in rows:
        zero=zero_by[row["scene"]]
        if row["status"]=="SUCCEEDED":
            row.update({f"delta_{m}":row[m]-zero[m] for m in METRICS})
            modality="rotation" if row["translation_fraction"]==0 else "translation"
            if row["condition"]=="zero":
                for m in ("rotation","translation"):chart_rows.append({**row,"modality":m,"level":0.})
            else:chart_rows.append({**row,"modality":modality,
                "level":row["rotation_degrees"] if modality=="rotation" else row["translation_fraction"]})
            deltas[f'{row["scene"]}/{row["condition"]}']={m:row[f"delta_{m}"] for m in METRICS}
        else:
            row.update({f"delta_{m}":None for m in METRICS})
    round_gpu_hours=sum(task_gpu_hours(queue["tasks"][x["run_id"]]) for x in rows if not x["zero_reused"])
    charged_now=float(queue["charged_gpu_hours"])
    report={"schema":1,"protocol_id":"pose_sensitivity_v1","protocol_sha256":file_hash(matrix["protocol"]),
        "matrix_sha256":file_hash(matrix_path),"probe_sha256":file_hash(probe_path),
        "scope":protocol["scope"],"interpretation_limits":protocol["interpretation_limits"],
        "camera_radius_definition":protocol["camera_radius_definition"],"rotation_convention":protocol["rotation_convention"],
        "translation_convention":protocol["translation_convention"],"anchor":protocol["anchor"],
        "zero_reuse_code_audit":{"training_evaluation_modules_match":True,
            "allowed_wrapper_differences":["bgs/__main__.py: configure-only numerical-policy argument",
                "bgs/pipeline.py: policy validation and initialization_numerics expected-output check",
                "three removed macOS AppleDouble files that Python never imported"],
            "boundary":"The wrapper source tree is not claimed byte-identical; matched training/evaluation behavior is supported by the module audit, exact initialization files, exact alignment and zero-perturbation tests/probe."},
        "rows":rows,"delta_from_zero":deltas,"budget":{"total_cap_gpu_hours":queue["budget_gpu_hours"],
            "round_start_charged_gpu_hours":matrix["round_start_charged_gpu_hours"],"charged_at_collection":charged_now,
            "round_new_gpu_hours":round_gpu_hours,"total_remaining_gpu_hours":queue["budget_gpu_hours"]-charged_now,
            "round_cap_gpu_hours":matrix["round_cap_gpu_hours"]},
        "failure_accounting":"Every frozen condition is retained; failed arms have null quality metrics.",
        "visual_protocol":"Each fixed panel is reference | prediction | absolute RGB error for the preregistered first holdout."}
    out.mkdir(parents=True);atomic_json(out/"results.json",report)
    fields=list(rows[0]);
    with (out/"results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    svg_chart(out/"rotation_sensitivity.svg","Controlled rotation sensitivity","rotation",chart_rows)
    svg_chart(out/"translation_sensitivity.svg","Controlled translation sensitivity","translation",chart_rows)
    def fmt(value,digits=4):return "n/a" if value is None else f"{value:.{digits}f}"
    lines=["# Controlled pose sensitivity (development only)","",
        "This protocol adds controlled error to existing estimated **training** poses. It is not known absolute pose error, geometry accuracy, or a general robustness proof. Alignment was fixed before perturbation; initial points, intrinsics, RGB and held-out cameras were unchanged.","",
        "| Scene | Condition | Status | PSNR (delta) | SSIM (delta) | LPIPS (delta) | GS | train s | peak GiB | queue GPUh |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        def metric(name):return "n/a" if x[name] is None else f'{x[name]:.4f} ({x["delta_"+name]:+.4f})'
        lines.append(f'| {x["scene"]} | {x["condition"]} | {x["status"]} | {metric("psnr")} | {metric("ssim")} | {metric("lpips")} | {x["num_gaussians"] or "n/a"} | {fmt(x["training_seconds"],1)} | {fmt(x["peak_allocated_bytes"]/2**30 if x["peak_allocated_bytes"] else None,2)} | {x["queue_gpu_hours"]:.4f} |')
    lines+=["",f'New matrix cost: **{round_gpu_hours:.6f} GPUh**; total ledger: **{charged_now:.6f}/{queue["budget_gpu_hours"]:.0f} GPUh**.',"",
        "Curves: [rotation](rotation_sensitivity.svg) · [translation](translation_sensitivity.svg). Fixed-view private panel paths are recorded in `results.json`.","",
        "One training seed and one noise-direction set are shown. No monotonicity or cross-scene generalization is assumed."]
    (out/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out/"results.json")


if __name__=="__main__":
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as error:raise SystemExit(f"BLOCKED: {error}")
