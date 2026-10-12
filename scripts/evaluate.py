#!/usr/bin/env python3
"""Collect experiment evidence and retain failures. This CLI does not launch training.
Run --help for commands; COMMAND --help for its argument contract.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bgs.core import WorkflowError

import re

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

def aggregate_ba_operation_seeds():
    import argparse

    from pathlib import Path

    import statistics

    import sys

    from bgs.core import WorkflowError,atomic_json,read_json

    def main():
        p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
        args=p.parse_args();out=Path(args.out).resolve()
        if out.exists():raise WorkflowError('Refusing to overwrite seed aggregate')
        reports=[read_json(x) for x in args.inputs];metrics=('psnr','ssim','lpips','num_gaussians','model_bytes')
        scenes={}
        for scene in ('bonsai','room'):
            pairs=[]
            for report in reports:
                delta=report['post_minus_pre'][scene]
                pairs.append({'seed':report['seed'],'status':delta['status'],
                              **{key:delta.get(key) for key in metrics}})
            summary={}
            for key in metrics:
                values=[row[key] for row in pairs if row['status']=='VALID']
                summary[key]=({'mean':statistics.mean(values),'stdev':statistics.stdev(values) if len(values)>1 else None,
                               'same_sign':all(x>=0 for x in values) or all(x<=0 for x in values)} if values else None)
            scenes[scene]={'paired_deltas':pairs,'valid_quality_pairs':sum(x['status']=='VALID' for x in pairs),
                           'summary':summary}
        atomic_json(out,{'schema':1,'scope':'training-randomness repeat using the same immutable initialization snapshots',
            'seeds':[r['seed'] for r in reports],'scenes':scenes,
            'limit':'Not independent initialization randomness; two development scenes only.'})
        print(out)

    return main()

def analyze_ba_pair():
    import argparse

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError,atomic_json,file_hash,read_json

    def pose_matrix(image):
        import numpy as np
        pose=image.cam_from_world;pose=pose() if callable(pose) else pose
        return np.asarray(pose.matrix(),dtype=float)

    def reprojection(reconstruction):
        import numpy as np
        errors=[];behind=0
        for point in reconstruction.points3D.values():
            xyz=np.asarray(point.xyz,dtype=float)
            for element in point.track.elements:
                image=reconstruction.images[element.image_id];camera=reconstruction.cameras[image.camera_id]
                cam=pose_matrix(image)[:,:3]@xyz+pose_matrix(image)[:,3]
                if cam[2]<=0:behind+=1;continue
                uv=np.asarray(camera.calibration_matrix(),dtype=float)@cam;uv=uv[:2]/uv[2]
                observed=np.asarray(image.points2D[element.point2D_idx].xy,dtype=float)
                errors.append(float(np.linalg.norm(uv-observed)))
        a=np.asarray(errors,dtype=float)
        if not len(a):raise WorkflowError('No valid observations for reprojection analysis')
        return {'valid_observations':len(a),'behind_camera_observations':behind,
            'mean_px':float(a.mean()),'rmse_px':float(np.sqrt(np.mean(a*a))),
            'median_px':float(np.median(a)),'p95_px':float(np.quantile(a,.95)),'max_px':float(a.max())}

    def deltas(pre,post):
        import numpy as np
        common_images=sorted(set(pre.images)&set(post.images));centers=[];rotations=[];intrinsics=[]
        for image_id in common_images:
            a=pose_matrix(pre.images[image_id]);b=pose_matrix(post.images[image_id])
            ca=-a[:,:3].T@a[:,3];cb=-b[:,:3].T@b[:,3];centers.append(float(np.linalg.norm(ca-cb)))
            rel=b[:,:3]@a[:,:3].T
            rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(rel)-1)/2,-1,1)))))
            pa=np.asarray(pre.cameras[pre.images[image_id].camera_id].params,dtype=float)
            pb=np.asarray(post.cameras[post.images[image_id].camera_id].params,dtype=float)
            intrinsics.extend(np.abs(pa-pb).tolist())
        common_points=sorted(set(pre.points3D)&set(post.points3D))
        movement=np.asarray([np.linalg.norm(np.asarray(pre.points3D[i].xyz)-np.asarray(post.points3D[i].xyz))
                             for i in common_points],dtype=float)
        def summary(values):
            a=np.asarray(values,dtype=float)
            return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),
                    'max':float(a.max())}
        return {'common_images':len(common_images),'common_points3D':len(common_points),
            'camera_center_shift':summary(centers),'camera_rotation_degrees':summary(rotations),
            'camera_parameter_absolute_change':summary(intrinsics),'point3D_shift':summary(movement)}

    def main():
        p=argparse.ArgumentParser();p.add_argument('--pair-root',required=True);args=p.parse_args()
        root=Path(args.pair_root).resolve();out=root/'geometry_analysis.json'
        if out.exists():raise WorkflowError('Refusing to overwrite BA geometry analysis')
        pair=read_json(root/'pair_receipt.json');pre=pycolmap.Reconstruction(str(root/'pre_ba'))
        post=pycolmap.Reconstruction(str(root/'post_ba'))
        atomic_json(out,{'schema':1,'pair_receipt_sha256':file_hash(root/'pair_receipt.json'),
            'pre_ba':reprojection(pre),'post_ba':reprojection(post),'pre_post_delta':deltas(pre,post),
            'ba_seconds':pair['ba_seconds'],'limit':'Joint camera/point BA effect; camera and point contributions are not separated.'})
        print(out)

    return main()

def collect_ba_operation_ablation():
    import argparse

    import csv

    import json

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError,atomic_json,digest,file_hash,read_json

    from bgs.queue import _category

    def main():
        p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--revision',default='v1')
        p.add_argument('--seed',type=int,default=42);p.add_argument('--pair-revision');p.add_argument('--pair-seed',type=int,default=42)
        p.add_argument('--skip-benchmarks',action='store_true')
        p.add_argument('--out-dir',required=True);args=p.parse_args()
        root=Path(args.root).resolve();out=Path(args.out_dir).resolve()
        if out.exists():raise WorkflowError('Refusing to overwrite BA ablation report')
        queue=read_json(root/'work'/'formal_stage_v1'/'queue.json');rows=[];pairs={}
        for scene in ('bonsai','room'):
            pair_root=root/'work'/f'ba_operation_ablation_{args.pair_revision or args.revision}'/'pairs'/f'm360_{scene}_dev_v24_s{args.pair_seed}'
            geometry=read_json(pair_root/'geometry_analysis.json');pair=read_json(pair_root/'pair_receipt.json')
            pairs[scene]={'pair_receipt_sha256':file_hash(pair_root/'pair_receipt.json'),
                'geometry_analysis_sha256':file_hash(pair_root/'geometry_analysis.json'),
                'ba_seconds':pair['ba_seconds'],'geometry':geometry}
            for arm in ('pre','post'):
                run_id=f'm360_{scene}_dev_v24_vggt_ba_{arm}_s{args.seed}_7k_{args.revision}'
                task=queue['tasks'].get(run_id)
                if not task or task['status'] in ('PENDING','RESERVED','RUNNING'):
                    raise WorkflowError(f'Incomplete task: {run_id}')
                config=read_json(task['config_path']);run=Path(config['root'])/'runs'/run_id
                experiment=read_json(run/'experiment.json')
                summary=read_json(run/'summary.json') if (run/'summary.json').is_file() else None
                metrics=read_json(run/'training'/'metrics.json') if (run/'training'/'metrics.json').is_file() else None
                init=read_json(run/'input'/'init_receipt.json') if (run/'input'/'init_receipt.json').is_file() else None
                numerics_path=run/'training'/'initialization_numerics.json'
                numerics=read_json(numerics_path) if numerics_path.is_file() else None
                benchmark=None;benchmark_status='NOT_REQUESTED'
                if not args.skip_benchmarks:
                    benchmark_id=f'bench_{run_id}_mc_v1';benchmark_task=queue['tasks'].get(benchmark_id)
                    benchmark_status=benchmark_task['status'] if benchmark_task else 'MISSING_TASK'
                    benchmark_path=root/'work'/f'ba_operation_ablation_{args.revision}'/'performance'/f'{benchmark_id}.json'
                    if benchmark_status=='SUCCEEDED':benchmark=read_json(benchmark_path)
                ledger=read_json(Path(config['root'])/'ledger.json')
                stages={key.rsplit('/',1)[-1]:value for key,value in ledger['stages'].items() if key.startswith(run_id+'/')}
                elapsed={name:sum(a.get('elapsed_s',0) for a in value['attempts']) for name,value in stages.items()}
                valid=task['status']=='SUCCEEDED' and metrics is not None
                attempts=task.get('attempts',[])
                rows.append({'run_id':run_id,'scene':scene,'seed':args.seed,'arm':arm+'_ba','status':task['status'],
                    'attempt_count':len(attempts),
                    'source_tree_digest':digest(experiment['source']),
                    'manifest_digest':digest(experiment['manifest']),
                    'upstream_digest':digest(experiment['upstream']),
                    'training_contract_digest':digest({key:config.get(key) for key in
                        ('steps','max_edge','seed','noise_seed','checkpoint_every','max_gaussians',
                         'rotation_degrees','translation_fraction','pose_opt','cache_images',
                         'splat_numerics_policy')}),
                    'failure_categories_recorded':'|'.join(str(x.get('category','')) for x in attempts if x.get('category')),
                    'failure_categories_interpreted':'|'.join(_category(x['log'],x.get('status',task['status']))
                                                              for x in attempts if x.get('status')!='SUCCEEDED'),
                    'failure_log_sha256':'|'.join(file_hash(x['log']) for x in attempts if Path(x.get('log','')).is_file()),
                    'registered_views':len(init['registered_names']) if init else None,
                    'initial_points':init['num_points'] if init else None,
                    'splat_numerics_policy':config.get('splat_numerics_policy','upstream_v1'),
                    'clamped_scale_points':numerics.get('clamped_point_count') if numerics else None,
                    'positive_scale_floor':numerics.get('positive_scale_floor') if numerics else None,
                    'first_invalid_after_policy':numerics.get('first_invalid_after_policy') if numerics else None,
                    'psnr':metrics['mean']['psnr'] if valid else None,
                    'ssim':metrics['mean']['ssim'] if valid else None,
                    'lpips':metrics['mean']['lpips'] if valid else None,
                    'num_gaussians':metrics['num_gaussians'] if valid else None,
                    'model_bytes':metrics['model_bytes'] if valid else None,
                    'render_fps_single_camera_receipt':metrics['render_fps'] if valid else None,
                    'render_all_camera_mean_ms':benchmark['latency_ms']['mean'] if benchmark else None,
                    'render_all_camera_p95_ms':benchmark['latency_ms']['p95'] if benchmark else None,
                    'render_all_camera_fps':benchmark['fps_from_mean_latency'] if benchmark else None,
                    'render_camera_count':benchmark['camera_count'] if benchmark else None,
                    'benchmark_status':benchmark_status,
                    'peak_allocated_bytes':metrics['peak_allocated_bytes'] if valid else None,
                    'initialize_s':elapsed.get('initialize'),'train_evaluate_s':elapsed.get('train_evaluate'),
                    'end_to_end_s':summary['end_to_end_seconds_known_attempts'] if summary else None,
                    'queue_gpu_hours':sum(x.get('gpu_hours',0) for x in attempts),
                    'model_sha256':file_hash(run/'training'/'model.ply') if valid else None,
                    'metrics_sha256':file_hash(run/'training'/'metrics.json') if valid else None,
                    'init_receipt_sha256':file_hash(run/'input'/'init_receipt.json') if init else None,
                    'initialization_numerics_sha256':file_hash(numerics_path) if numerics else None})
        by={(x['scene'],x['arm']):x for x in rows};deltas={}
        for scene in ('bonsai','room'):
            a,b=by[scene,'pre_ba'],by[scene,'post_ba']
            if a['status']=='SUCCEEDED' and b['status']=='SUCCEEDED':
                keys=['psnr','ssim','lpips','num_gaussians','model_bytes','train_evaluate_s','end_to_end_s','queue_gpu_hours']
                if not args.skip_benchmarks:keys+=['render_all_camera_mean_ms','render_all_camera_p95_ms']
                deltas[scene]={'status':'VALID',**{key:b[key]-a[key] for key in keys}}
            else:
                deltas[scene]={'status':'UNAVAILABLE','pre_status':a['status'],'post_status':b['status'],
                    'reason':'A failed arm has no protocol-valid held-out quality/model/render metrics.'}
        out.mkdir(parents=True);atomic_json(out/'results.json',{'schema':1,'revision':args.revision,'seed':args.seed,
            'scope':'isolated official pycolmap.bundle_adjustment operation on one persisted reconstruction per scene',
            'limit':'Joint camera/point optimization; does not separate their contributions. Failed arms stay in the denominator.',
            'original_no_ba_branch':'Separate initialization pipeline and not an ablation arm.',
            'seed_scope':('Training randomness only; immutable initialization pair reused.' if args.pair_revision else
                          'Declared seed used for pair generation and training.'),
            'rows':rows,'post_minus_pre':deltas,'pairs':pairs})
        fields=list(rows[0]);
        with (out/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
        operation_text=('Both arms reuse the immutable seed-42 pre/post reconstruction pair; this seed tests downstream training randomness only.'
                        if args.pair_revision else
                        'Both arms reload isolated copies of the same pre-BA official-branch reconstruction. Only the post arm executes `pycolmap.bundle_adjustment`.')
        lines=['# Matched BA operation ablation','',
            operation_text+' The comparison does not separate camera from point optimization.','',
            '| Scene | Arm | Status | PSNR | SSIM | LPIPS | GS | PLY bytes | init s | train/eval s | render mean/p95 ms | queue GPUh |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(value,digits=4):return 'n/a' if value is None else f'{value:.{digits}f}'
        for x in rows:
            render=('n/a' if x['render_all_camera_mean_ms'] is None else
                    f"{x['render_all_camera_mean_ms']:.3f}/{x['render_all_camera_p95_ms']:.3f}")
            lines.append(f"| {x['scene']} | {x['arm']} | {x['status']} | {fmt(x['psnr'])} | {fmt(x['ssim'])} | {fmt(x['lpips'])} | {x['num_gaussians'] or 'n/a'} | {x['model_bytes'] or 'n/a'} | {fmt(x['initialize_s'],1)} | {fmt(x['train_evaluate_s'],1)} | {render} | {x['queue_gpu_hours']:.4f} |")
        lines+=['','Post minus pre:']
        for scene,x in deltas.items():
            if x['status']=='VALID':lines.append(f"- {scene}: PSNR {x['psnr']:+.4f}, SSIM {x['ssim']:+.4f}, LPIPS {x['lpips']:+.4f}, GS {x['num_gaussians']:+d}, PLY {x['model_bytes']:+d} bytes.")
            else:lines.append(f"- {scene}: unavailable ({x['pre_status']} pre, {x['post_status']} post); no failed arm was assigned quality metrics.")
        if all(x['status']=='FAILED' for x in rows if x['arm']=='pre_ba'):
            retried=all(x['attempt_count']==2 for x in rows if x['arm']=='pre_ba')
            retry_text=' on the original attempt and the one allowed exact retry' if retried else ''
            lines+=['', f'Both pre-BA arms failed the finite-state gate at step 100{retry_text}. This validates no paired image-quality, model-size or render-cost delta.']
        (out/'results.md').write_text('\n'.join(lines)+'\n')
        print(out/'results.json')

    return main()

def collect_final_results():
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

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--state", required=True)
        parser.add_argument("--protocol", required=True)
        parser.add_argument("--registry", required=True)
        parser.add_argument("--json-out", required=True)
        parser.add_argument("--csv-out", required=True)
        args = parser.parse_args()
        import pycolmap

        state = read_json(Path(args.state).resolve())
        protocol_path = Path(args.protocol).resolve()
        protocol = read_json(protocol_path)
        registry_path = Path(args.registry).resolve()
        registry = read_json(registry_path)
        expected = list(protocol.get("run_ids", []))
        if protocol.get("status") != "FROZEN_BEFORE_SCORING" or len(expected) < 6:
            raise WorkflowError("Invalid frozen final protocol")
        registry_scenes = {item["scene_id"]: item for item in registry.get("final", [])}
        if (registry.get("status") != "FROZEN_FINAL"
                or set(registry_scenes) != set(protocol.get("scene_ids", []))):
            raise WorkflowError("Final registry does not match the frozen protocol")
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
                "category": registry_scenes[manifest["scene_id"]]["category"],
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
            "registry_sha256": file_hash(registry_path),
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
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"json": str(json_out), "csv": str(csv_out), "rows": len(rows)}, indent=2))

    return main()

def collect_formal_results():
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

    return main()

def collect_pose_sensitivity():
    import argparse

    import csv

    import html

    from pathlib import Path

    import sys

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

    return main()

COMMANDS = {
    'aggregate-ba-operation-seeds': aggregate_ba_operation_seeds,
    'analyze-ba-pair': analyze_ba_pair,
    'collect-ba-operation-ablation': collect_ba_operation_ablation,
    'collect-final-results': collect_final_results,
    'collect-formal-results': collect_formal_results,
    'collect-pose-sensitivity': collect_pose_sensitivity,
}

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not args or args[0] in ("-h","--help"):
        parser.print_help()
        return
    command=parser.parse_args(args[:1]).command
    saved=sys.argv
    try:
        sys.argv=[saved[0]+" "+command, *args[1:]]
        return COMMANDS[command]()
    finally:
        sys.argv=saved

if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
