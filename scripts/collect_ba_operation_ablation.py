#!/usr/bin/env python3
"""Collect the completed same-reconstruction BA operation ablation."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
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


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
