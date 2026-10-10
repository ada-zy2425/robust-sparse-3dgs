#!/usr/bin/env python3
"""Build bounded all-held-out-camera render benchmarks for BA operation arms."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bgs.core import WorkflowError,atomic_json,file_hash,read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--revision',default='v1')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-gpu-hours',type=float,default=.25);p.add_argument('--timeout-s',type=int,default=900)
    p.add_argument('--gpu-uuids',nargs=4,required=True,metavar=('GPU0','GPU1','GPU2','GPU3'))
    args=p.parse_args();root=Path(args.root).resolve();gpu_uuids=dict(enumerate(args.gpu_uuids))
    if args.max_gpu_hours<=0 or args.timeout_s<=0 or args.timeout_s/3600>args.max_gpu_hours:
        raise WorkflowError('Benchmark timeout must be positive and fit its GPU-hour bound')
    if any(not value.startswith('GPU-') for value in gpu_uuids.values()):raise WorkflowError('Every GPU UUID must start with GPU-')
    taskdir=root/'remote_tasks'/f'ba_operation_ablation_{args.revision}_benchmarks'
    outdir=root/'work'/f'ba_operation_ablation_{args.revision}'/'performance'
    if taskdir.exists():raise WorkflowError('Refusing to overwrite BA benchmark tasks')
    taskdir.mkdir(parents=True);rows=[]
    for scene,post_gpu,pre_gpu in [('bonsai',0,2),('room',1,3)]:
        for arm,gpu in [('post',post_gpu),('pre',pre_gpu)]:
            run_id=f'm360_{scene}_dev_v24_vggt_ba_{arm}_s{args.seed}_7k_{args.revision}'
            config=root/'configs'/f'ba_operation_ablation_{args.revision}'/f'{run_id}.json'
            task_id=f'bench_{run_id}_mc_v1';output=outdir/f'{task_id}.json'
            task={'task_id':task_id,'kind':'render_benchmark',
                'argv':[str(root/'envs'/'gsplat'/'bin'/'python'),str(root/'scripts'/'benchmark_render.py'),
                    '--config',str(config),'--out',str(output),'--warmup','5','--repeats','30','--max-cameras','64'],
                'cwd':str(root),'env':{'PYTHONPATH':str(root),'TMPDIR':str(root/'tmp'/f'ba_operation_ablation_{args.revision}'/task_id)},
                'dependencies':[run_id],'config_path':str(config),'config_sha256':file_hash(config),
                'gpu_index':gpu,'gpu_uuid':gpu_uuids[gpu],'gpu_count':1,
                'max_gpu_hours':args.max_gpu_hours,'timeout_s':args.timeout_s,
                'disk_reservation_bytes':2**30,'expected_outputs':[str(output)],'priority':30}
            path=taskdir/f'{task_id}.task.json';atomic_json(path,task);rows.append(str(path))
    atomic_json(taskdir/'matrix_receipt.json',{'schema':1,'revision':args.revision,'seed':args.seed,
        'protocol':'every fixed held-out camera; 5 warmups and 30 synchronized repeats',
        'per_task_max_gpu_hours':args.max_gpu_hours,'timeout_s':args.timeout_s,'tasks':rows})
    print('\n'.join(rows))


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
