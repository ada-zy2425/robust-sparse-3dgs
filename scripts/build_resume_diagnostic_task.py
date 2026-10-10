#!/usr/bin/env python3
"""Materialize the bounded first-divergence diagnostic task."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bgs.core import WorkflowError,atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--mode',choices=['first','boundaries'],default='first');args=p.parse_args()
    if args.gpu<0 or not args.gpu_uuid.startswith('GPU-'):raise WorkflowError('GPU index and UUID must be explicit')
    root=Path(args.root).resolve()
    task_id='sb_smoke_resume1400_boundaries_v6' if args.mode=='boundaries' else 'sb_smoke_resume1400_first_divergence_v8'
    config=root/'south_building_window0141_0156_colmap_smoke_v2.json'
    folder='resume_boundaries_1400_v6' if args.mode=='boundaries' else 'resume_divergence_1400_v8'
    output=root/'work'/'south_building_smoke_campaign_v1'/'runs'/'south_building_window0141_0156_colmap_smoke_v2'/folder/'qualification.json'
    task_path=root/'remote_tasks'/'resume_diagnostics_v1'/f'{task_id}.task.json'
    if task_path.exists():raise WorkflowError('Refusing to overwrite diagnostic task')
    task_path.parent.mkdir(parents=True,exist_ok=True)
    mode_arg='--diagnose-boundaries' if args.mode=='boundaries' else '--diagnose-first-divergence'
    task={'task_id':task_id,'kind':'resume_qualification',
        'argv':[str(root/'envs'/'gsplat'/'bin'/'python'),str(root/'scripts'/'qualify_resume.py'),
                '--config',str(config),'--steps','1400',mode_arg],
        'cwd':str(root),'env':{'PYTHONPATH':str(root),'TMPDIR':str(root/'tmp'/'resume_diagnostics_v1'/task_id)},
        'dependencies':[],'config_path':str(config),'config_sha256':file_hash(config),
        'gpu_index':args.gpu,'gpu_uuid':args.gpu_uuid,'gpu_count':1,
        'max_gpu_hours':0.25,'timeout_s':900,'disk_reservation_bytes':2*2**30,
        'expected_outputs':[str(output)],'priority':5}
    atomic_json(task_path,task);print(task_path)


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
