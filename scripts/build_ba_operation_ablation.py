#!/usr/bin/env python3
"""Build the two-scene, same-reconstruction pre/post BA operation ablation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from bgs.core import WorkflowError,atomic_json,file_hash,read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--revision',default='v1')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--reuse-pair-revision');p.add_argument('--pair-seed',type=int,default=42)
    p.add_argument('--gpu-uuids',nargs=4,required=True,metavar=('GPU0','GPU1','GPU2','GPU3'))
    args=p.parse_args();gpu_uuids=dict(enumerate(args.gpu_uuids))
    if any(not value.startswith('GPU-') for value in gpu_uuids.values()):raise WorkflowError('Every GPU UUID must start with GPU-')
    root=Path(args.root).resolve();stage=root/'work'/f'ba_operation_ablation_{args.revision}'
    config_dir=root/'configs'/f'ba_operation_ablation_{args.revision}'
    task_dir=root/'remote_tasks'/f'ba_operation_ablation_{args.revision}'
    if config_dir.exists() or task_dir.exists():raise WorkflowError('Refusing to overwrite immutable ablation matrix')
    config_dir.mkdir(parents=True);task_dir.mkdir(parents=True)
    created=[]
    for scene,post_gpu,pre_gpu in [('bonsai',0,2),('room',1,3)]:
        base_path=root/'configs'/'mipnerf360_ba_v2'/f'm360_{scene}_dev_v24_vggt_ba_s42_7k_v2.json'
        base=read_json(base_path)
        pair_stage=root/'work'/f'ba_operation_ablation_{args.reuse_pair_revision or args.revision}'
        pair_root=pair_stage/'pairs'/f'm360_{scene}_dev_v24_s{args.pair_seed}'
        post_id=f'm360_{scene}_dev_v24_vggt_ba_post_s{args.seed}_7k_{args.revision}'
        pre_id=f'm360_{scene}_dev_v24_vggt_ba_pre_s{args.seed}_7k_{args.revision}'
        pair_dependency=(f'm360_{scene}_dev_v24_vggt_ba_post_s{args.pair_seed}_7k_{args.reuse_pair_revision}'
                         if args.reuse_pair_revision else None)
        for arm,gpu,run_id,generate,dependencies in [
            ('post_ba',post_gpu,post_id,not args.reuse_pair_revision,[pair_dependency] if pair_dependency else []),
            ('pre_ba',pre_gpu,pre_id,False,[pair_dependency] if pair_dependency else [post_id])]:
            config={**base,'run_id':run_id,'root':str(stage/'campaigns'/run_id),'gpu':str(gpu),
                'gpu_hours':1.25,'timeout_s':4200,'seed':args.seed,
                'vggt_ba_ablation_arm':arm,'vggt_ba_pair_root':str(pair_root),
                'vggt_ba_pair_generate':generate,
                'hypothesis':'Holding the official BA-branch reconstruction fixed, the pycolmap BA call changes held-out quality.',
                'independent_contribution':'Same-snapshot operation ablation; does not separate camera from point optimization.',
                'ablation_scope':'Only the pycolmap.bundle_adjustment call differs; official no-BA remains a separate pipeline.'}
            config.pop('control_of',None);config.pop('control_of_sha256',None)
            config_path=config_dir/f'{run_id}.json';atomic_json(config_path,config)
            run=Path(config['root'])/'runs'/run_id
            expected=[run/'summary.json',run/'input'/'init_receipt.json',run/'training'/'model.ply',
                      run/'training'/'metrics.json',run/'training'/'complete.json']
            task={'task_id':run_id,'kind':'bgs_run',
                'argv':[config['python'],'-m','bgs','run','--config',str(config_path)],
                'cwd':str(root),'env':{'PYTHONPATH':str(root),'TORCH_HOME':str(root/'cache'/'torch'),
                    'TMPDIR':str(root/'tmp'/f'ba_operation_ablation_{args.revision}'/run_id)},
                'dependencies':dependencies,'config_path':str(config_path),'config_sha256':file_hash(config_path),
                'gpu_index':gpu,'gpu_uuid':gpu_uuids[gpu],'gpu_count':1,'max_gpu_hours':1.25,
                'timeout_s':4200,'disk_reservation_bytes':10*2**30,
                'expected_outputs':[str(x) for x in expected],'priority':10 if generate else 20}
            task_path=task_dir/f'{run_id}.task.json';atomic_json(task_path,task)
            created.append({'run_id':run_id,'arm':arm,'scene':scene,'config':str(config_path),
                            'config_sha256':file_hash(config_path),'task':str(task_path)})
    atomic_json(task_dir/'matrix_receipt.json',{'schema':1,'revision':args.revision,'seed':args.seed,
        'scenes':['bonsai','room'],'fixed':['same persisted pre-BA reconstruction','images','tracks','initial points',
        'camera model','preprocessing','7000-step gsplat schedule','max_edge=1024'],
        'only_operation_difference':'pycolmap.bundle_adjustment executed for post_ba arm',
        'pair_source':{'revision':args.reuse_pair_revision or args.revision,'seed':args.pair_seed},
        'seed_scope':('Training randomness only; both arms reuse the immutable seed-42 initialization snapshots.'
                      if args.reuse_pair_revision else 'Initialization and training use the declared seed.'),
        'limit':'Does not separate camera optimization from point optimization.',
        'original_no_ba_branch':'Separate initialization pipeline; not an arm of this operation ablation.',
        'created':created})
    print(task_dir/'matrix_receipt.json')


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
