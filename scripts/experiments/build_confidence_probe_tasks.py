#!/usr/bin/env python3
"""Build bounded confidence probes for the two authorized development scenes."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from bgs.core import WorkflowError,atomic_json,file_hash,read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--revision',default='v1')
    p.add_argument('--gpu-indices',nargs=2,type=int,required=True,metavar=('BONSAI_GPU','ROOM_GPU'))
    p.add_argument('--gpu-uuids',nargs=2,required=True,metavar=('BONSAI_UUID','ROOM_UUID'))
    args=p.parse_args();root=Path(args.root).resolve()
    if any(index<0 for index in args.gpu_indices) or any(not value.startswith('GPU-') for value in args.gpu_uuids):
        raise WorkflowError('GPU indices and UUIDs must be explicit')
    outdir=root/'work'/f'vggt_confidence_diagnostic_{args.revision}'
    taskdir=root/'remote_tasks'/f'vggt_confidence_diagnostic_{args.revision}'
    if taskdir.exists():raise WorkflowError('Refusing to overwrite confidence tasks')
    taskdir.mkdir(parents=True)
    rows=[]
    values=[('bonsai',args.gpu_indices[0],args.gpu_uuids[0]),
            ('room',args.gpu_indices[1],args.gpu_uuids[1])]
    for scene,gpu,uuid in values:
        base_path=root/'configs'/'mipnerf360_dev_v3'/f'm360_{scene}_dev_v24_vggt_s42_7k_v3.json'
        config=read_json(base_path);run=Path(config['root'])/'runs'/config['run_id']
        task_id=f'm360_{scene}_v24_vggt_confidence_s42_{args.revision}'
        output=outdir/f'{task_id}.json'
        task={'task_id':task_id,'kind':'initialization_diagnostic',
            'argv':[str(root/'envs'/'vggt'/'bin'/'python'),str(root/'scripts'/"diagnostics" / 'vggt_confidence_probe.py'),
                '--demo',str(root/'third_party'/'vggt'/'demo_colmap.py'),'--scene-dir',str(run/'input'),
                '--out',str(output),'--seed','42','--threshold','5'],
            'cwd':str(root),'env':{'PYTHONPATH':str(root),'TORCH_HOME':str(root/'cache'/'torch'),
                'TMPDIR':str(root/'tmp'/f'vggt_confidence_diagnostic_{args.revision}'/task_id)},
            'dependencies':[config['run_id']],'config_path':str(base_path),'config_sha256':file_hash(base_path),
            'gpu_index':gpu,'gpu_uuid':uuid,'gpu_count':1,'max_gpu_hours':.25,'timeout_s':900,
            'disk_reservation_bytes':2*2**30,'expected_outputs':[str(output)],'priority':15}
        path=taskdir/f'{task_id}.task.json';atomic_json(path,task);rows.append(str(path))
    atomic_json(taskdir/'matrix_receipt.json',{'schema':1,'scope':'read-only no-BA confidence distribution',
        'scenes':['bonsai','room'],'seed':42,'threshold':5.,'revision':args.revision,'tasks':rows})
    print('\n'.join(rows))


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
