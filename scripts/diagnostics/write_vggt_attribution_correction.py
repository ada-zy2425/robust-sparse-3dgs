#!/usr/bin/env python3
"""Add a versioned effective-config index without modifying historical receipts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from bgs.core import WorkflowError,atomic_json,file_hash,read_json
from bgs.initialize import _vggt_policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--queue',required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(args.root).resolve();queue=read_json(args.queue);out=Path(args.out).resolve()
    if out.exists():raise WorkflowError('Refusing to overwrite attribution correction')
    rows=[];seen=set()
    for task_id,task in sorted(queue['tasks'].items()):
        config_path=Path(task['config_path'])
        if not config_path.is_file():continue
        config=read_json(config_path)
        if config.get('initializer')!='vggt':continue
        run=Path(config['root'])/'runs'/config['run_id'];receipt=run/'input'/'init_receipt.json'
        if not receipt.is_file():continue
        key=(config['run_id'],file_hash(config_path),file_hash(receipt))
        if key in seen:continue
        seen.add(key)
        rows.append({'run_id':key[0],'historical_config_sha256':key[1],
            'historical_init_receipt_sha256':key[2],
            'historical_files_unchanged':True,'corrected_policy':_vggt_policy(config)})
    demo=root/'third_party'/'vggt'/'demo_colmap.py'
    atomic_json(out,{'schema':1,'correction':'effective configuration interpretation only',
        'historical_artifacts_modified':False,'pinned_demo_sha256':file_hash(demo),
        'source_facts':{
            'official_use_ba_branch':['VGGSfM tracking','track/visibility/reprojection filtering',
                'tracked-point COLMAP construction','SIMPLE_PINHOLE by default','pycolmap.bundle_adjustment'],
            'official_feedforward_branch':['depth confidence threshold','seeded random truncation to at most 100000 points',
                'direct depth-point COLMAP construction','PINHOLE','no bundle adjustment'],
            'attribution_limit':'Historical branch-level results do not isolate BA, camera changes, or point changes.'},
        'records':rows})
    print(out)


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
