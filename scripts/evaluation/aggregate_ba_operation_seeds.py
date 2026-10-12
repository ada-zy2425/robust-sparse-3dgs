#!/usr/bin/env python3
"""Aggregate paired BA-operation deltas across completed training seeds."""
import argparse
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
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


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
