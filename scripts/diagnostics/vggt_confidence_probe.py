#!/usr/bin/env python3
"""Read-only confidence distribution probe for the pinned VGGT feed-forward branch."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from bgs.core import WorkflowError,atomic_json,file_hash,read_json


RUN_CALL="extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)"


class ProbeComplete(Exception):pass


def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--demo',required=True);p.add_argument('--scene-dir',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--threshold',type=float,default=5.)
    args=p.parse_args();demo=Path(args.demo).resolve();scene=Path(args.scene_dir).resolve();out=Path(args.out).resolve()
    if out.exists():raise WorkflowError('Refusing to overwrite confidence probe')
    pins=read_json(Path(__file__).resolve().parents[2]/'upstream.lock.json')
    if file_hash(demo)!=pins['vggt']['files']['demo_colmap.py']:
        raise WorkflowError('VGGT confidence probe requires the exact pinned demo')
    if not (scene/'staging.json').is_file():raise WorkflowError('Confidence probe requires a train-only staging receipt')
    sys.path.insert(0,str(demo.parent))
    source=demo.read_text()
    if source.count(RUN_CALL)!=1:raise WorkflowError('Pinned VGGT inference call contract changed')
    patched=source.replace(RUN_CALL,RUN_CALL+'\n    __bgs_confidence_probe__(depth_conf)')
    result={}
    def probe(depth_conf):
        a=np.asarray(depth_conf,dtype=float);finite=a[np.isfinite(a)];accepted=finite>=args.threshold
        result.update({'schema':1,'demo_sha256':file_hash(demo),'scene_id':scene.name,'seed':args.seed,
            'threshold':args.threshold,'shape':list(a.shape),'total_values':int(a.size),'finite_values':int(finite.size),
            'quantiles':{str(q):float(np.quantile(finite,q)) for q in (0,.25,.5,.75,.9,.95,.99,.999,1)},
            'accepted_before_limit':int(accepted.sum()),'accepted_fraction':float(accepted.mean()),
            'accepted_after_seeded_random_100k_limit':min(100000,int(accepted.sum())),
            'interpretation':'The threshold and 100k random limit are effective only in the official feed-forward/no-BA branch.'})
        raise ProbeComplete()
    namespace={'__name__':'__main__','__file__':str(demo),'__bgs_confidence_probe__':probe}
    old=sys.argv
    try:
        sys.argv=[str(demo),'--scene_dir',str(scene),'--seed',str(args.seed),'--conf_thres_value',str(args.threshold)]
        try:exec(compile(patched,str(demo),'exec'),namespace,namespace)
        except ProbeComplete:pass
        if not result:raise WorkflowError('VGGT probe hook did not execute')
        out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,result);print(out)
    finally:sys.argv=old


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
