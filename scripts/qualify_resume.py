#!/usr/bin/env python3
"""GPU-only interrupted/uninterrupted comparison, charged to the campaign.
Run after initialization exists. Does not constitute a reconstruction result."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bgs.core import Engine,WorkflowError,read_json,atomic_json,file_hash
from bgs.pipeline import load_spec


def selected_visible_device(spec):
    """Retain an outer queue assignment; fall back to the submitted config."""
    return os.environ.get('CUDA_VISIBLE_DEVICES',str(spec['gpu']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--worker',action='store_true')
    p.add_argument('--require-shape-change',action='store_true')
    args=p.parse_args();spec=load_spec(args.config)
    if args.steps<4 or args.steps%2:raise WorkflowError('Use an even number of at least 4 steps')
    protocol_version=3 if args.require_shape_change else 2
    suffix=f'resume_qualification_{args.steps}_shape_v3' if args.require_shape_change else f'resume_qualification_{args.steps}_v2'
    root=Path(spec['root']);run=root/'runs'/spec['run_id'];out=run/suffix
    if not (run/'input/init_receipt.json').exists():raise WorkflowError('Complete initialization first')
    if not args.worker:
        with Engine(root,spec['gpu_hours']) as engine:
            argv=[spec['python'],str(Path(__file__).resolve()),'--config',str(Path(args.config).resolve()),'--steps',str(args.steps),'--worker']
            if args.require_shape_change:argv.append('--require-shape-change')
            engine.run(spec['run_id'],suffix,
                argv,
                str(Path(__file__).resolve().parents[1]),
                {'CUDA_VISIBLE_DEVICES':selected_visible_device(spec),'CUDA_DEVICE_ORDER':'PCI_BUS_ID'},
                [out/'qualification.json'],spec['timeout_s'],fingerprint={'spec':spec,'steps':args.steps,
                    'protocol_version':protocol_version,'require_shape_change':args.require_shape_change,
                    'script_sha256':file_hash(__file__)})
        print((out/'qualification.json').read_text());return
    if out.exists():raise WorkflowError('Preserve earlier qualification output; use a new experiment for requalification')
    out.mkdir(parents=True)
    import numpy as np
    import torch
    from PIL import Image
    from bgs.train_adapter import evaluate,make_runner,load_checkpoint
    if torch.cuda.device_count()!=1:raise WorkflowError('Select exactly one GPU')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=read_json(spec['manifest']);spec={**spec,'steps':args.steps,'checkpoint_every':args.steps//2}
    a,_=make_runner(spec,manifest,run/'input',out/'continuous')
    initial_gaussians=len(a.splats['means'])
    a.train();reference={k:v.detach().cpu().clone() for k,v in a.splats.items()}
    continuous_metrics=evaluate(a,manifest,out/'continuous_eval')
    midpoint=args.steps//2
    prefix=out/'continuous'/'checkpoints'/f'step_{midpoint:08d}.pt'
    if not prefix.is_file():raise WorkflowError('Continuous branch did not persist the midpoint checkpoint')
    prefix_payload=torch.load(prefix,map_location='cpu',weights_only=False)
    midpoint_gaussians=len(prefix_payload['splats']['means'])
    if args.require_shape_change and midpoint_gaussians==initial_gaussians:
        atomic_json(out/'qualification.json',{'status':'FAIL_NO_SHAPE_CHANGE','protocol_version':protocol_version,
            'initial_gaussians':initial_gaussians,'midpoint_gaussians':midpoint_gaussians,
            'midpoint':midpoint,'next':'Choose an interruption after an observed densification/pruning event'})
        raise WorkflowError('Required Gaussian parameter shape change did not occur before the midpoint')
    del a;gc.collect();torch.cuda.empty_cache()
    resumed=out/'resumed'
    c,_=make_runner(spec,manifest,run/'input',resumed)
    target=resumed/'checkpoints'/prefix.name;target.parent.mkdir(exist_ok=True)
    shutil.copy2(prefix,target)
    atomic_json(resumed/'progress.json',{'next_step':midpoint,'total_steps':args.steps,
        'checkpoint':str(target.resolve()),'checkpoint_sha256':file_hash(target),
        'source':'same persisted prefix as uninterrupted branch'})
    if not load_checkpoint(c) or c._resume_step!=midpoint:raise WorkflowError('Did not resume at the intended optimizer step')
    c.train();resumed_metrics=evaluate(c,manifest,out/'resumed_eval');details={};passed=True
    for key,value in c.splats.items():
        value=value.detach().cpu();target=reference[key]
        if value.shape!=target.shape:
            details[key]={'shape_match':False,'actual':list(value.shape),'reference':list(target.shape)};passed=False
        else:
            finite=bool(torch.isfinite(value).all() and torch.isfinite(target).all())
            ok=finite and torch.allclose(value,target,rtol=1e-5,atol=1e-6)
            details[key]={'shape_match':True,'max_absolute_difference':float((value-target).abs().max()) if finite else None,'within_tolerance':bool(ok)}
            passed=passed and bool(ok)
    def compare_tree(left,right):
        summary={'structure_match':True,'tensor_count':0,'max_absolute_difference':0.,'within_tolerance':True}
        def visit(x,y):
            if torch.is_tensor(x) and torch.is_tensor(y):
                summary['tensor_count']+=1
                if x.shape!=y.shape or x.dtype!=y.dtype:
                    summary['structure_match']=False;summary['within_tolerance']=False;return
                if x.dtype.is_floating_point:
                    finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    diff=float((x-y).abs().max()) if finite and x.numel() else 0.
                    summary['max_absolute_difference']=max(summary['max_absolute_difference'],diff)
                    summary['within_tolerance']=summary['within_tolerance'] and finite and bool(torch.allclose(x,y,rtol=1e-5,atol=1e-6))
                else:summary['within_tolerance']=summary['within_tolerance'] and bool(torch.equal(x,y))
            elif isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
                summary['tensor_count']+=1
                if x.shape!=y.shape or x.dtype!=y.dtype or not np.array_equal(x,y):summary['within_tolerance']=False
            elif isinstance(x,dict) and isinstance(y,dict):
                if set(x)!=set(y):summary['structure_match']=False;summary['within_tolerance']=False;return
                for key in x:visit(x[key],y[key])
            elif isinstance(x,(list,tuple)) and isinstance(y,type(x)):
                if len(x)!=len(y):summary['structure_match']=False;summary['within_tolerance']=False;return
                for a_item,b_item in zip(x,y):visit(a_item,b_item)
            elif x!=y:summary['structure_match']=False;summary['within_tolerance']=False
        visit(left,right);return summary
    continuous_final=out/'continuous'/'checkpoints'/f'step_{args.steps:08d}.pt'
    resumed_final=out/'resumed'/'checkpoints'/f'step_{args.steps:08d}.pt'
    pa=torch.load(continuous_final,map_location='cpu',weights_only=False)
    pb=torch.load(resumed_final,map_location='cpu',weights_only=False)
    state_details={key:compare_tree(pa[key],pb[key]) for key in
                   ('optimizers','schedulers','strategy_state','rng_python','rng_numpy','rng_torch','rng_cuda')}
    passed=passed and all(row['within_tolerance'] and row['structure_match'] for row in state_details.values())
    render_rows=[];render_pass=True
    for row_a,row_b in zip(continuous_metrics['images'],resumed_metrics['images']):
        if row_a['name']!=row_b['name']:render_pass=False;continue
        left=np.asarray(Image.open(out/'continuous_eval'/row_a['visualization']),dtype=np.int16)
        right=np.asarray(Image.open(out/'resumed_eval'/row_b['visualization']),dtype=np.int16)
        width=row_a['width'];delta=np.abs(left[:,width:2*width]-right[:,width:2*width])
        metric_delta={name:abs(row_a[name]-row_b[name]) for name in ('psnr','ssim','lpips')}
        ok=delta.max()<=1 and max(metric_delta.values())<=1e-4
        render_rows.append({'name':row_a['name'],'max_prediction_uint8_difference':int(delta.max()),
            'mean_prediction_uint8_difference':float(delta.mean()),'metric_absolute_difference':metric_delta,
            'within_diagnostic_tolerance':bool(ok)})
        render_pass=render_pass and bool(ok)
    passed=passed and render_pass
    atomic_json(out/'qualification.json',{'status':'PASS_CONFIGURED_TOLERANCE' if passed else 'FAIL',
        'protocol_version':protocol_version,'comparison':'uninterrupted continuation vs reload from the same persisted midpoint prefix',
        'total_steps':args.steps,'interrupted_at':midpoint,'rtol':1e-5,'atol':1e-6,
        'initial_gaussians':initial_gaussians,'midpoint_gaussians':midpoint_gaussians,
        'final_gaussians_continuous':len(pa['splats']['means']),'final_gaussians_resumed':len(pb['splats']['means']),
        'shape_change_required':args.require_shape_change,'shape_change_observed':midpoint_gaussians!=initial_gaussians,
        'parameters':details,'training_state':state_details,'heldout_render_comparison':render_rows,
        'scope':'One same-GPU schedule and one persisted prefix; not proof of bitwise determinism across independent CUDA prefixes',
        'next':'Use continuous formal runs until this shape-changing qualification passes' if not passed else 'Record the exact strategy boundary and retain the qualification artifacts'})
    if not passed:
        raise WorkflowError('Resume qualification failed; review qualification.json before using resumed results')


if __name__=='__main__':main()
