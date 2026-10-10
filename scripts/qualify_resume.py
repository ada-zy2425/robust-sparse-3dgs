#!/usr/bin/env python3
"""GPU-only interrupted/uninterrupted comparison, charged to the campaign.
Run after initialization exists. Does not constitute a reconstruction result."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bgs.core import Engine,WorkflowError,read_json,atomic_json,file_hash
from bgs.pipeline import load_spec


def selected_visible_device(spec):
    """Retain an outer queue assignment; fall back to the submitted config."""
    return os.environ.get('CUDA_VISIBLE_DEVICES',str(spec['gpu']))


def _cpu_tree(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _compare_tree(left,right):
    import numpy as np
    import torch
    out={'structure_match':True,'exact':True,'tensor_count':0,'max_absolute_difference':0.}
    def visit(x,y):
        if torch.is_tensor(x) and torch.is_tensor(y):
            out['tensor_count']+=1
            if x.shape!=y.shape or x.dtype!=y.dtype:
                out['structure_match']=False;out['exact']=False;return
            if not torch.equal(x,y):
                out['exact']=False
                if x.numel() and (x.dtype.is_floating_point or x.dtype.is_complex):
                    finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    out['max_absolute_difference']=max(out['max_absolute_difference'],
                        float((x-y).abs().max()) if finite else float('inf'))
        elif isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            out['tensor_count']+=1
            if x.shape!=y.shape or x.dtype!=y.dtype:
                out['structure_match']=False;out['exact']=False
            elif not np.array_equal(x,y):
                out['exact']=False
                if x.size and np.issubdtype(x.dtype,np.number):
                    out['max_absolute_difference']=max(out['max_absolute_difference'],float(np.max(np.abs(x-y))))
        elif isinstance(x,dict) and isinstance(y,dict):
            if set(x)!=set(y):out['structure_match']=False;out['exact']=False;return
            for key in x:visit(x[key],y[key])
        elif isinstance(x,(list,tuple)) and isinstance(y,type(x)):
            if len(x)!=len(y):out['structure_match']=False;out['exact']=False;return
            for a,b in zip(x,y):visit(a,b)
        elif x!=y:out['structure_match']=False;out['exact']=False
    visit(left,right);return out


def _semantic_trace_record(record):
    """Exclude non-contributing screen coordinates for culled Gaussians.

    The dense CUDA projection output may leave ``means2d`` entries for
    radius-zero Gaussians dependent on execution history. Those entries do
    not contribute to ``renders`` or ``alphas`` and must not turn an otherwise
    exact forward pass into a reported model-state divergence.  The raw trace
    is still retained and compared separately.
    """
    import torch
    out=_cpu_tree(record)
    info=out.get('render_info',{})
    means2d=info.get('means2d');radii=info.get('radii')
    if torch.is_tensor(means2d) and torch.is_tensor(radii) and means2d.shape[:-1]==radii.shape:
        means2d[radii<=0]=0
    return out


def _trace_local_names(label):
    names=['image_ids']
    if label!='sample':names+=['renders','alphas']
    if label in ('pre_backward','backward','post_update','pre_strategy','post_strategy','after_step'):names+=['loss']
    return names


def _trace_recorder(runner,minimum_step):
    import numpy as np
    import torch
    trace={}
    def record(label,step,local_vars):
        if step<minimum_step:return
        optimizers={}
        for name,opt in runner.optimizers.items():
            params=[p for group in opt.param_groups for p in group['params']]
            optimizers[name]={'bound_to_current_parameter':len(params)==1 and params[0] is runner.splats[name],
                              'state':_cpu_tree(opt.state_dict())}
        record={
            'splats':_cpu_tree(runner.splats.state_dict()),
            'grads':{k:_cpu_tree(v.grad) if v.grad is not None else None for k,v in runner.splats.items()},
            'optimizers':optimizers,'strategy_state':_cpu_tree(runner.strategy_state),
            'scene_is_stage_scene':runner.stage.get_scene(runner.scene.id) is runner.scene,
            'splats_is_scene_splats':runner.splats is runner.scene.splats,
            'component_index':_cpu_tree(runner.scene.component_index),
            'rng_torch':torch.get_rng_state().clone(),
            'rng_cuda':[x.cpu().clone() for x in torch.cuda.get_rng_state_all()],
            'rng_python':_cpu_tree(random.getstate()),
            'rng_numpy':_cpu_tree(np.random.get_state()),
        }
        for name in _trace_local_names(label):
            if name in local_vars:record[name]=_cpu_tree(local_vars[name])
        if label!='sample' and 'info' in local_vars:
            record['render_info']={k:_cpu_tree(local_vars['info'][k]) for k in
                ('radii','gaussian_ids','means2d') if k in local_vars['info']}
        if 'schedulers' in local_vars:record['schedulers']=_cpu_tree([x.state_dict() for x in local_vars['schedulers']])
        trace[f'{step}:{label}']=record
    return trace,record


def _diagnose_first_step(args,spec,run,out,manifest):
    import torch
    from bgs.train_adapter import make_runner,load_checkpoint
    midpoint=args.steps//2
    diagnostic_spec={**spec,'steps':args.steps,'checkpoint_every':midpoint}
    branches={};traces={}

    continuous,_=make_runner(diagnostic_spec,manifest,run/'input',out/'continuous')
    continuous._diagnostic_stop_step=midpoint+1
    traces['continuous'],continuous._trace_point=_trace_recorder(continuous,midpoint)
    continuous.train()
    prefix=out/'continuous'/'checkpoints'/f'step_{midpoint:08d}.pt'
    if not prefix.is_file():raise WorkflowError('Diagnostic continuous branch did not persist its midpoint')
    branches['continuous']={'gaussians':len(continuous.splats['means'])}

    for name in ('resumed_a','resumed_b'):
        runner,_=make_runner(diagnostic_spec,manifest,run/'input',out/name)
        target=out/name/'checkpoints'/prefix.name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(prefix,target)
        atomic_json(out/name/'progress.json',{'next_step':midpoint,'total_steps':args.steps,
            'checkpoint':str(target.resolve()),'checkpoint_sha256':file_hash(target),'source':str(prefix)})
        if not load_checkpoint(runner):raise WorkflowError('Diagnostic checkpoint load failed')
        runner._diagnostic_stop_step=midpoint+1
        traces[name],runner._trace_point=_trace_recorder(runner,midpoint)
        # Binding and rebuilt Scene/Stage state before any resumed work.
        runner._trace_point('loaded',midpoint,{})
        runner.train()
        branches[name]={'gaussians':len(runner.splats['means'])}
        del runner;gc.collect();torch.cuda.empty_cache()

    order=['sample','forward','backward','post_update','pre_strategy','post_strategy','after_step']
    comparisons={};raw_comparisons={}
    first=None
    for label in order:
        key=f'{midpoint}:{label}'
        raw_comparisons[label]={
            'continuous_vs_resumed':_compare_tree(traces['continuous'][key],traces['resumed_a'][key]),
            'resumed_repeat':_compare_tree(traces['resumed_a'][key],traces['resumed_b'][key])}
        comparisons[label]={
            'continuous_vs_resumed':_compare_tree(_semantic_trace_record(traces['continuous'][key]),
                                                   _semantic_trace_record(traces['resumed_a'][key])),
            'resumed_repeat':_compare_tree(_semantic_trace_record(traces['resumed_a'][key]),
                                            _semantic_trace_record(traces['resumed_b'][key]))}
        if first is None and not comparisons[label]['continuous_vs_resumed']['exact']:first=label
    loaded={name:{
        'optimizer_binding':{k:v['bound_to_current_parameter'] for k,v in traces[name][f'{midpoint}:loaded']['optimizers'].items()},
        'scene_is_stage_scene':traces[name][f'{midpoint}:loaded']['scene_is_stage_scene'],
        'splats_is_scene_splats':traces[name][f'{midpoint}:loaded']['splats_is_scene_splats']}
        for name in ('resumed_a','resumed_b')}
    trace_path=out/'trace_tensors.pt';torch.save(traces,trace_path)
    atomic_json(out/'qualification.json',{
        'status':'DIAGNOSED','protocol_version':8,
        'comparison':'continuous first post-checkpoint step vs two independent reloads of the same prefix',
        'total_schedule_steps':args.steps,'checkpoint_step':midpoint,'diagnosed_step':midpoint,
        'first_continuous_vs_resumed_divergence':first,
        'same_condition_repeat_exact_through_step':all(row['resumed_repeat']['exact'] for row in comparisons.values()),
        'loaded_state':loaded,'stage_comparisons':comparisons,
        'raw_stage_comparisons':raw_comparisons,
        'semantic_comparison_note':'Radius-zero projected means do not contribute to RGB/alpha; raw differences remain recorded.',
        'branch_final':branches,
        'trace_tensors':str(trace_path),'trace_sha256':file_hash(trace_path),
        'sample_state_fields':['splats','gradients','optimizer binding/state/step/lr','scheduler state',
            'strategy state','Python/NumPy/Torch/CUDA RNG','camera sample','Scene/Stage references'],
        'interpretation':'The first contributing difference before strategy separates restore/execution state from later densification amplification; repeat comparison measures same-condition execution determinism.'})


def _diagnose_boundaries(args,spec,run,out,manifest):
    import torch
    from bgs.train_adapter import make_runner,load_checkpoint
    midpoint=args.steps//2;diagnostic_spec={**spec,'steps':args.steps,'checkpoint_every':midpoint}
    traces={};final={}
    def recorder(runner):
        trace={'samples':{},'post_strategy_gaussians':{}}
        def record(label,step,local_vars):
            if step<midpoint:return
            if label=='sample':trace['samples'][str(step)]=int(local_vars['image_ids'].detach().cpu().item())
            if label=='post_strategy' and step%100==0:
                trace['post_strategy_gaussians'][str(step)]=len(runner.splats['means'])
        return trace,record
    a,_=make_runner(diagnostic_spec,manifest,run/'input',out/'continuous')
    traces['continuous'],a._trace_point=recorder(a);a.train()
    prefix=out/'continuous'/'checkpoints'/f'step_{midpoint:08d}.pt'
    if not prefix.is_file():raise WorkflowError('Boundary diagnostic did not persist midpoint')
    final['continuous']=len(a.splats['means']);del a;gc.collect();torch.cuda.empty_cache()
    for name in ('resumed_a','resumed_b'):
        runner,_=make_runner(diagnostic_spec,manifest,run/'input',out/name)
        target=out/name/'checkpoints'/prefix.name;target.parent.mkdir(exist_ok=True);shutil.copy2(prefix,target)
        atomic_json(out/name/'progress.json',{'next_step':midpoint,'total_steps':args.steps,
            'checkpoint':str(target.resolve()),'checkpoint_sha256':file_hash(target),'source':str(prefix)})
        if not load_checkpoint(runner):raise WorkflowError('Boundary diagnostic checkpoint load failed')
        traces[name],runner._trace_point=recorder(runner);runner.train();final[name]=len(runner.splats['means'])
        del runner;gc.collect();torch.cuda.empty_cache()
    boundaries=sorted(set(traces['continuous']['post_strategy_gaussians']) &
                      set(traces['resumed_a']['post_strategy_gaussians']) &
                      set(traces['resumed_b']['post_strategy_gaussians']),key=int)
    first_continuous=next((s for s in boundaries if traces['continuous']['post_strategy_gaussians'][s]!=traces['resumed_a']['post_strategy_gaussians'][s]),None)
    first_repeat=next((s for s in boundaries if traces['resumed_a']['post_strategy_gaussians'][s]!=traces['resumed_b']['post_strategy_gaussians'][s]),None)
    samples_equal=(traces['continuous']['samples']==traces['resumed_a']['samples']==traces['resumed_b']['samples'])
    atomic_json(out/'qualification.json',{'status':'DIAGNOSED','protocol_version':6,
        'comparison':'structural decisions across all post-checkpoint densification boundaries plus same-condition repeat',
        'checkpoint_step':midpoint,'total_schedule_steps':args.steps,'sample_order_exact':samples_equal,
        'first_continuous_vs_resumed_gaussian_count_divergence_step':int(first_continuous) if first_continuous else None,
        'first_resumed_repeat_gaussian_count_divergence_step':int(first_repeat) if first_repeat else None,
        'boundary_gaussian_counts':{s:{name:traces[name]['post_strategy_gaussians'][s] for name in traces} for s in boundaries},
        'final_gaussians':final,
        'interpretation':'Equal sampling with a later structural split after v5 first found tiny backward differences tests numerical-perturbation amplification rather than missing checkpoint or sampling state.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--worker',action='store_true')
    p.add_argument('--require-shape-change',action='store_true')
    p.add_argument('--diagnose-first-divergence',action='store_true')
    p.add_argument('--diagnose-boundaries',action='store_true')
    args=p.parse_args();spec=load_spec(args.config)
    if args.steps<4 or args.steps%2:raise WorkflowError('Use an even number of at least 4 steps')
    if args.diagnose_first_divergence and args.diagnose_boundaries:raise WorkflowError('Choose one diagnostic mode')
    protocol_version=6 if args.diagnose_boundaries else (8 if args.diagnose_first_divergence else (3 if args.require_shape_change else 2))
    suffix=(f'resume_boundaries_{args.steps}_v6' if args.diagnose_boundaries else
            (f'resume_divergence_{args.steps}_v8' if args.diagnose_first_divergence else
            (f'resume_qualification_{args.steps}_shape_v3' if args.require_shape_change else f'resume_qualification_{args.steps}_v2')))
    root=Path(spec['root']);run=root/'runs'/spec['run_id'];out=run/suffix
    if not (run/'input/init_receipt.json').exists():raise WorkflowError('Complete initialization first')
    if not args.worker:
        with Engine(root,spec['gpu_hours']) as engine:
            argv=[spec['python'],str(Path(__file__).resolve()),'--config',str(Path(args.config).resolve()),'--steps',str(args.steps),'--worker']
            if args.require_shape_change:argv.append('--require-shape-change')
            if args.diagnose_first_divergence:argv.append('--diagnose-first-divergence')
            if args.diagnose_boundaries:argv.append('--diagnose-boundaries')
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
    manifest=read_json(spec['manifest'])
    if args.diagnose_first_divergence:
        _diagnose_first_step(args,spec,run,out,manifest);return
    if args.diagnose_boundaries:
        _diagnose_boundaries(args,spec,run,out,manifest);return
    spec={**spec,'steps':args.steps,'checkpoint_every':args.steps//2}
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
