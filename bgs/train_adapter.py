"""Pinned gsplat adapter: all selected training images, independent held-out
evaluation, deterministic sample stream, and full-state checkpoints.

GPU integration and interrupted-vs-uninterrupted equivalence MUST be qualified
on the target host before resume is described as validated. This is not an
upstream feature or a novel reconstruction algorithm.
"""
from __future__ import annotations
import ast
import copy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types

from .core import WorkflowError, atomic_json, read_json, file_hash, digest, validate_metrics
from .splat_numerics import (DEDUP_POLICY,FLOOR_POLICY,POLICIES,UPSTREAM_POLICY,
                            first_nonfinite_splat,guarded_log_scales,stable_exact_deduplicate)


def patch_train_function(source):
    """Attach narrowly scoped hooks to the pinned source without modifying it."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    hits = {"start": 0, "loader": 0, "restore": 0, "loop": 0, "finite": 0, "pre_backward":0,
            "stop":0,"sample":0,"forward":0,"scheduler":0,"strategy":0}
    class Patch(ast.NodeTransformer):
        def visit_Assign(self, node):
            name = node.targets[0].id if len(node.targets)==1 and isinstance(node.targets[0], ast.Name) else None
            if name == "init_step":
                node.value = ast.parse("self._resume_step", mode="eval").body
                hits["start"] += 1
            elif name == "trainloader":
                node.value = ast.parse("self._make_trainloader()", mode="eval").body
                hits["loader"] += 1
            elif name == "global_tic":
                hits["restore"] += 1
                return ast.parse("self._restore_training_state(schedulers)").body + [node]
            elif name == "pbar":
                node.value=ast.parse("tqdm.tqdm(range(init_step, min(max_steps, self._diagnostic_stop_step or max_steps)))",mode="eval").body
                hits["stop"]+=1
            node=self.generic_visit(node)
            if name=="image_ids":
                hits["sample"]+=1
                return [node]+ast.parse("self._trace_point('sample', step, locals())").body
            target=node.targets[0] if len(node.targets)==1 else None
            if isinstance(target,(ast.Tuple,ast.List)) and [x.id for x in target.elts if isinstance(x,ast.Name)]==["renders","alphas","info"]:
                hits["forward"]+=1
                return [node]+ast.parse("self._trace_point('forward', step, locals())").body
            return node
        def visit_For(self, node):
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id == "step":
                hits["loop"] += 1
                node.body += ast.parse("self._after_step(step + 1, schedulers)\nself._trace_point('after_step', step, locals())").body
            elif isinstance(node.target,ast.Name) and node.target.id=="scheduler":
                hits["scheduler"]+=1
                node.body += ast.parse("self._trace_point('post_update', step, locals())").body
            return node
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "backward" and isinstance(call.func.value, ast.Name) and call.func.value.id == "loss":
                hits["finite"] += 1
                hits["pre_backward"] += 1
                return ast.parse("self._trace_point('pre_backward', step, locals())\nif not torch.isfinite(loss).all():\n    raise FloatingPointError('Non-finite training loss')").body + [node]+ast.parse("self._trace_point('backward', step, locals())").body
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=="step_post_backward":
                hits["strategy"]+=1
                return ast.parse("self._trace_point('pre_strategy', step, locals())").body+[node]+ast.parse("self._trace_point('post_strategy', step, locals())").body
            return self.generic_visit(node)
    fn = Patch().visit(fn)
    expected={key:1 for key in hits};expected["strategy"]=2
    if hits!=expected:
        raise WorkflowError(f"Upstream training contract changed: {hits}")
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    return compile(module, "<bgs_pinned_train_hooks>", "exec")


def sized_tensor(image, K, max_edge):
    import numpy as np
    import torch
    from PIL import Image
    h,w = image.shape[:2]
    ratio = min(1., max_edge/max(h,w))
    nh,nw = max(1,round(h*ratio)),max(1,round(w*ratio))
    if min(nh,nw) < 64:
        raise WorkflowError("Resized images need both dimensions >=64 for the fixed evaluation protocol")
    if (nh,nw)!=(h,w):
        im = Image.fromarray(image.cpu().numpy().astype(np.uint8), mode="RGB")
        image = torch.from_numpy(np.array(im.resize((nw,nh),Image.Resampling.LANCZOS),copy=True))
        K = K.clone(); K[0] *= nw/w; K[1] *= nh/h
    return image.float(),K


def viewer_camera_parameters(data):
    import numpy as np
    c2w=np.asarray(data["camtoworld"],dtype=float)
    K=np.asarray(data["K"],dtype=float)
    h,w=map(int,data["image"].shape[:2])
    if c2w.shape!=(4,4) or K.shape!=(3,3) or K[1,1]<=0:
        raise WorkflowError("Invalid calibrated viewer camera")
    return c2w,2*np.arctan(h/(2*K[1,1])),w/h


def apply_viewer_camera(camera,data,rotation_from_matrix):
    """Apply the writable calibrated pose/FOV fields of a viser camera."""
    c2w,fov,source_aspect=viewer_camera_parameters(data)
    camera.wxyz=rotation_from_matrix(c2w[:3,:3]).wxyz
    camera.position=c2w[:3,3]
    camera.fov=float(fov)
    # viser derives aspect from the connected browser viewport and exposes it
    # read-only.  The calibrated vertical FOV remains valid as the canvas is
    # resized or letterboxed.
    return source_aspect


def install_viewer_camera_presets(runner,manifest):
    """Expose calibrated train/held-out viewpoints without changing rendering."""
    import numpy as np
    import viser.transforms as vt
    train_names=list(runner.parser.image_names)
    holdout_names=list(manifest["holdout"])
    options=[f"training | {name}" for name in train_names]+[
             f"held-out | {name}" for name in holdout_names]
    lookup={option:("training",i) for i,option in enumerate(options[:len(train_names)])}
    lookup.update({option:("held-out",i) for i,option in enumerate(options[len(train_names):])})

    def record(option):
        split,index=lookup[option]
        return runner.trainset[index] if split=="training" else runner.valset[index]

    def move(client,data):
        with runner.server.atomic():
            apply_viewer_camera(client.camera,data,vt.SO3.from_matrix)

    with runner.server.gui.add_folder("Calibrated camera presets"):
        runner.server.gui.add_markdown(
            "Choose a **training** or genuine **held-out** calibrated view, then use the mouse to orbit. "
            "Held-out RGB is used only for evaluation and is not shown in the interactive renderer.")
        selector=runner.server.gui.add_dropdown("View",options=options,initial_value=options[0])
        go=runner.server.gui.add_button("Go to selected view")
        @go.on_click
        def _go(event):
            move(event.client,record(selector.value))

    @runner.server.on_client_connect
    def _default_camera(client):
        move(client,record(options[0]))


def load_upstream(spec):
    root = Path(spec["gsplat_root"]).resolve()
    lockfile = read_json(Path(__file__).resolve().parents[1]/"upstream.lock.json")
    for name, expected in lockfile["gsplat"]["files"].items():
        if file_hash(root/name) != expected:
            raise WorkflowError(f"Pinned gsplat source changed: {name}; explicitly requalify a new adapter version")
    sys.path.insert(0,str(root/"examples"))
    sys.path.insert(0,str(root))
    module = importlib.import_module("simple_trainer")
    import gsplat
    if not Path(gsplat.__file__).resolve().is_relative_to(root):
        raise WorkflowError("Imported gsplat differs from the pinned checkout")
    return module,lockfile


def make_runner(spec, manifest, training_dir, output, viewer=False, read_only=False):
    import numpy as np
    import torch
    from .geometry import fit_similarity, reference_to_model, rotation_error_degrees, perturb_training_poses
    module,pins = load_upstream(spec)
    if not hasattr(module,"_bgs_original_seed"):
        module._bgs_original_seed=module.set_random_seed
        module._bgs_original_dataset=module.Dataset
        module._bgs_original_parser=module.Parser
    original_seed = module._bgs_original_seed
    module.set_random_seed = lambda _unused: original_seed(spec["seed"])
    BaseDataset = module._bgs_original_dataset
    BaseParser = module._bgs_original_parser
    policy=spec.get("splat_numerics_policy",UPSTREAM_POLICY)
    if policy not in POLICIES:raise WorkflowError(f"Unknown splat_numerics_policy: {policy}")
    if policy==DEDUP_POLICY:
        class DeduplicatingParser(BaseParser):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                points,colors,errors,indices,_,receipt=stable_exact_deduplicate(
                    self.points,self.points_rgb,self.points_err,self.point_indices)
                self.points=points;self.points_rgb=colors;self.points_err=errors;self.point_indices=indices
                self._bgs_splat_numerics=receipt
        module.Parser=DeduplicatingParser
    else:
        module.Parser=BaseParser
    class AllSelectedDataset(BaseDataset):
        def __init__(self, parser, split="train", **kwargs):
            super().__init__(parser,split=split,**kwargs)
            self.indices = np.arange(len(parser.image_names)) if split=="train" else np.array([],dtype=int)
        def __getitem__(self,item):
            data=super().__getitem__(item)
            data["image"],data["K"] = sized_tensor(data["image"],data["K"],spec["max_edge"])
            return data
    module.Dataset = AllSelectedDataset
    cfg=module.Config()
    settings={"data_dir":str(training_dir),"result_dir":str(output),"data_factor":1,
              "max_steps":spec["steps"],"batch_size":1,"eval_steps":[],"save_steps":[],
              "save_ply":False,"disable_video":True,"disable_viewer":not viewer,
              "load_exposure":False,"cache_images":bool(spec.get("cache_images",False)),
              "pose_opt":bool(spec.get("pose_opt",False)),"pose_noise":0.,
              "app_opt":False,"post_processing":None,"depth_loss":False,
              "use_color_correction_metric":False,"lpips_net":"alex","port":spec.get("viewer_port",8080)}
    for name,value in settings.items():
        if not hasattr(cfg,name):
            raise WorkflowError(f"Missing upstream option: {name}")
        setattr(cfg,name,value)
    if viewer:
        from functools import partial
        module.viser.ViserServer=partial(module.viser.ViserServer,host="127.0.0.1")
    runner=module.Runner(0,0,1,cfg)
    numerics=getattr(runner.parser,"_bgs_splat_numerics",{"policy":policy,"input_points":len(runner.parser.points),
        "output_points":len(runner.parser.points),"point_count_changed":False,"colors_changed":False,
        "spatial_coverage_changed":False})
    if policy==FLOOR_POLICY:
        points=torch.from_numpy(runner.parser.points).float()
        distances=module.knn(points,4)
        guarded,guard_receipt=guarded_log_scales(points,distances,runner.scene_scale,runner.cfg.init_scale)
        current=runner.splats["scales"].detach().cpu();changed_rows=(current!=guarded).any(-1)
        if int(changed_rows.sum())!=guard_receipt["clamped_point_count"]:
            raise WorkflowError("Scale guard changed values outside its declared clamped set")
        with torch.no_grad():runner.splats["scales"].copy_(guarded.to(runner.device))
        numerics=guard_receipt
    first_invalid=first_nonfinite_splat(runner.splats)
    numerics={**numerics,"first_invalid_after_policy":first_invalid,
              "adapter_policy_field":"splat_numerics_policy"}
    if not read_only:atomic_json(Path(output)/"initialization_numerics.json",numerics)
    if first_invalid is not None and not spec.get("splat_numerics_allow_nonfinite_audit",False):
        raise FloatingPointError(f"Non-finite initial splat state: {first_invalid}")
    if set(runner.parser.image_names)!=set(manifest["train"]):
        raise WorkflowError("Initialization registration differs from the complete training split")
    if len(runner.parser.points)<4 or not np.isfinite(runner.parser.points).all():
        raise WorkflowError("Insufficient/non-finite initialization points")
    refs=read_json(manifest["reference"])["cameras"]
    model_poses=runner.parser.camtoworlds.copy()
    ref_poses=np.array([refs[n]["c2w"] for n in runner.parser.image_names])
    scale,rotation,translation,residuals=fit_similarity(model_poses[:,:3,3],ref_poses[:,:3,3])
    orientations=[rotation_error_degrees(rotation@m[:3,:3],r[:3,:3]) for m,r in zip(model_poses,ref_poses)]
    if not read_only:
        atomic_json(Path(output)/"alignment.json",{"source":"training_camera_centers_only",
            "scale":scale,"R":rotation.tolist(),"t":translation.tolist(),
            "training_names":runner.parser.image_names,"center_residuals":residuals.tolist(),
            "orientation_errors_degrees":orientations,"reference_source":manifest["reference_source"]})
    # Alignment uses unperturbed input poses, then only training observations are
    # perturbed. Evaluation poses and the initial points remain fixed.
    runner.parser.camtoworlds=perturb_training_poses(model_poses,spec["noise_seed"],
                spec.get("rotation_degrees",0),spec.get("translation_fraction",0))
    from PIL import Image
    class HeldoutDataset:
        def __len__(self): return len(manifest["holdout"])
        def __getitem__(self,index):
            name=manifest["holdout"][index]; record=refs[name]
            with Image.open(Path(manifest["image_root"])/name) as image:
                array=np.array(image,copy=True)
            image=torch.from_numpy(array)
            K=torch.tensor(record["K"],dtype=torch.float32)
            image,K=sized_tensor(image,K,spec["max_edge"])
            pose=reference_to_model(record["c2w"],scale,rotation,translation)
            return {"name":name,"image":image,"K":K,
                    "camtoworld":torch.tensor(pose,dtype=torch.float32),"camera_idx":0}
    runner.valset=HeldoutDataset()
    runner._resume_step=0
    runner._resume_payload=None
    runner._upstream=module
    runner._spec=spec
    runner._signature=digest({"spec":spec,"manifest":manifest,"pins":pins,
        "torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)})
    runner._checkpoint=Path(output)/"resume.pt"
    runner._manifest=manifest
    runner._milestone_eval_seconds=0.
    runner._diagnostic_stop_step=None
    runner._trace_point=lambda label,step,local_vars:None
    def make_loader(self):
        n=len(self.trainset)
        sequence=[]
        cache={}
        for step in range(self._resume_step,self.cfg.max_steps):
            epoch,index=divmod(step,n)
            if epoch not in cache:
                cache={epoch:np.random.default_rng(self._spec["seed"]+epoch).permutation(n)}
            sequence.append(int(cache[epoch][index]))
        return torch.utils.data.DataLoader(self.trainset,batch_size=1,sampler=sequence,
                    num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(self._spec["seed"]))
    def restore(self,schedulers):
        p=self._resume_payload
        if p is None: return
        for name,opt in self.optimizers.items(): opt.load_state_dict(p["optimizers"][name])
        for opt,state in zip(self.pose_optimizers,p["pose_optimizers"]): opt.load_state_dict(state)
        if len(schedulers)!=len(p["schedulers"]): raise WorkflowError("Scheduler mismatch")
        for sched,state in zip(schedulers,p["schedulers"]): sched.load_state_dict(state)
        self.strategy_state=p["strategy_state"]
        random.setstate(p["rng_python"]); np.random.set_state(p["rng_numpy"])
        torch.set_rng_state(p["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in p["rng_cuda"]])
    def after(self,next_step,schedulers):
        if len(self.splats["means"])>self._spec["max_gaussians"]:
            raise WorkflowError("Gaussian budget exceeded; no silent pruning or changed baseline")
        if next_step % self._spec["checkpoint_every"] and next_step != self.cfg.max_steps:
            return
        if not all(torch.isfinite(x).all().item() for x in self.splats.values()):
            raise FloatingPointError("Non-finite model state")
        payload={"schema":1,"signature":self._signature,"next_step":next_step,
           "splats":self.splats.state_dict(),"optimizers":{k:v.state_dict() for k,v in self.optimizers.items()},
           "pose_adjust":self.pose_adjust.state_dict() if self.cfg.pose_opt else None,
           "pose_optimizers":[x.state_dict() for x in self.pose_optimizers],
           "schedulers":[x.state_dict() for x in schedulers],"strategy_state":self.strategy_state,
           "rng_python":random.getstate(),"rng_numpy":np.random.get_state(),
           "rng_torch":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all()}
        checkpoint_dir=Path(self.cfg.result_dir)/"checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        target=checkpoint_dir/f"step_{next_step:08d}.pt"
        temp=target.with_suffix(".tmp.pt")
        try:
            torch.save(payload,temp)
            with temp.open("rb") as f: os.fsync(f.fileno())
            os.replace(temp,target)
        finally:
            temp.unlink(missing_ok=True)
        progress_file=Path(self.cfg.result_dir)/"progress.json"
        previous=read_json(progress_file).get("retained_checkpoints",[]) if progress_file.exists() else []
        milestone_paths={str(checkpoint_dir/f"step_{step:08d}.pt") for step in self._spec.get("diagnostic_steps",[])}
        pinned=[path for path in previous if path in milestone_paths]
        rolling=[path for path in previous if path not in milestone_paths]
        retained=list(dict.fromkeys(pinned+(rolling+[str(target)])[-2:]))
        atomic_json(progress_file,{"next_step":next_step,
               "total_steps":self.cfg.max_steps,"checkpoint":str(target),"checkpoint_sha256":file_hash(target),
               "num_gaussians":len(self.splats["means"]),"signature":self._signature,
               "retained_checkpoints":retained})
        self._checkpoint=target
        # Only obsolete checkpoints explicitly tracked by this run are removed;
        # completed models and other experiments are never touched.
        for old in previous:
            old_path=Path(old).resolve()
            if old not in retained and old_path.is_relative_to(checkpoint_dir.resolve()):
                old_path.unlink(missing_ok=True)
        if next_step in self._spec.get("diagnostic_steps",[]):
            milestone_started=time.perf_counter()
            evaluate(self,self._manifest,Path(self.cfg.result_dir)/"milestones"/f"step_{next_step:08d}")
            self._milestone_eval_seconds+=time.perf_counter()-milestone_started
    runner._make_trainloader=types.MethodType(make_loader,runner)
    runner._restore_training_state=types.MethodType(restore,runner)
    runner._after_step=types.MethodType(after,runner)
    namespace={}
    exec(patch_train_function((Path(spec["gsplat_root"])/"examples/simple_trainer.py").read_text()),module.__dict__,namespace)
    runner.train=types.MethodType(namespace["train"],runner)
    return runner,module


def load_checkpoint(runner):
    import torch
    progress_path=Path(runner.cfg.result_dir)/"progress.json"
    if not progress_path.exists(): return False
    progress=read_json(progress_path)
    runner._checkpoint=Path(progress["checkpoint"]).resolve()
    if not runner._checkpoint.is_relative_to(Path(runner.cfg.result_dir).resolve()/"checkpoints"):
        raise WorkflowError("Checkpoint pointer outside this run")
    if file_hash(runner._checkpoint)!=progress["checkpoint_sha256"]:
        raise WorkflowError("Checkpoint/progress mismatch; restore an intact pair")
    # Only this workflow's own local checkpoint is loaded; never use this path
    # to deserialize an untrusted downloaded pickle.
    p=torch.load(runner._checkpoint,map_location=runner.device,weights_only=False)
    if p["schema"]!=1 or p["signature"]!=runner._signature:
        raise WorkflowError("Checkpoint inputs/environment changed; start a separate experiment")
    if not 0<=p["next_step"]<=runner.cfg.max_steps:
        raise WorkflowError("Invalid resume step")
    if set(p["splats"])!=set(runner.splats): raise WorkflowError("Model schema changed")
    for key,param in runner.splats.items(): param.data=p["splats"][key].to(runner.device)
    module=runner._upstream
    runner.scene=module.GaussianScene.from_splats(runner.splats,id="scene")
    runner.splats=runner.scene.splats
    runner.stage=module.Stage()
    runner.stage.add_scene(runner.scene,runner.rasterize_splats)
    if runner.cfg.pose_opt: runner.pose_adjust.load_state_dict(p["pose_adjust"])
    runner._resume_payload=p; runner._resume_step=p["next_step"]
    return True


def evaluate(runner,manifest,output):
    import numpy as np
    import torch
    from PIL import Image
    output=Path(output); (output/"renders").mkdir(parents=True,exist_ok=True)
    def render(data):
        image=data["image"].to(runner.device)/255.
        h,w=image.shape[:2]
        color,_,_=runner.stage.render(runner.scene.id,
             camtoworlds=data["camtoworld"][None].to(runner.device),Ks=data["K"][None].to(runner.device),
             width=w,height=h,sh_degree=runner.cfg.sh_degree,near_plane=runner.cfg.near_plane,
             far_plane=runner.cfg.far_plane,frame_idcs=None,
             camera_idcs=torch.tensor([0],device=runner.device),exposure=None)
        return color.clamp(0,1),image[None]
    rows=[]
    with torch.no_grad():
        for i in range(len(runner.valset)):
            data=runner.valset[i]
            pred,target=render(data)
            if not torch.isfinite(pred).all(): raise FloatingPointError("Non-finite render")
            p,t=pred.permute(0,3,1,2),target.permute(0,3,1,2)
            row={"name":data["name"],"width":int(pred.shape[2]),"height":int(pred.shape[1])}
            for key in ("psnr","ssim","lpips"):
                metric=getattr(runner,key); metric.reset()
                row[key]=float(metric(p,t).item())
            error=(target-pred).abs().mean(-1,keepdim=True).repeat(1,1,1,3)
            canvas=torch.cat([target,pred,error],dim=2)[0].cpu().numpy()
            name=f"renders/{i:04d}.png"
            Image.fromarray((np.clip(canvas,0,1)*255).round().astype(np.uint8)).save(output/name)
            row["visualization"]=name; rows.append(row)
        # GPU render-only benchmark: inputs already on the device; transfers,
        # quality metrics and file writing are outside the timed region.
        first=runner.valset[0]
        cam=first["camtoworld"][None].to(runner.device); K=first["K"][None].to(runner.device)
        h,w=first["image"].shape[:2]
        kwargs=dict(camtoworlds=cam,Ks=K,width=w,height=h,sh_degree=runner.cfg.sh_degree,
                    near_plane=runner.cfg.near_plane,far_plane=runner.cfg.far_plane,
                    frame_idcs=None,camera_idcs=torch.tensor([0],device=runner.device),exposure=None)
        for _ in range(5): runner.stage.render(runner.scene.id,**kwargs)
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(30): runner.stage.render(runner.scene.id,**kwargs)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    result={"schema":1,"status":"VALID","images":rows,"protocol":"raw_rgb_full_image_lpips_alex",
        "reference_source":manifest["reference_source"],"alignment":"training_cameras_only",
        "render_fps":30/elapsed,"fps_scope":"one held-out camera; render only; 5 warmup + 30 repeats",
        "fps_resolution":[w,h],"num_gaussians":len(runner.splats["means"]),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    atomic_json(output/"metrics.json",result)
    result=validate_metrics(output/"metrics.json",manifest["holdout"])
    atomic_json(output/"metrics.json",result)
    return result


def execute(spec,manifest,training_dir,output,viewer=False):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise WorkflowError("Select exactly one CUDA device for this qualified single-GPU adapter")
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    runner,module=make_runner(spec,manifest,training_dir,output,viewer)
    resumed=load_checkpoint(runner)
    if viewer:
        if not resumed: raise WorkflowError("Viewer requires an existing workflow checkpoint")
        install_viewer_camera_presets(runner,manifest)
        runner.viewer.complete()
        print(f"Viewer bound to localhost:{spec.get('viewer_port',8080)}",flush=True)
        while True: time.sleep(1)
    torch.cuda.synchronize()
    started=time.perf_counter(); initial=runner._resume_step
    if initial<spec["steps"]: runner.train()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    module.export_splats(means=runner.splats["means"],scales=runner.splats["scales"],
         quats=runner.splats["quats"],opacities=runner.splats["opacities"],
         sh0=runner.splats["sh0"],shN=runner.splats["shN"],format="ply",save_to=str(output/"model.ply"))
    result=evaluate(runner,manifest,output)
    result.update({"resumed_from_step":initial,"training_seconds_this_attempt":elapsed,
                   "milestone_evaluation_seconds":runner._milestone_eval_seconds,
                   "optimization_seconds_this_attempt":max(0.,elapsed-runner._milestone_eval_seconds),
                   "model_bytes":(output/"model.ply").stat().st_size,
                   "resume_qualification":"IMPLEMENTED_NOT_GPU_QUALIFIED"})
    atomic_json(output/"metrics.json",result)
    atomic_json(output/"complete.json",{"status":"VALID","steps":spec["steps"],
                  "metrics_sha256":file_hash(output/"metrics.json"),"signature":runner._signature})
