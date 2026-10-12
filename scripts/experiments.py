#!/usr/bin/env python3
"""Build and freeze immutable experiment tasks. This CLI does not launch training.
Run --help for commands; COMMAND --help for its argument contract.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bgs.core import WorkflowError

def build_ba_ablation_benchmarks():
    import argparse

    from pathlib import Path

    import sys

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
                    'argv':[str(root/'envs'/'gsplat'/'bin'/'python'),str(root/'scripts'/'render_benchmark.py'),
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

    return main()

def build_ba_operation_ablation():
    import argparse

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError,atomic_json,file_hash,read_json

    from bgs.splat_numerics import POLICIES,UPSTREAM_POLICY

    def main():
        p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--revision',default='v1')
        p.add_argument('--seed',type=int,default=42);p.add_argument('--reuse-pair-revision');p.add_argument('--pair-seed',type=int,default=42)
        p.add_argument('--splat-numerics-policy',choices=sorted(POLICIES),default=UPSTREAM_POLICY)
        p.add_argument('--max-gpu-hours',type=float,default=1.25)
        p.add_argument('--timeout-s',type=int,default=4200)
        p.add_argument('--gpu-uuids',nargs=4,required=True,metavar=('GPU0','GPU1','GPU2','GPU3'))
        args=p.parse_args();gpu_uuids=dict(enumerate(args.gpu_uuids))
        if args.max_gpu_hours<=0:raise WorkflowError('max-gpu-hours must be positive')
        if args.timeout_s<=0 or args.timeout_s/3600>args.max_gpu_hours:
            raise WorkflowError('timeout-s must be positive and fit max-gpu-hours')
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
                    'gpu_hours':args.max_gpu_hours,'timeout_s':args.timeout_s,'seed':args.seed,
                    'splat_numerics_policy':args.splat_numerics_policy,
                    'vggt_ba_ablation_arm':arm,'vggt_ba_pair_root':str(pair_root),
                    'vggt_ba_pair_generate':generate,
                    'hypothesis':'Holding the official BA-branch reconstruction fixed, the pycolmap BA call changes held-out quality.',
                    'independent_contribution':'Same-snapshot operation ablation; does not separate camera from point optimization.',
                    'ablation_scope':'Only the pycolmap.bundle_adjustment call differs; official no-BA remains a separate pipeline.'}
                config.pop('control_of',None);config.pop('control_of_sha256',None)
                config_path=config_dir/f'{run_id}.json';atomic_json(config_path,config)
                run=Path(config['root'])/'runs'/run_id
                expected=[run/'summary.json',run/'input'/'init_receipt.json',run/'training'/'model.ply',
                          run/'training'/'metrics.json',run/'training'/'complete.json',
                          run/'training'/'initialization_numerics.json']
                task={'task_id':run_id,'kind':'bgs_run',
                    'argv':[config['python'],'-m','bgs','run','--config',str(config_path)],
                    'cwd':str(root),'env':{'PYTHONPATH':str(root),'TORCH_HOME':str(root/'cache'/'torch'),
                        'TMPDIR':str(root/'tmp'/f'ba_operation_ablation_{args.revision}'/run_id)},
                    'dependencies':dependencies,'config_path':str(config_path),'config_sha256':file_hash(config_path),
                    'gpu_index':gpu,'gpu_uuid':gpu_uuids[gpu],'gpu_count':1,'max_gpu_hours':args.max_gpu_hours,
                    'timeout_s':args.timeout_s,'disk_reservation_bytes':10*2**30,
                    'expected_outputs':[str(x) for x in expected],'priority':10 if generate else 20}
                task_path=task_dir/f'{run_id}.task.json';atomic_json(task_path,task)
                created.append({'run_id':run_id,'arm':arm,'scene':scene,'config':str(config_path),
                                'config_sha256':file_hash(config_path),'task':str(task_path)})
        atomic_json(task_dir/'matrix_receipt.json',{'schema':1,'revision':args.revision,'seed':args.seed,
            'scenes':['bonsai','room'],'fixed':['same persisted pre-BA reconstruction','images','tracks','initial points',
            'camera model','preprocessing','7000-step gsplat schedule','max_edge=1024'],
            'only_operation_difference':'pycolmap.bundle_adjustment executed for post_ba arm',
            'shared_splat_numerics_policy':args.splat_numerics_policy,
            'per_task_max_gpu_hours':args.max_gpu_hours,
            'per_task_timeout_s':args.timeout_s,
            'pair_source':{'revision':args.reuse_pair_revision or args.revision,'seed':args.pair_seed},
            'seed_scope':('Training randomness only; both arms reuse the immutable seed-42 initialization snapshots.'
                          if args.reuse_pair_revision else 'Initialization and training use the declared seed.'),
            'limit':'Does not separate camera optimization from point optimization.',
            'original_no_ba_branch':'Separate initialization pipeline; not an arm of this operation ablation.',
            'created':created})
        print(task_dir/'matrix_receipt.json')

    return main()

def build_confidence_probe_tasks():
    import argparse

    from pathlib import Path

    import sys

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

    return main()

def build_final_matrix():
    import argparse

    from pathlib import Path

    from bgs.core import WorkflowError, atomic_json, file_hash, read_json

    from bgs.data import validate_manifest, verify_inputs

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--project-root", required=True)
        parser.add_argument("--protocol", required=True)
        parser.add_argument("--scene", action="append", required=True,
                            help="scene_id:24-view-final-manifest")
        parser.add_argument("--gpu", action="append", required=True, help="INDEX:GPU-UUID")
        parser.add_argument("--dinov2-root", required=True)
        parser.add_argument("--max-gpu-hours", type=float, default=2.0)
        parser.add_argument("--timeout-s", type=int, default=5400)
        parser.add_argument("--initialization-reserve-s", type=int, default=900)
        args = parser.parse_args()

        root = Path(args.project_root).resolve()
        protocol_path = Path(args.protocol).resolve()
        protocol = read_json(protocol_path)
        if protocol.get("schema") != 1 or protocol.get("status") != "FROZEN_BEFORE_SCORING":
            raise WorkflowError("Final protocol is not frozen")
        if (args.initialization_reserve_s + args.timeout_s + 5) / 3600 > args.max_gpu_hours:
            raise WorkflowError("Initialization plus training cleanup exceeds task reservation")

        gpus = []
        for item in args.gpu:
            index, separator, uuid = item.partition(":")
            if not separator or not index.isdigit() or not uuid.startswith("GPU-"):
                raise WorkflowError(f"Invalid GPU mapping: {item}")
            gpus.append((int(index), uuid))
        if not 1 <= len(gpus) <= 4 or len({uuid for _, uuid in gpus}) != len(gpus):
            raise WorkflowError("Provide one to four unique GPU mappings")

        manifests = {}
        for value in args.scene:
            scene_id, separator, manifest_value = value.partition(":")
            if not separator or scene_id in manifests:
                raise WorkflowError(f"Invalid or duplicate scene mapping: {value}")
            manifest_path = Path(manifest_value).resolve()
            manifest = validate_manifest(read_json(manifest_path))
            verify_inputs(manifest)
            if (manifest["partition"] != "final" or manifest["scene_id"] != scene_id
                    or len(manifest["train"]) != 24
                    or Path(manifest["final_protocol"]).resolve() != protocol_path):
                raise WorkflowError(f"Final manifest contract mismatch: {manifest_path}")
            manifests[scene_id] = manifest_path
        if set(manifests) != set(protocol.get("scene_ids", [])):
            raise WorkflowError("Scene mappings differ from frozen protocol")

        matrix_id = protocol["matrix_id"]
        config_dir = root / "configs" / matrix_id
        task_dir = root / "remote_tasks" / matrix_id
        if config_dir.exists() or task_dir.exists():
            raise WorkflowError("Refusing to overwrite immutable final matrix")
        dinov2_root = Path(args.dinov2_root).resolve()
        expected_ids = set(protocol.get("run_ids", []))
        planned = []
        assignment = 0
        for scene_id in protocol["scene_ids"]:
            for initializer, ba in (("colmap", False), ("vggt", True)):
                arm = "vggt_ba" if ba else "colmap"
                run_id = f"{scene_id}_final_v24_{arm}_s42_7k_v1"
                if run_id not in expected_ids:
                    raise WorkflowError(f"Run was not predeclared: {run_id}")
                gpu_index, gpu_uuid = gpus[assignment % len(gpus)]
                assignment += 1
                planned.append((scene_id, initializer, ba, run_id, gpu_index, gpu_uuid))
        if {row[3] for row in planned} != expected_ids:
            raise WorkflowError("Frozen run matrix contains unsupported or missing entries")

        config_dir.mkdir(parents=True)
        task_dir.mkdir(parents=True)
        created = []
        for scene_id, initializer, ba, run_id, gpu_index, gpu_uuid in planned:
            manifest_path = manifests[scene_id]
            campaign = root / "work" / matrix_id / "campaigns" / run_id
            spec = {
                "run_id": run_id, "root": str(campaign), "manifest": str(manifest_path),
                "gsplat_root": str(root / "third_party" / "gsplat"),
                "python": str(root / "envs" / "gsplat" / "bin" / "python"),
                "initializer": initializer, "gpu": str(gpu_index), "gpu_hours": args.max_gpu_hours,
                "steps": 7000, "max_edge": 1024, "seed": 42, "noise_seed": 1729,
                "checkpoint_every": 100, "diagnostic_steps": [1000, 3000, 7000],
                "max_gaussians": 2000000, "timeout_s": args.timeout_s,
                "rotation_degrees": 0.0, "translation_fraction": 0.0, "pose_opt": False,
                "cache_images": False,
                "hypothesis": "Official BA improves 24-view reliability and held-out quality on untouched final scenes.",
                "independent_contribution": "Frozen paired evaluation and reliability diagnosis; BA is established upstream work.",
                "final_protocol": str(protocol_path), "final_protocol_sha256": file_hash(protocol_path),
            }
            if initializer == "colmap":
                spec.update(colmap_binary=str(root / "envs" / "gsplat" / "bin" / "colmap"),
                            colmap_use_gpu=False)
            else:
                spec.update(vggt_root=str(root / "third_party" / "vggt"),
                            vggt_python=str(root / "envs" / "vggt" / "bin" / "python3.10"),
                            vggt_ba=True, vggt_confidence=5.0, vggt_max_points=100000,
                            vggt_point_sampling="upstream_seeded_random_limit",
                            dinov2_root=str(dinov2_root))
            config_path = config_dir / f"{run_id}.json"
            atomic_json(config_path, spec)
            run = campaign / "runs" / run_id
            outputs = [run / "summary.json", run / "input" / "init_receipt.json", run / "training" / "model.ply"]
            outputs += [run / "training" / "milestones" / f"step_{step:08d}" / "metrics.json"
                        for step in (1000, 3000, 7000)]
            task = {
                "task_id": run_id, "kind": "bgs_run",
                "argv": [str(root / "envs" / "gsplat" / "bin" / "python"), "-m", "bgs", "run",
                         "--config", str(config_path)],
                "cwd": str(root),
                "env": {"PYTHONPATH": str(root), "TORCH_HOME": str(root / "cache" / "torch"),
                        "TMPDIR": str(root / "tmp" / matrix_id / run_id)},
                "dependencies": [], "config_path": str(config_path), "config_sha256": file_hash(config_path),
                "gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "gpu_count": 1,
                "max_gpu_hours": args.max_gpu_hours, "timeout_s": args.timeout_s,
                "disk_reservation_bytes": 10 * 2 ** 30, "expected_outputs": [str(path) for path in outputs],
                "priority": 200,
            }
            task_path = task_dir / f"{run_id}.json"
            atomic_json(task_path, task)
            created.append({"run_id": run_id, "config": str(config_path), "task": str(task_path)})
        receipt = {
            "schema": 1, "status": "MATERIALIZED_FROM_FROZEN_PROTOCOL",
            "protocol": str(protocol_path), "protocol_sha256": file_hash(protocol_path),
            "initialization_reserve_s": args.initialization_reserve_s, "tasks": created,
        }
        atomic_json(task_dir / "matrix_receipt.json", receipt)
        print(task_dir / "matrix_receipt.json")

    return main()

def build_formal_matrix():
    import argparse

    from pathlib import Path

    from bgs.core import WorkflowError, atomic_json, file_hash, read_json

    def main():
        p=argparse.ArgumentParser()
        p.add_argument("--project-root",required=True)
        p.add_argument("--registry",required=True)
        p.add_argument("--gpu",action="append",required=True,help="INDEX:GPU-UUID; repeat up to four times")
        p.add_argument("--steps",type=int,default=7000)
        p.add_argument("--revision",type=int,default=1)
        p.add_argument("--max-gpu-hours",type=float,default=2.0)
        p.add_argument("--timeout-s",type=int,default=5400)
        p.add_argument("--initialization-reserve-s",type=int,default=900)
        args=p.parse_args()
        root=Path(args.project_root).resolve();registry_path=Path(args.registry).resolve()
        registry=read_json(registry_path)
        if registry.get("schema")!=1 or registry.get("status")!="FROZEN_DEVELOPMENT":
            raise WorkflowError("Registry must be schema 1 and FROZEN_DEVELOPMENT")
        scenes=registry.get("development",[])
        if len(scenes)<3:raise WorkflowError("Need at least three frozen development scenes")
        gpus=[]
        for item in args.gpu:
            index,sep,uuid=item.partition(":")
            if not sep or not index.isdigit() or not uuid.startswith("GPU-"):
                raise WorkflowError(f"Invalid --gpu mapping: {item}")
            gpus.append((int(index),uuid))
        if not 1<=len(gpus)<=4 or len({u for _,u in gpus})!=len(gpus):
            raise WorkflowError("Provide one to four unique GPU UUID mappings")
        if args.steps!=7000:
            raise WorkflowError("Formal v1 matrix fixes a predeclared 7000-step total schedule")
        if args.revision<1:
            raise WorkflowError("revision must be a positive integer")
        if args.initialization_reserve_s<=0:
            raise WorkflowError("initialization reserve must be positive")
        # Initialization is charged before the training stage asks to reserve its
        # timeout plus a five-second cleanup window.  Preserve an explicit margin
        # so a successful initializer does not make training impossible to launch.
        required_hours=(args.initialization_reserve_s+args.timeout_s+5)/3600
        if required_hours>args.max_gpu_hours:
            raise WorkflowError("initialization plus training cleanup window exceeds task GPU-hour reservation")

        matrix_tag=f"mipnerf360_dev_v{args.revision}"
        configs=root/"configs"/matrix_tag
        tasks=root/"remote_tasks"/matrix_tag
        configs.mkdir(parents=True,exist_ok=True);tasks.mkdir(parents=True,exist_ok=True)
        created=[];assignment=0
        for scene in scenes:
            scene_id=scene["scene_id"]
            manifests=scene.get("manifests",{})
            for views in (6,12,24):
                manifest=Path(manifests.get(str(views),"")).resolve()
                if not manifest.is_file():raise WorkflowError(f"Missing {scene_id} {views}-view manifest")
                record=read_json(manifest)
                if record.get("scene_id")!=scene_id or record.get("partition")!="development" or len(record.get("train",[]))!=views:
                    raise WorkflowError(f"Manifest contract mismatch: {manifest}")
                for initializer in ("colmap","vggt"):
                    run_id=f"{scene_id}_v{views:02d}_{initializer}_s42_7k_v{args.revision}"
                    config_path=configs/f"{run_id}.json";task_path=tasks/f"{run_id}.json"
                    if config_path.exists() or task_path.exists():
                        raise WorkflowError(f"Refusing to overwrite immutable matrix entry: {run_id}")
                    gpu_index,gpu_uuid=gpus[assignment%len(gpus)];assignment+=1
                    campaign=root/"work"/matrix_tag/"campaigns"/run_id
                    spec={"run_id":run_id,"root":str(campaign),"manifest":str(manifest),
                          "gsplat_root":str(root/"third_party"/"gsplat"),
                          "python":str(root/"envs"/"gsplat"/"bin"/"python"),
                          "initializer":initializer,"gpu":str(gpu_index),"gpu_hours":args.max_gpu_hours,
                          "steps":7000,"max_edge":1024,"seed":42,"noise_seed":1729,
                          "checkpoint_every":100,"diagnostic_steps":[1000,3000,7000],
                          "max_gaussians":2000000,"timeout_s":args.timeout_s,
                          "rotation_degrees":0.0,"translation_fraction":0.0,"pose_opt":False,
                          "cache_images":False,
                          "hypothesis":"Paired sparse-view baseline at fixed inputs, hardware, resolution, schedule and seed.",
                          "independent_contribution":"Reproducible integration and held-out evaluation; upstream initialization and gsplat are not new algorithms."}
                    if initializer=="colmap":
                        spec.update(colmap_binary=str(root/"envs"/"gsplat"/"bin"/"colmap"),colmap_use_gpu=False)
                    else:
                        spec.update(vggt_root=str(root/"third_party"/"vggt"),
                                    vggt_python=str(root/"envs"/"vggt"/"bin"/"python3.10"),
                                    vggt_ba=False,vggt_confidence=5.0,vggt_max_points=100000,
                                    vggt_point_sampling="upstream_seeded_random_limit")
                    atomic_json(config_path,spec)
                    run=campaign/"runs"/run_id
                    expected=[run/"summary.json",run/"input"/"init_receipt.json",run/"training"/"model.ply"]
                    expected += [run/"training"/"milestones"/f"step_{step:08d}"/"metrics.json" for step in (1000,3000,7000)]
                    task={"task_id":run_id,"kind":"bgs_run",
                          "argv":[str(root/"envs"/"gsplat"/"bin"/"python"),"-m","bgs","run","--config",str(config_path)],
                          "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                          "TMPDIR":str(root/"tmp"/matrix_tag/run_id)},"dependencies":[],
                          "config_path":str(config_path),"config_sha256":file_hash(config_path),
                          "gpu_index":gpu_index,"gpu_uuid":gpu_uuid,"gpu_count":1,
                          "max_gpu_hours":args.max_gpu_hours,"timeout_s":args.timeout_s,
                          "disk_reservation_bytes":10*2**30,"expected_outputs":[str(x) for x in expected],
                          "priority":20+views+(0 if initializer=="colmap" else 1)}
                    atomic_json(task_path,task);created.append({"run_id":run_id,"config":str(config_path),"task":str(task_path)})
        receipt={"schema":1,"registry":str(registry_path),"registry_sha256":file_hash(registry_path),
                 "revision":args.revision,"steps":7000,"seed":42,"max_edge":1024,
                 "initialization_reserve_s":args.initialization_reserve_s,"tasks":created}
        receipt_path=tasks/"matrix_receipt.json"
        if receipt_path.exists():raise WorkflowError("Matrix receipt already exists")
        atomic_json(receipt_path,receipt)
        print(receipt_path)

    return main()

def build_pose_sensitivity():
    import argparse

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError,atomic_json,file_hash,read_json

    from bgs.splat_numerics import FLOOR_POLICY

    CONDITIONS=(
        ("r05",.5,0.),
        ("r10",1.,0.),
        ("t005",0.,.005),
        ("t010",0.,.01),
    )

    def main():
        p=argparse.ArgumentParser()
        p.add_argument("--root",required=True)
        p.add_argument("--revision",default="v1")
        p.add_argument("--gpu-uuids",nargs=4,required=True,metavar=("GPU0","GPU1","GPU2","GPU3"))
        p.add_argument("--round-start-charged",type=float,required=True)
        p.add_argument("--round-cap-gpu-hours",type=float,default=5.)
        p.add_argument("--max-gpu-hours",type=float,default=.6)
        p.add_argument("--timeout-s",type=int,default=2100)
        args=p.parse_args()
        if any(not x.startswith("GPU-") for x in args.gpu_uuids):raise WorkflowError("Explicit GPU UUIDs required")
        if args.timeout_s<=0 or args.timeout_s/3600>args.max_gpu_hours:raise WorkflowError("Timeout must fit per-task budget")
        if len(CONDITIONS)*2*args.max_gpu_hours>args.round_cap_gpu_hours+1e-12:
            raise WorkflowError("Worst-case task reservations exceed round cap")
        root=Path(args.root).resolve();protocol=root/"protocols"/"pose_sensitivity_v1.json"
        if not protocol.is_file():raise WorkflowError("Frozen pose_sensitivity_v1 protocol missing")
        queue_path=root/"work"/"formal_stage_v1"/"queue.json";queue=read_json(queue_path)
        if abs(queue["charged_gpu_hours"]-args.round_start_charged)>1e-9 or queue["reserved_gpu_hours"]!=0:
            raise WorkflowError("Queue charge/reservation changed since round budget audit")
        if any(x["status"] in ("PENDING","RESERVED","RUNNING") for x in queue["tasks"].values()):
            raise WorkflowError("Queue is not idle")
        if queue["max_parallel"]>4:raise WorkflowError("Queue parallelism exceeds protocol limit")
        worst_reservation=len(CONDITIONS)*2*args.max_gpu_hours
        if queue["charged_gpu_hours"]+worst_reservation>min(queue["budget_gpu_hours"],args.round_start_charged+args.round_cap_gpu_hours)+1e-12:
            raise WorkflowError("Frozen matrix does not fit both the total ledger and round cap")

        stage=root/"work"/f"pose_sensitivity_{args.revision}"
        config_dir=root/"configs"/f"pose_sensitivity_{args.revision}"
        task_dir=root/"remote_tasks"/f"pose_sensitivity_{args.revision}"
        if stage.exists() or config_dir.exists() or task_dir.exists():raise WorkflowError("Refusing to overwrite pose-sensitivity identity")
        config_dir.mkdir(parents=True);task_dir.mkdir(parents=True);rows=[];zero_references={}
        for scene_index,scene in enumerate(("bonsai","room")):
            zero_id=f"m360_{scene}_dev_v24_vggt_ba_post_s42_7k_floor_v1"
            zero_config_path=root/"configs"/"ba_operation_ablation_floor_v1"/f"{zero_id}.json"
            base=read_json(zero_config_path);zero_run=Path(base["root"])/"runs"/zero_id
            required={"steps":7000,"max_edge":1024,"seed":42,"noise_seed":1729,"rotation_degrees":0.,
                      "translation_fraction":0.,"pose_opt":False,"splat_numerics_policy":FLOOR_POLICY,
                      "vggt_ba_ablation_arm":"post_ba"}
            mismatch={k:(base.get(k),v) for k,v in required.items() if base.get(k)!=v}
            if mismatch:raise WorkflowError(f"Zero-reference contract mismatch for {scene}: {mismatch}")
            zero_outputs={
                "experiment":zero_run/"experiment.json","init_receipt":zero_run/"input"/"init_receipt.json",
                "alignment":zero_run/"training"/"alignment.json","metrics":zero_run/"training"/"metrics.json",
                "model":zero_run/"training"/"model.ply","complete":zero_run/"training"/"complete.json",
            }
            if any(not p.is_file() for p in zero_outputs.values()):raise WorkflowError(f"Incomplete zero reference for {scene}")
            experiment=read_json(zero_outputs["experiment"]);init=read_json(zero_outputs["init_receipt"])
            zero_references[scene]={"run_id":zero_id,"config":str(zero_config_path),
                "config_sha256":file_hash(zero_config_path),"outputs":{k:str(v) for k,v in zero_outputs.items()},
                "output_sha256":{k:file_hash(v) for k,v in zero_outputs.items()},
                "fingerprint_hash":experiment["fingerprint_hash"],"source_hashes":experiment["source"],
                "initialization_files":init["files"],"pair_root":base["vggt_ba_pair_root"]}
            for condition_index,(condition,rotation,translation) in enumerate(CONDITIONS):
                gpu=condition_index;run_id=f"m360_{scene}_dev_v24_vggt_ba_post_s42_7k_pose_{condition}_{args.revision}"
                config={**base,"run_id":run_id,"root":str(stage/"campaigns"/run_id),"gpu":str(gpu),
                    "gpu_hours":args.max_gpu_hours,"timeout_s":args.timeout_s,
                    "rotation_degrees":rotation,"translation_fraction":translation,
                    "vggt_ba_pair_generate":False,"pose_sensitivity_protocol":"pose_sensitivity_v1",
                    "pose_sensitivity_condition":condition,"zero_reference_run_id":zero_id,
                    "zero_reference_config_sha256":file_hash(zero_config_path),
                    "hypothesis":"Added training-pose error changes genuine held-out rendering quality at fixed initialization and schedule.",
                    "independent_contribution":"Controlled sensitivity integration and measurement; not a new pose or BA algorithm.",
                    "scope":"Development-only estimated-pose perturbation; no geometry ground truth or broad robustness claim."}
                config.pop("control_of",None);config.pop("control_of_sha256",None)
                config_path=config_dir/f"{run_id}.json";atomic_json(config_path,config)
                run=Path(config["root"])/"runs"/run_id
                expected=[run/"experiment.json",run/"summary.json",run/"report.html",run/"input"/"init_receipt.json",
                    run/"training"/"alignment.json",run/"training"/"initialization_numerics.json",
                    run/"training"/"metrics.json",run/"training"/"model.ply",run/"training"/"complete.json"]
                task={"task_id":run_id,"kind":"bgs_run",
                    "argv":[config["python"],"-m","bgs","run","--config",str(config_path)],
                    "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                        "TMPDIR":str(root/"tmp"/f"pose_sensitivity_{args.revision}"/run_id)},
                    "dependencies":[],"config_path":str(config_path),"config_sha256":file_hash(config_path),
                    "gpu_index":gpu,"gpu_uuid":args.gpu_uuids[gpu],"gpu_count":1,
                    "max_gpu_hours":args.max_gpu_hours,"timeout_s":args.timeout_s,
                    "disk_reservation_bytes":4*2**30,"expected_outputs":[str(x) for x in expected],
                    "priority":30+scene_index}
                task_path=task_dir/f"{run_id}.task.json";atomic_json(task_path,task)
                rows.append({"scene":scene,"condition":condition,"rotation_degrees":rotation,
                    "translation_fraction":translation,"config":str(config_path),"config_sha256":file_hash(config_path),
                    "task":str(task_path),"task_sha256":file_hash(task_path),"gpu_index":gpu})
        atomic_json(task_dir/"matrix_receipt.json",{"schema":1,"revision":args.revision,
            "protocol":str(protocol),"protocol_sha256":file_hash(protocol),"created_before_scores":True,
            "queue":str(queue_path),"round_start_charged_gpu_hours":args.round_start_charged,
            "total_ledger_cap_gpu_hours":queue["budget_gpu_hours"],"total_ledger_remaining_at_freeze":queue["budget_gpu_hours"]-args.round_start_charged,
            "round_cap_gpu_hours":args.round_cap_gpu_hours,"per_task_max_gpu_hours":args.max_gpu_hours,
            "per_task_timeout_s":args.timeout_s,"worst_case_reservation_gpu_hours":worst_reservation,
            "max_parallel":4,"zero_references":zero_references,"new_runs":rows,
            "limits":["development scenes only","no final_v1 access","one training seed and one noise-direction set",
                      "controlled errors added to estimated poses, not known absolute pose error","no geometry ground truth"]})
        print(task_dir/"matrix_receipt.json")

    return main()

def build_render_benchmark_task():
    import argparse

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError, atomic_json, file_hash, read_json

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--project-root", required=True)
        parser.add_argument("--source-root", help="Clean source tree to execute; defaults to project root")
        parser.add_argument("--config", required=True)
        parser.add_argument("--gpu", required=True, help="INDEX:GPU-UUID")
        parser.add_argument("--out-dir", required=True)
        parser.add_argument("--revision", type=int, default=1)
        parser.add_argument("--max-gpu-hours", type=float, default=0.25)
        parser.add_argument("--timeout-s", type=int, default=900)
        args = parser.parse_args()

        root = Path(args.project_root).resolve()
        source_root = Path(args.source_root).resolve() if args.source_root else root
        benchmark_script=source_root / "scripts" / "render_benchmark.py"
        if not benchmark_script.is_file():
            raise WorkflowError("Source root does not contain scripts/render_benchmark.py")
        config_path = Path(args.config).resolve()
        config = read_json(config_path)
        index, separator, gpu_uuid = args.gpu.partition(":")
        if not separator or not index.isdigit() or not gpu_uuid.startswith("GPU-"):
            raise WorkflowError("--gpu must be INDEX:GPU-UUID")
        if args.revision < 1 or args.timeout_s / 3600 > args.max_gpu_hours:
            raise WorkflowError("Invalid benchmark revision or budget")

        run_id = config["run_id"]
        task_id = f"bench_{run_id}_mc_v{args.revision}"
        out_dir = Path(args.out_dir).resolve()
        task_path = out_dir / f"{task_id}.json"
        result_path = root / "work" / "performance_mipnerf360_dev_v1" / f"{task_id}.json"
        if task_path.exists() or result_path.exists():
            raise WorkflowError(f"Refusing to overwrite benchmark identity: {task_id}")
        task = {
            "task_id": task_id,
            "kind": "render_benchmark",
            "argv": [
                str(root / "envs" / "gsplat" / "bin" / "python"),
                str(benchmark_script),
                "--config", str(config_path),
                "--out", str(result_path),
                "--warmup", "5", "--repeats", "30", "--max-cameras", "64",
            ],
            "cwd": str(source_root),
            "env": {
                "PYTHONPATH": str(source_root),
                "TORCH_HOME": str(root / "cache" / "torch"),
                "TMPDIR": str(root / "tmp" / "performance_mipnerf360_dev_v1" / task_id),
            },
            "dependencies": [run_id],
            "config_path": str(config_path),
            "config_sha256": file_hash(config_path),
            "gpu_index": int(index),
            "gpu_uuid": gpu_uuid,
            "gpu_count": 1,
            "max_gpu_hours": args.max_gpu_hours,
            "timeout_s": args.timeout_s,
            "disk_reservation_bytes": 2 ** 30,
            "expected_outputs": [str(result_path)],
            "priority": 100,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(task_path, task)
        print(task_path)

    return main()

def build_resume_diagnostic_task():
    import argparse

    from pathlib import Path

    import sys

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
            'argv':[str(root/'envs'/'gsplat'/'bin'/'python'),str(root/'scripts'/"diagnostics" / 'qualify_resume.py'),
                    '--config',str(config),'--steps','1400',mode_arg],
            'cwd':str(root),'env':{'PYTHONPATH':str(root),'TMPDIR':str(root/'tmp'/'resume_diagnostics_v1'/task_id)},
            'dependencies':[],'config_path':str(config),'config_sha256':file_hash(config),
            'gpu_index':args.gpu,'gpu_uuid':args.gpu_uuid,'gpu_count':1,
            'max_gpu_hours':0.25,'timeout_s':900,'disk_reservation_bytes':2*2**30,
            'expected_outputs':[str(output)],'priority':5}
        atomic_json(task_path,task);print(task_path)

    return main()

def build_splat_initialization_diagnostics():
    import argparse

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError,atomic_json,file_hash

    def main():
        p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--revision",default="v1")
        p.add_argument("--gpu-uuids",nargs=4,required=True,metavar=("GPU0","GPU1","GPU2","GPU3"));args=p.parse_args()
        if any(not x.startswith("GPU-") for x in args.gpu_uuids):raise WorkflowError("Explicit GPU UUIDs required")
        root=Path(args.root).resolve();task_dir=root/"remote_tasks"/f"splat_initialization_diagnostic_{args.revision}"
        out_dir=root/"work"/f"splat_initialization_diagnostic_{args.revision}"
        if task_dir.exists() or out_dir.exists():raise WorkflowError("Refusing to overwrite initialization diagnostic")
        task_dir.mkdir(parents=True);rows=[]
        assignments=[("bonsai","pre",0),("bonsai","post",1),("room","pre",2),("room","post",3)]
        for scene,arm,gpu in assignments:
            run_id=f"m360_{scene}_dev_v24_vggt_ba_{arm}_s42_7k_v1"
            config=root/"configs"/"ba_operation_ablation_v1"/f"{run_id}.json"
            task_id=f"diag_init_{scene}_{arm}_s42_{args.revision}"
            output=out_dir/f"{task_id}.json"
            task={"task_id":task_id,"kind":"initialization_diagnostic",
                "argv":[str(root/"envs"/"gsplat"/"bin"/"python"),str(root/"scripts"/"diagnostics" / "diagnose_splat_initialization.py"),
                        "--config",str(config),"--out",str(output)],
                "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                        "TMPDIR":str(root/"tmp"/f"splat_initialization_diagnostic_{args.revision}"/task_id)},
                "dependencies":[],"config_path":str(config),"config_sha256":file_hash(config),
                "gpu_index":gpu,"gpu_uuid":args.gpu_uuids[gpu],"gpu_count":1,"max_gpu_hours":.05,
                "timeout_s":180,"disk_reservation_bytes":2**30,"expected_outputs":[str(output)],"priority":3}
            path=task_dir/f"{task_id}.task.json";atomic_json(path,task);rows.append(str(path))
        atomic_json(task_dir/"matrix_receipt.json",{"schema":1,"revision":args.revision,
            "scope":"read-only exact point-to-splat initialization numerics","tasks":rows})
        print("\n".join(rows))

    return main()

def build_splat_stability_candidates():
    import argparse

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError,atomic_json,file_hash,read_json

    from bgs.splat_numerics import DEDUP_POLICY,FLOOR_POLICY

    def main():
        p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--revision",default="v1")
        p.add_argument("--gpu-uuids",nargs=4,required=True);args=p.parse_args()
        if any(not x.startswith("GPU-") for x in args.gpu_uuids):raise WorkflowError("Explicit GPU UUIDs required")
        root=Path(args.root).resolve();config_dir=root/"configs"/f"splat_stability_candidates_{args.revision}"
        task_dir=root/"remote_tasks"/f"splat_stability_candidates_{args.revision}"
        out_dir=root/"work"/f"splat_stability_candidates_{args.revision}"
        if config_dir.exists() or task_dir.exists() or out_dir.exists():raise WorkflowError("Refusing to overwrite candidate probes")
        config_dir.mkdir(parents=True);task_dir.mkdir(parents=True);rows=[]
        assignments=[("bonsai",DEDUP_POLICY,0),("bonsai",FLOOR_POLICY,1),
                     ("room",DEDUP_POLICY,2),("room",FLOOR_POLICY,3)]
        for scene,policy,gpu in assignments:
            source_id=f"m360_{scene}_dev_v24_vggt_ba_pre_s42_7k_v1"
            source_path=root/"configs"/"ba_operation_ablation_v1"/f"{source_id}.json";source=read_json(source_path)
            label="dedup" if policy==DEDUP_POLICY else "floor"
            run_id=f"m360_{scene}_pre_s42_{label}_stability801_{args.revision}"
            old_run=Path(source["root"])/"runs"/source_id
            config={**source,"run_id":run_id,"root":str(out_dir/"campaigns"/run_id),"steps":801,
                "checkpoint_every":801,"diagnostic_steps":[],"splat_numerics_policy":policy,
                "diagnostic_training_input":str(old_run/"input"),"expected_initial_points":22192 if scene=="bonsai" else 36226,
                "source_config":str(source_path),"source_config_sha256":file_hash(source_path),
                "hypothesis":"A causal initialization numerics policy removes the pre-training non-finite scale and remains finite through an actual structural update.",
                "scope":"development-only short stability probe; no quality claim"}
            config_path=config_dir/f"{run_id}.json";atomic_json(config_path,config)
            output=out_dir/f"{run_id}.json";task_id=f"probe_{run_id}"
            task={"task_id":task_id,"kind":"initialization_diagnostic",
                "argv":[str(root/"envs"/"gsplat"/"bin"/"python"),str(root/"scripts"/"diagnostics" / "run_splat_stability_probe.py"),
                        "--config",str(config_path),"--out",str(output)],
                "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                    "TMPDIR":str(root/"tmp"/f"splat_stability_candidates_{args.revision}"/task_id)},
                "dependencies":[],"config_path":str(config_path),"config_sha256":file_hash(config_path),
                "gpu_index":gpu,"gpu_uuid":args.gpu_uuids[gpu],"gpu_count":1,"max_gpu_hours":.2,
                "timeout_s":720,"disk_reservation_bytes":3*2**30,"expected_outputs":[str(output)],"priority":4}
            task_path=task_dir/f"{task_id}.task.json";atomic_json(task_path,task)
            rows.append({"scene":scene,"policy":policy,"config":str(config_path),"config_sha256":file_hash(config_path),"task":str(task_path)})
        atomic_json(task_dir/"matrix_receipt.json",{"schema":1,"revision":args.revision,"seed":42,
            "steps":801,"arms":"pre_ba only","candidates":[DEDUP_POLICY,FLOOR_POLICY],
            "selection_rule":"prefer a finite policy that crosses a real structural event with fewer changes to points, colors and coverage",
            "rows":rows})
        print(task_dir/"matrix_receipt.json")

    return main()

def build_splat_stability_post_controls():
    import argparse

    from pathlib import Path

    import sys

    from bgs.core import WorkflowError,atomic_json,file_hash,read_json

    from bgs.splat_numerics import FLOOR_POLICY

    def main():
        p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--revision",default="v1")
        p.add_argument("--gpu-uuids",nargs=2,required=True);args=p.parse_args()
        if any(not x.startswith("GPU-") for x in args.gpu_uuids):raise WorkflowError("Explicit GPU UUIDs required")
        root=Path(args.root).resolve();config_dir=root/"configs"/f"splat_stability_post_{args.revision}"
        task_dir=root/"remote_tasks"/f"splat_stability_post_{args.revision}"
        out_dir=root/"work"/f"splat_stability_post_{args.revision}"
        if config_dir.exists() or task_dir.exists() or out_dir.exists():raise WorkflowError("Refusing to overwrite post controls")
        config_dir.mkdir(parents=True);task_dir.mkdir(parents=True);rows=[]
        for gpu,scene in enumerate(("bonsai","room")):
            source_id=f"m360_{scene}_dev_v24_vggt_ba_post_s42_7k_v1"
            source_path=root/"configs"/"ba_operation_ablation_v1"/f"{source_id}.json";source=read_json(source_path)
            run_id=f"m360_{scene}_post_s42_floor_stability801_{args.revision}"
            old_run=Path(source["root"])/"runs"/source_id
            config={**source,"run_id":run_id,"root":str(out_dir/"campaigns"/run_id),"steps":801,
                "checkpoint_every":801,"diagnostic_steps":[],"splat_numerics_policy":FLOOR_POLICY,
                "diagnostic_training_input":str(old_run/"input"),"expected_initial_points":22192 if scene=="bonsai" else 36226,
                "source_config":str(source_path),"source_config_sha256":file_hash(source_path),
                "hypothesis":"The selected floor policy is a no-op on the finite post-BA snapshot and remains finite through an actual structural update.",
                "scope":"development-only short matched-policy control; no quality claim"}
            config_path=config_dir/f"{run_id}.json";atomic_json(config_path,config)
            output=out_dir/f"{run_id}.json";task_id=f"probe_{run_id}"
            task={"task_id":task_id,"kind":"initialization_diagnostic",
                "argv":[str(root/"envs"/"gsplat"/"bin"/"python"),str(root/"scripts"/"diagnostics" / "run_splat_stability_probe.py"),
                        "--config",str(config_path),"--out",str(output)],
                "cwd":str(root),"env":{"PYTHONPATH":str(root),"TORCH_HOME":str(root/"cache"/"torch"),
                    "TMPDIR":str(root/"tmp"/f"splat_stability_post_{args.revision}"/task_id)},
                "dependencies":[],"config_path":str(config_path),"config_sha256":file_hash(config_path),
                "gpu_index":gpu,"gpu_uuid":args.gpu_uuids[gpu],"gpu_count":1,"max_gpu_hours":.2,
                "timeout_s":720,"disk_reservation_bytes":3*2**30,"expected_outputs":[str(output)],"priority":4}
            task_path=task_dir/f"{task_id}.task.json";atomic_json(task_path,task)
            rows.append({"scene":scene,"policy":FLOOR_POLICY,"config":str(config_path),
                         "config_sha256":file_hash(config_path),"task":str(task_path)})
        atomic_json(task_dir/"matrix_receipt.json",{"schema":1,"revision":args.revision,"seed":42,
            "steps":801,"arms":"post_ba","selected_policy":FLOOR_POLICY,
            "selection_evidence":"splat_stability_candidates_v1","rows":rows})
        print(task_dir/"matrix_receipt.json")

    return main()

def build_vggt_ba_controls():
    import argparse

    from pathlib import Path

    from bgs.core import WorkflowError, atomic_json, file_hash, read_json

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--project-root", required=True)
        parser.add_argument("--base-config", action="append", required=True)
        parser.add_argument("--gpu", action="append", required=True, help="INDEX:GPU-UUID")
        parser.add_argument("--dinov2-root", required=True)
        parser.add_argument("--revision", type=int, default=1)
        parser.add_argument("--max-gpu-hours", type=float, default=2.0)
        parser.add_argument("--timeout-s", type=int, default=5400)
        args = parser.parse_args()

        root = Path(args.project_root).resolve()
        if len(args.base_config) != 3 or len(args.gpu) != 3:
            raise WorkflowError("Official use_ba branch matrix requires exactly three base configs and GPUs")
        if args.revision < 1:
            raise WorkflowError("revision must be positive")
        if (args.timeout_s + 905) / 3600 > args.max_gpu_hours:
            raise WorkflowError("Budget must cover initialization allowance and training cleanup")
        mappings = []
        for value in args.gpu:
            index, separator, gpu_uuid = value.partition(":")
            if not separator or not index.isdigit() or not gpu_uuid.startswith("GPU-"):
                raise WorkflowError(f"Invalid GPU mapping: {value}")
            mappings.append((int(index), gpu_uuid))

        dinov2_root = Path(args.dinov2_root).resolve()
        tag = f"mipnerf360_ba_v{args.revision}"
        config_dir = root / "configs" / tag
        task_dir = root / "remote_tasks" / tag
        if config_dir.exists() or task_dir.exists():
            raise WorkflowError("Refusing to overwrite immutable use_ba branch matrix")
        config_dir.mkdir(parents=True)
        task_dir.mkdir(parents=True)
        created = []
        for base_value, (gpu_index, gpu_uuid) in zip(args.base_config, mappings):
            base_path = Path(base_value).resolve()
            base = read_json(base_path)
            manifest = read_json(base["manifest"])
            if (
                base.get("initializer") != "vggt"
                or base.get("vggt_ba") is not False
                or base.get("steps") != 7000
                or base.get("seed") != 42
                or base.get("max_edge") != 1024
                or len(manifest.get("train", [])) != 24
            ):
                raise WorkflowError(f"Base is not a fixed 24-view no-BA formal config: {base_path}")
            source_suffix = "_vggt_s42_7k_v3"
            if not base["run_id"].endswith(source_suffix):
                raise WorkflowError(f"Unexpected base identity: {base['run_id']}")
            run_id = base["run_id"][:-len(source_suffix)] + f"_vggt_ba_s42_7k_v{args.revision}"
            campaign = root / "work" / tag / "campaigns" / run_id
            config = dict(base)
            config.update(
                run_id=run_id,
                root=str(campaign),
                gpu=str(gpu_index),
                gpu_hours=args.max_gpu_hours,
                timeout_s=args.timeout_s,
                vggt_ba=True,
                dinov2_root=str(dinov2_root),
                hypothesis="The complete official use_ba initialization branch improves reliability at fixed inputs and training schedule.",
                independent_contribution="Controlled diagnosis and offline qualification; bundle adjustment is established upstream work.",
                control_of=base["run_id"],
                control_of_sha256=file_hash(base_path),
            )
            config_path = config_dir / f"{run_id}.json"
            atomic_json(config_path, config)
            run = campaign / "runs" / run_id
            expected = [run / "summary.json", run / "input" / "init_receipt.json", run / "training" / "model.ply"]
            expected += [run / "training" / "milestones" / f"step_{step:08d}" / "metrics.json" for step in (1000, 3000, 7000)]
            task = {
                "task_id": run_id,
                "kind": "bgs_run",
                "argv": [str(root / "envs" / "gsplat" / "bin" / "python"), "-m", "bgs", "run", "--config", str(config_path)],
                "cwd": str(root),
                "env": {
                    "PYTHONPATH": str(root),
                    "TORCH_HOME": str(root / "cache" / "torch"),
                    "TMPDIR": str(root / "tmp" / tag / run_id),
                },
                "dependencies": [],
                "config_path": str(config_path),
                "config_sha256": file_hash(config_path),
                "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuid,
                "gpu_count": 1,
                "max_gpu_hours": args.max_gpu_hours,
                "timeout_s": args.timeout_s,
                "disk_reservation_bytes": 10 * 2 ** 30,
                "expected_outputs": [str(path) for path in expected],
                "priority": 50,
            }
            task_path = task_dir / f"{run_id}.json"
            atomic_json(task_path, task)
            created.append({"run_id": run_id, "base_config": str(base_path), "base_sha256": file_hash(base_path),
                            "config": str(config_path), "task": str(task_path)})
        atomic_json(task_dir / "matrix_receipt.json", {
            "schema": 1,
            "revision": args.revision,
            "factor": "complete official feed-forward branch -> complete official use_ba branch",
            "fixed": ["manifest", "24 views", "7000 steps", "max_edge=1024", "seed=42", "renderer", "GPU family"],
            "inactive_in_ba_branch": ["conf_thres_value", "100000-point random truncation"],
            "scope": "official upstream branch-level reliability comparison; not an isolated BA operation or a new algorithm",
            "created": created,
        })
        print(task_dir / "matrix_receipt.json")

    return main()

def freeze_development_registry():
    import argparse

    from datetime import datetime, timezone

    from pathlib import Path

    from bgs.core import WorkflowError, atomic_json, file_hash, read_json

    from bgs.data import validate_manifest, verify_inputs

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        parser.add_argument("--source-url", required=True)
        parser.add_argument("--archive-bytes", required=True, type=int)
        parser.add_argument("--archive-md5", required=True)
        parser.add_argument("--archive-sha256", required=True)
        parser.add_argument("--license-note", required=True)
        parser.add_argument(
            "--scene", action="append", required=True,
            help="scene_id:category:reference:manifest6:manifest12:manifest24",
        )
        args = parser.parse_args()

        output = Path(args.out).resolve()
        if output.exists():
            raise WorkflowError(f"Refusing to overwrite frozen registry: {output}")
        scenes = []
        seen = set()
        for value in args.scene:
            fields = value.split(":", 5)
            if len(fields) != 6:
                raise WorkflowError(f"Invalid --scene record: {value}")
            scene_id, category, reference, manifest6, manifest12, manifest24 = fields
            if scene_id in seen:
                raise WorkflowError(f"Duplicate scene id: {scene_id}")
            seen.add(scene_id)
            reference_path = Path(reference).resolve()
            manifests = {
                "6": Path(manifest6).resolve(),
                "12": Path(manifest12).resolve(),
                "24": Path(manifest24).resolve(),
            }
            records = {}
            for count, path in manifests.items():
                record = validate_manifest(read_json(path))
                verify_inputs(record)
                if (
                    record["scene_id"] != scene_id
                    or record["partition"] != "development"
                    or len(record["train"]) != int(count)
                    or Path(record["reference"]).resolve() != reference_path
                ):
                    raise WorkflowError(f"Manifest contract mismatch: {path}")
                records[count] = record
            if not (
                set(records["6"]["train"]) < set(records["12"]["train"])
                < set(records["24"]["train"])
            ):
                raise WorkflowError(f"Train splits are not strictly nested: {scene_id}")
            if not (records["6"]["holdout"] == records["12"]["holdout"] == records["24"]["holdout"]):
                raise WorkflowError(f"Holdout sets differ across view counts: {scene_id}")
            scenes.append({
                "scene_id": scene_id,
                "category": category,
                "reference": str(reference_path),
                "reference_sha256": file_hash(reference_path),
                "manifests": {count: str(path) for count, path in manifests.items()},
                "manifest_sha256": {count: file_hash(path) for count, path in manifests.items()},
                "holdout_count": len(records["6"]["holdout"]),
            })

        registry = {
            "schema": 1,
            "status": "FROZEN_DEVELOPMENT",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "url": args.source_url,
                "archive_bytes": args.archive_bytes,
                "archive_md5": args.archive_md5,
                "archive_sha256": args.archive_sha256,
                "license_note": args.license_note,
                "rgb_distribution": "private_only",
            },
            "selection": "predeclared category coverage; no held-out RGB or score used",
            "development": scenes,
        }
        atomic_json(output, registry)
        print(output)

    return main()

def freeze_final_protocol():
    import argparse

    from datetime import datetime, timezone

    from pathlib import Path

    from bgs.core import WorkflowError, atomic_json

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", required=True)
        parser.add_argument("--matrix-id", default="mipnerf360_final_v1")
        parser.add_argument("--scene", action="append", required=True)
        args = parser.parse_args()

        output = Path(args.out).resolve()
        scenes = list(args.scene)
        if output.exists():
            raise WorkflowError(f"Refusing to overwrite frozen final protocol: {output}")
        if len(scenes) < 3 or len(scenes) != len(set(scenes)):
            raise WorkflowError("Final protocol requires at least three unique scenes")

        run_ids = []
        for scene in scenes:
            if not scene or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in scene):
                raise WorkflowError(f"Unsafe scene id: {scene!r}")
            run_ids.extend([
                f"{scene}_final_v24_colmap_s42_7k_v1",
                f"{scene}_final_v24_vggt_ba_s42_7k_v1",
            ])
        atomic_json(output, {
            "schema": 1,
            "status": "FROZEN_BEFORE_SCORING",
            "matrix_id": args.matrix_id,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "scene_ids": scenes,
            "run_ids": run_ids,
            "partition": "final",
            "selection": "unseen scenes selected by category before any final RGB score",
            "fixed": {
                "train_views": 24,
                "holdout_rule": "every eighth ordered image",
                "steps": 7000,
                "diagnostic_steps": [1000, 3000, 7000],
                "max_edge": 1024,
                "seed": 42,
                "hardware_family": "NVIDIA A800-SXM4-80GB",
                "arms": ["COLMAP", "VGGT with official upstream bundle adjustment"],
            },
            "decision_rule": "report every predeclared run and failure; do not tune from final scores",
            "claim_scope": "external confirmation of a development-set diagnosis; BA is established upstream work",
        })
        print(output)

    return main()

COMMANDS = {
    'build-ba-ablation-benchmarks': build_ba_ablation_benchmarks,
    'build-ba-operation-ablation': build_ba_operation_ablation,
    'build-confidence-probe-tasks': build_confidence_probe_tasks,
    'build-final-matrix': build_final_matrix,
    'build-formal-matrix': build_formal_matrix,
    'build-pose-sensitivity': build_pose_sensitivity,
    'build-render-benchmark-task': build_render_benchmark_task,
    'build-resume-diagnostic-task': build_resume_diagnostic_task,
    'build-splat-initialization-diagnostics': build_splat_initialization_diagnostics,
    'build-splat-stability-candidates': build_splat_stability_candidates,
    'build-splat-stability-post-controls': build_splat_stability_post_controls,
    'build-vggt-ba-controls': build_vggt_ba_controls,
    'freeze-development-registry': freeze_development_registry,
    'freeze-final-protocol': freeze_final_protocol,
}

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not args or args[0] in ("-h","--help"):
        parser.print_help()
        return
    command=parser.parse_args(args[:1]).command
    saved=sys.argv
    try:
        sys.argv=[saved[0]+" "+command, *args[1:]]
        return COMMANDS[command]()
    finally:
        sys.argv=saved

if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"BLOCKED: {error}")
