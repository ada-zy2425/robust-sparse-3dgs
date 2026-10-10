import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from bgs.core import Engine,WorkflowError,atomic_json,read_json,validate_metrics,file_hash,lock
from bgs.data import split_names,validate_manifest,stage_training,prepare,verify_inputs
from bgs.queue import add_task,can_reserve,new_state,retry_task,set_parallel,task_snapshot,worker


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def job(self,engine,script,outputs=(),timeout=2,**kw):
        return engine.run('test','stage',[sys.executable,'-c',script],str(self.root),{},outputs,timeout,gpu_count=0,**kw)
    def test_success_is_idempotent_and_tamper_is_rejected(self):
        out=self.root/'artifact.txt'
        script="from pathlib import Path; p=Path('artifact.txt'); p.write_text('first' if not p.exists() else 'duplicate')"
        with Engine(self.root,0) as e:
            self.job(e,script,[out]);self.job(e,script,[out])
            self.assertEqual(out.read_text(),'first')
            out.write_text('tampered')
            with self.assertRaises(WorkflowError):self.job(e,script,[out])
    def test_failure_is_not_retried_implicitly(self):
        with Engine(self.root,0) as e:
            with self.assertRaises(WorkflowError):self.job(e,'raise SystemExit(7)')
            with self.assertRaises(WorkflowError):self.job(e,'raise SystemExit(7)')
            self.assertEqual(len(e.ledger['stages']['test/stage']['attempts']),1)
            with self.assertRaises(WorkflowError):self.job(e,'raise SystemExit(7)',retry_failed=True)
            self.assertEqual(len(e.ledger['stages']['test/stage']['attempts']),2)
            with self.assertRaises(WorkflowError):self.job(e,'raise SystemExit(7)',retry_failed=True)
    def test_budget_blocks_before_launch(self):
        marker=self.root/'started'
        with Engine(self.root,0) as e:
            with self.assertRaises(WorkflowError):
                e.run('x','gpu',[sys.executable,'-c',"open('started','w').write('x')"],str(self.root),{},[marker],1,gpu_count=1)
        self.assertFalse(marker.exists())
    def test_timeout_is_persisted(self):
        with Engine(self.root,0) as e:
            with self.assertRaises(WorkflowError):self.job(e,'import time;time.sleep(30)',timeout=.15)
        self.assertEqual(read_json(self.root/'ledger.json')['stages']['test/stage']['status'],'TIMED_OUT')
    def test_missing_output_fails(self):
        with Engine(self.root,0) as e:
            with self.assertRaises(WorkflowError):self.job(e,'pass',[self.root/'missing'])
    def test_changed_task_identity_rejected(self):
        with Engine(self.root,0) as e:
            self.job(e,'pass')
            with self.assertRaises(WorkflowError):self.job(e,"print('different')")
    def test_process_lock_prevents_second_runner(self):
        with Engine(self.root,0):
            code="from bgs.core import Engine; from pathlib import Path; e=Engine(Path(__import__('sys').argv[1]),0);e.__enter__()"
            env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
            child=subprocess.run([sys.executable,'-c',code,str(self.root)],env=env,capture_output=True)
            self.assertNotEqual(child.returncode,0)
    def test_nonfinite_metrics_and_denominator_rejected(self):
        p=self.root/'metrics.json'
        atomic_json(p,{'status':'VALID','images':[{'name':'a','psnr':20.,'ssim':.5,'lpips':.4}]})
        self.assertEqual(validate_metrics(p,['a'])['mean']['psnr'],20)
        with self.assertRaises(WorkflowError):validate_metrics(p,['a','b'])
        p.write_text('{"status":"VALID","images":[{"name":"a","psnr":NaN,"ssim":0.5,"lpips":0.4}]}')
        with self.assertRaises(WorkflowError):validate_metrics(p,['a'])
    def test_live_orphan_cannot_be_relaunched(self):
        from bgs.core import pid_token
        import socket
        with Engine(self.root,0) as e:
            self.job(e,'pass')
            state=e.ledger['stages']['test/stage'];state.update(status='RUNNING',host=socket.gethostname(),pid=os.getpid(),pid_token=pid_token(os.getpid()))
            e.save()
            with self.assertRaises(WorkflowError):self.job(e,'pass',retry_failed=True,recover_stale=True)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.config=self.root/'config.json';atomic_json(self.config,{'fixed':True})
    def tearDown(self):self.tmp.cleanup()
    def task(self,task_id='a',deps=(),hours=.5):
        return {'task_id':task_id,'kind':'bgs_run',
          'argv':[sys.executable,'-m','bgs','run','--config',str(self.config)],
          'cwd':str(self.root),'env':{},'dependencies':list(deps),
          'config_path':str(self.config),'config_sha256':file_hash(self.config),
          'gpu_index':0,'gpu_uuid':'GPU-test','gpu_count':1,'max_gpu_hours':hours,
          'timeout_s':min(60,hours*3600),'disk_reservation_bytes':1024,
          'expected_outputs':[str(self.root/f'{task_id}.out')]}
    def state(self,budget=1):return new_state(self.root,budget,2,2**30,0)
    def test_queue_deduplicates_and_requires_existing_dependencies(self):
        state=self.state();add_task(state,self.task())
        with self.assertRaises(WorkflowError):add_task(state,self.task())
        with self.assertRaises(WorkflowError):add_task(state,self.task('b',['missing']))
        add_task(state,self.task('b',['a']))
        self.assertEqual(state['tasks']['b']['dependencies'],['a'])
    def test_queue_uses_one_shared_reservation_budget(self):
        state=self.state(.75);a=add_task(state,self.task('a',hours=.5));b=add_task(state,self.task('b',hours=.5))
        self.assertTrue(can_reserve(state,a));state['reserved_gpu_hours']+=a['max_gpu_hours']
        self.assertFalse(can_reserve(state,b))
    def test_queue_parallelism_is_bounded_and_audited(self):
        state=self.state();set_parallel(state,4)
        self.assertEqual(state['max_parallel'],4)
        self.assertEqual(state['parallelism_changes'][-1]['max_parallel'],4)
        with self.assertRaises(WorkflowError):set_parallel(state,5)
    def test_render_benchmark_is_explicitly_bounded(self):
        task=self.task('benchmark')
        task['kind']='render_benchmark'
        task['argv']=[sys.executable,str(self.root/'benchmark_render.py'),'--config',str(self.config)]
        state=self.state();added=add_task(state,task)
        self.assertEqual(added['kind'],'render_benchmark')
        task=self.task('bad_benchmark');task['kind']='render_benchmark'
        with self.assertRaises(WorkflowError):add_task(state,task)

    def test_initialization_diagnostic_is_narrowly_scoped(self):
        task=self.task('probe');task['kind']='initialization_diagnostic'
        task['argv']=[sys.executable,str(self.root/'vggt_confidence_probe.py'),'--out',str(self.root/'x')]
        self.assertEqual(add_task(self.state(),task)['kind'],'initialization_diagnostic')
        task=self.task('splat_probe');task['kind']='initialization_diagnostic'
        task['argv']=[sys.executable,str(self.root/'diagnose_splat_initialization.py'),'--out',str(self.root/'y')]
        self.assertEqual(add_task(self.state(),task)['kind'],'initialization_diagnostic')
        bad=self.task('bad_probe');bad['kind']='initialization_diagnostic'
        with self.assertRaises(WorkflowError):add_task(self.state(),bad)
    def test_explicit_bgs_retry_reaches_inner_bounded_runner(self):
        task=self.task('retry')
        first=task_snapshot(task,1);second=task_snapshot(task,2)
        self.assertNotIn('--retry-failed',first['argv'])
        self.assertEqual(second['argv'][-1],'--retry-failed')
        self.assertEqual(task['argv'][-1],str(self.config))
    def test_retry_clears_stale_runtime_fields_but_preserves_attempt(self):
        state=self.state();task=add_task(state,self.task('retry_state'))
        task.update(status='FAILED',finished_at='old',started_at='old',pid=42,pid_token='x')
        task['attempts']=[{'number':1,'category':'PROCESS_FAILURE'}]
        retry_task(state,'retry_state','dependency cache repaired')
        self.assertEqual(task['status'],'PENDING')
        self.assertEqual(len(task['attempts']),1)
        self.assertNotIn('finished_at',task)
        self.assertNotIn('pid',task)
    def test_worker_writes_timeout_receipt(self):
        task={'identity_hash':'fixed','argv':[sys.executable,'-c','import time;time.sleep(30)'],
              'cwd':str(self.root),'env':{},'gpu_index':0,'timeout_s':.05}
        snapshot=self.root/'snapshot.json';receipt=self.root/'receipt.json'
        atomic_json(snapshot,{'task':task})
        self.assertEqual(worker(snapshot,receipt,file_hash(snapshot)),1)
        self.assertEqual(read_json(receipt)['status'],'TIMED_OUT')
    def test_worker_rejects_modified_snapshot(self):
        snapshot=self.root/'snapshot.json';receipt=self.root/'receipt.json'
        atomic_json(snapshot,{'task':{'identity_hash':'fixed'}});expected=file_hash(snapshot)
        atomic_json(snapshot,{'task':{'identity_hash':'changed'}})
        with self.assertRaises(WorkflowError):worker(snapshot,receipt,expected)
        self.assertFalse(receipt.exists())

    def test_queue_failure_category_follows_bounded_inner_log(self):
        from bgs.queue import _category
        inner=self.root/'MixedCaseInner.log';inner.write_text('FloatingPointError: Non-finite model state\n')
        outer=self.root/'outer.log';outer.write_text(f'FAILED; see {inner}\n')
        self.assertEqual(_category(outer,'FAILED'),'NAN_NONFINITE')


class SplitTests(unittest.TestCase):
    def test_final_manifest_requires_frozen_predeclared_run(self):
        from bgs.data import validate_manifest,validate_final_run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);protocol=root/'protocol.json'
            atomic_json(protocol,{"schema":1,"status":"FROZEN_BEFORE_SCORING","matrix_id":"m1",
                        "frozen_at":"2026-10-10T00:00:00Z","scene_ids":["scene"],"run_ids":["allowed"]})
            manifest={"schema":1,"scene_id":"scene","partition":"final","train":["a","b","c"],
                      "holdout":["d"],"images":{x:{} for x in "abcd"},"final_protocol":str(protocol),
                      "final_freeze":{"status":"FROZEN_BEFORE_SCORING","matrix_id":"m1",
                        "frozen_at":"2026-10-10T00:00:00Z","protocol_sha256":file_hash(protocol)}}
            validate_manifest(manifest);validate_final_run(manifest,"allowed")
            with self.assertRaises(WorkflowError):validate_final_run(manifest,"not_frozen")

    def test_viewer_camera_uses_calibrated_intrinsics(self):
        import numpy as np
        from bgs.train_adapter import apply_viewer_camera,viewer_camera_parameters
        c2w=np.eye(4);K=np.array([[500.,0,400.],[0,500.,300.],[0,0,1.]])
        data={'camtoworld':c2w,'K':K,'image':np.zeros((600,800,3))}
        got,fov,aspect=viewer_camera_parameters(data)
        self.assertTrue(np.array_equal(got,c2w))
        self.assertAlmostEqual(fov,2*np.arctan(.6))
        self.assertAlmostEqual(aspect,4/3)
        class ReadOnlyAspectCamera:
            aspect=16/9
        camera=ReadOnlyAspectCamera()
        rotation=type('Rotation',(),{'wxyz':np.array([1.,0.,0.,0.])})()
        self.assertAlmostEqual(apply_viewer_camera(camera,data,lambda _:rotation),4/3)
        self.assertAlmostEqual(camera.aspect,16/9)
        self.assertAlmostEqual(camera.fov,fov)

    def test_resume_qualification_keeps_outer_gpu_assignment(self):
        from scripts.qualify_resume import selected_visible_device
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'3'}):
            self.assertEqual(selected_visible_device({'gpu':'0'}),'3')

    def test_resume_trace_does_not_compare_stale_previous_step_locals(self):
        from scripts.qualify_resume import _trace_local_names
        self.assertEqual(_trace_local_names('sample'),['image_ids'])
        self.assertNotIn('loss',_trace_local_names('forward'))
        self.assertIn('loss',_trace_local_names('backward'))

    def test_resume_trace_ignores_only_culled_projected_means(self):
        import torch
        from scripts.qualify_resume import _compare_tree,_semantic_trace_record
        left={'renders':torch.ones(1),'render_info':{
              'radii':torch.tensor([[2,0]]),'means2d':torch.tensor([[[1.,2.],[9.,9.]]])}}
        right={'renders':torch.ones(1),'render_info':{
               'radii':torch.tensor([[2,0]]),'means2d':torch.tensor([[[1.,2.],[-4.,7.]]])}}
        self.assertFalse(_compare_tree(left,right)['exact'])
        self.assertTrue(_compare_tree(_semantic_trace_record(left),
                                      _semantic_trace_record(right))['exact'])
        right['render_info']['means2d'][0,0,0]=1.5
        self.assertFalse(_compare_tree(_semantic_trace_record(left),
                                       _semantic_trace_record(right))['exact'])

    def test_formal_diagnostic_steps_are_predeclared(self):
        from bgs.__main__ import diagnostic_steps
        self.assertEqual(diagnostic_steps(200),[])
        self.assertEqual(diagnostic_steps(7000),[1000,3000,7000])

    def test_colmap_cpu_commands_are_explicit(self):
        from bgs.initialize import _colmap_commands
        commands=_colmap_commands({"colmap_binary":"/env/bin/colmap","colmap_use_gpu":False},
                                  Path("db"),Path("images"),Path("models"))
        self.assertEqual(commands[0][-2:],["--FeatureExtraction.use_gpu","0"])
        self.assertEqual(commands[1][-2:],["--FeatureMatching.use_gpu","0"])
        self.assertEqual(commands[0][0],"/env/bin/colmap")

    def test_vggt_baseline_policy_is_explicit(self):
        from bgs.initialize import _vggt_policy
        policy=_vggt_policy({'vggt_ba':False,'vggt_confidence':5.0,
                             'vggt_max_points':100000,
                             'vggt_point_sampling':'upstream_seeded_random_limit'})
        self.assertEqual(policy['effective_configuration']['max_points'],100000)
        self.assertFalse(policy['effective_configuration']['bundle_adjustment'])
        self.assertEqual(policy['argument_status']['conf_thres_value']['status'],'effective')
        with self.assertRaises(WorkflowError):_vggt_policy({'vggt_max_points':50000})

    def test_vggt_ba_policy_marks_no_ba_arguments_inactive(self):
        from bgs.initialize import _vggt_policy
        policy=_vggt_policy({'vggt_ba':True,'vggt_confidence':17.,
                             'vggt_max_points':123,'vggt_point_sampling':'not-used'})
        self.assertEqual(policy['branch'],'official_ba_complete_initialization_branch')
        self.assertTrue(policy['effective_configuration']['bundle_adjustment'])
        self.assertEqual(policy['effective_configuration']['tracker'],'VGGSfM predict_tracks (aliked+sp)')
        self.assertEqual(policy['argument_status']['conf_thres_value']['status'],
                         'passed_to_cli_but_inactive_in_ba_branch')
        self.assertNotIn('confidence_threshold',policy['effective_configuration'])

    def test_vggt_ba_pair_source_patch_is_exact(self):
        from bgs.vggt_ba_pair import BA_CALL,BA_REPLACEMENT,patch_demo_source
        source=f"before\n{BA_CALL}\nafter\n"
        patched=patch_demo_source(source)
        self.assertNotIn(BA_CALL,patched);self.assertIn(BA_REPLACEMENT,patched)
        with self.assertRaises(WorkflowError):patch_demo_source("no matching call")
        with self.assertRaises(WorkflowError):patch_demo_source(source+source)

    def test_vggt_wrapper_redirects_only_pinned_dinov2_repository(self):
        from bgs.vggt_wrapper import local_hub_loader
        calls=[]
        def original(*args,**kwargs):
            calls.append((args,kwargs));return "loaded"
        load=local_hub_loader(original,"/fixed/dinov2")
        self.assertEqual(load("facebookresearch/dinov2","dinov2_vitb14_reg",pretrained=True),"loaded")
        self.assertEqual(calls[-1][0][0],"/fixed/dinov2")
        self.assertEqual(calls[-1][1]["source"],"local")
        load("other/repository","model")
        self.assertEqual(calls[-1],(("other/repository","model"),{}))

    def test_vggt_ba_cache_is_hash_qualified_before_gpu_work(self):
        from bgs.initialize import _verify_ba_cache
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoints=root/'hub'/'checkpoints';checkpoints.mkdir(parents=True)
            files={'vggsfm_v2_tracker.pt':b'tracker','dinov2_vitb14_reg4_pretrain.pth':b'dino',
                   'aliked-n16.pth':b'aliked','superpoint_v1.pth':b'superpoint'}
            for name,data in files.items():(checkpoints/name).write_bytes(data)
            pins={'vggsfm_tracker':{'sha256':file_hash(checkpoints/'vggsfm_v2_tracker.pt')},
                  'dinov2':{'checkpoint_sha256':file_hash(checkpoints/'dinov2_vitb14_reg4_pretrain.pth')},
                  'aliked':{'sha256':file_hash(checkpoints/'aliked-n16.pth')},
                  'superpoint':{'sha256':file_hash(checkpoints/'superpoint_v1.pth')}}
            lockfile=root/'runtime.json';atomic_json(lockfile,pins)
            with mock.patch.dict(os.environ,{'TORCH_HOME':str(root)}):
                self.assertEqual(_verify_ba_cache(lockfile),checkpoints)
                (checkpoints/'aliked-n16.pth').write_bytes(b'tampered')
                with self.assertRaises(WorkflowError):_verify_ba_cache(lockfile)

    def test_pycolmap_camera_model_api_variants(self):
        from bgs.data import _camera_model_name
        old=types.SimpleNamespace(model_name="PINHOLE")
        new=types.SimpleNamespace(model=types.SimpleNamespace(name="SIMPLE_PINHOLE"))
        enum_string=types.SimpleNamespace(model="CameraModelId.PINHOLE")
        self.assertEqual(_camera_model_name(old),"PINHOLE")
        self.assertEqual(_camera_model_name(new),"SIMPLE_PINHOLE")
        self.assertEqual(_camera_model_name(enum_string),"PINHOLE")

    def test_nested_disjoint_splits(self):
        names=[f'{i:03d}.png' for i in range(40)]
        a,h=split_names(names,3);b,h2=split_names(names,6);c,_=split_names(names,12)
        self.assertEqual(h,h2);self.assertEqual(a,b[:3]);self.assertEqual(b,c[:6]);self.assertFalse(set(c)&set(h))
    def test_real_image_manifest_and_staging(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir();cameras={}
            for i in range(16):
                name=f'{i:02d}.png';Image.new('RGB',(32,24),(i,0,0)).save(images/name)
                cameras[name]={'width':32,'height':24,'K':[[10,0,16],[0,10,12],[0,0,1]],
                               'c2w':[[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
            ref=root/'ref.json';atomic_json(ref,{'schema':1,'source':'SYNTHETIC TEST FIXTURE','cameras':cameras})
            m=prepare(images,ref,root/'manifest.json',6)
            verify_inputs(m);stage_training(m,root/'staged')
            self.assertEqual({p.name for p in (root/'staged/images').iterdir()},set(m['train']))
            self.assertFalse(any(p.is_symlink() for p in (root/'staged/images').iterdir()))
            (images/m['train'][0]).write_bytes(b'changed')
            with self.assertRaises(WorkflowError):verify_inputs(m)
    def test_overlap_rejected(self):
        with self.assertRaises(WorkflowError):
            validate_manifest({'schema':1,'partition':'development','train':['a','b','c'],'holdout':['c'],'images':{}})


class GeometryTests(unittest.TestCase):
    def test_reference_camera_rejects_reflections(self):
        from bgs.data import validate_camera
        camera={'width':64,'height':64,'K':[[30,0,32],[0,30,32],[0,0,1]],
                'c2w':[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        with self.assertRaises(WorkflowError):validate_camera(camera,'reflected')

    def test_similarity_and_inverse_camera_transform(self):
        import numpy as np
        from bgs.geometry import fit_similarity,reference_to_model
        x=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,2,3]])
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        y=2*(R@x.T).T+np.array([3.,4,5])
        s,r,t,res=fit_similarity(x,y)
        np.testing.assert_allclose(s,2);np.testing.assert_allclose(r,R,atol=1e-12)
        self.assertLess(float(res.max()),1e-12)
        cam=np.eye(4);cam[:3,:3]=R;cam[:3,3]=y[1]
        restored=reference_to_model(cam,s,r,t)
        np.testing.assert_allclose(restored[:3,3],x[1],atol=1e-12)
        np.testing.assert_allclose(restored[:3,:3],np.eye(3),atol=1e-12)
    def test_degenerate_centers_rejected(self):
        from bgs.geometry import fit_similarity
        with self.assertRaises(WorkflowError):fit_similarity([[0,0,0],[1,0,0],[2,0,0]],[[1,0,0],[2,0,0],[3,0,0]])
    def test_pose_perturbation_preserves_rotation_and_anchor(self):
        import numpy as np
        from bgs.geometry import perturb_training_poses
        poses=np.repeat(np.eye(4)[None],4,axis=0);poses[:,:3,3]=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
        result=perturb_training_poses(poses,17,3.,.01)
        np.testing.assert_array_equal(result[0],poses[0])
        for r in result[:,:3,:3]:np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12)
        np.testing.assert_array_equal(result,perturb_training_poses(poses,17,3.,.01))


if __name__=='__main__':unittest.main()
