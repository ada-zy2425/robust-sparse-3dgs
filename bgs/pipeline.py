from pathlib import Path
import html
import os
import sys

from .core import WorkflowError, read_json, atomic_json, digest, file_hash, identifier, positive, Engine, validate_metrics
from .data import validate_manifest,verify_inputs,stage_training,validate_final_run
from .splat_numerics import POLICIES,UPSTREAM_POLICY


def load_spec(path):
    spec=read_json(path)
    required=["run_id","root","manifest","gsplat_root","python","initializer","gpu","gpu_hours",
              "steps","max_edge","seed","noise_seed","checkpoint_every","max_gaussians","timeout_s"]
    for key in required:
        if key not in spec: raise WorkflowError(f"Missing config field: {key}")
    identifier(spec["run_id"])
    if spec["initializer"] not in ("colmap","vggt"): raise WorkflowError("initializer must be colmap or vggt")
    if spec.get("splat_numerics_policy",UPSTREAM_POLICY) not in POLICIES:
        raise WorkflowError("Unknown splat_numerics_policy")
    if not str(spec["gpu"]).isdigit(): raise WorkflowError("gpu must be one physical GPU index")
    for key in ("steps","max_edge","checkpoint_every","max_gaussians"):
        positive(spec[key],key)
        if not isinstance(spec[key],int) or isinstance(spec[key],bool): raise WorkflowError(f"{key} must be integer")
    for key in ("seed","noise_seed"):
        if not isinstance(spec[key],int) or isinstance(spec[key],bool) or not 0<=spec[key]<2**31:
            raise WorkflowError(f"{key} must be an integer in [0,2^31)")
    positive(spec["gpu_hours"],"gpu_hours",zero=True)
    positive(spec["timeout_s"],"timeout_s")
    positive(spec.get("rotation_degrees",0),"rotation_degrees",zero=True)
    positive(spec.get("translation_fraction",0),"translation_fraction",zero=True)
    for key in ("root","manifest","gsplat_root","python"):
        if not Path(spec[key]).is_absolute(): raise WorkflowError(f"{key} must be absolute")
    return spec


def report(run_dir,manifest,result):
    run_dir=Path(run_dir)
    images=[]
    for row in result["images"]:
        images.append("<article><h3>"+html.escape(row["name"])+"</h3><p>PSNR "+f"{row['psnr']:.3f}"+
             " · SSIM "+f"{row['ssim']:.4f}"+" · LPIPS "+f"{row['lpips']:.4f}"+"</p><img src=\"training/"+
             html.escape(row["visualization"],quote=True)+"\"></article>")
    page="<!doctype html><html lang='zh'><meta charset='utf-8'><title>B: Reconstruction report</title>"+\
         "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f5f6f8;color:#182330}"+\
         "article{background:white;padding:20px;margin:18px 0;border-radius:12px}img{width:100%}code{word-break:break-all}</style>"+\
         "<h1>3DGS 留出视角评测</h1><p>左：参考图像；中：预测；右：绝对误差。</p>"+\
         "<p>场景："+html.escape(manifest["scene_id"])+"；训练视角："+str(len(manifest["train"]))+\
         "；留出视角："+str(len(manifest["holdout"]))+"。相机来源："+html.escape(manifest["reference_source"])+"。</p>"+\
         "<p>逐场景结果仅用于该场景；不会自动解释为跨场景泛化或 SOTA。</p>"+"".join(images)+"</html>"
    (run_dir/"report.html").write_text(page,encoding="utf-8")


def run(config,retry_failed=False,recover_stale=False):
    spec=load_spec(config)
    manifest=validate_manifest(read_json(spec["manifest"]))
    validate_final_run(manifest,spec["run_id"])
    verify_inputs(manifest)
    root=Path(spec["root"]); run_dir=root/"runs"/spec["run_id"]
    package_root=Path(__file__).resolve().parents[1]
    code={str(p.relative_to(package_root)):file_hash(p) for p in sorted((package_root/"bgs").glob("*.py"))}
    fingerprint={"spec":spec,"manifest":manifest,"source":code,"upstream":read_json(package_root/"upstream.lock.json")}
    fingerprint_hash=digest(fingerprint)
    with Engine(root,spec["gpu_hours"]) as engine:
        receipt=run_dir/"experiment.json"
        if receipt.exists() and read_json(receipt)["fingerprint_hash"]!=fingerprint_hash:
            raise WorkflowError("Code/config/input changed under this run id; create a new run id")
        run_dir.mkdir(parents=True,exist_ok=True)
        if not receipt.exists(): atomic_json(receipt,{"fingerprint_hash":fingerprint_hash,**fingerprint})
        atomic_json(run_dir/"config.json",spec)
        prepared=run_dir/"input"; training=run_dir/"training"
        stage_training(manifest,prepared)
        env={"CUDA_VISIBLE_DEVICES":str(spec["gpu"]),"CUDA_DEVICE_ORDER":"PCI_BUS_ID",
             "PYTHONPATH":str(package_root),"QT_QPA_PLATFORM":"offscreen"}
        base=[spec["python"],"-m","bgs","worker","--config",str(run_dir/"config.json")]
        engine.run(spec["run_id"],"initialize",base+["--phase","initialize"],str(package_root),env,
                   [prepared/"init_receipt.json"],spec["timeout_s"],retry_failed=retry_failed,
                   recover_stale=recover_stale,fingerprint=fingerprint_hash)
        init=read_json(prepared/"init_receipt.json")
        for name,expected in init["files"].items():
            if file_hash(Path(init["model"])/name)!=expected: raise WorkflowError("Initialization artifact modified")
        engine.run(spec["run_id"],"train_evaluate",base+["--phase","train"],str(package_root),env,
                   [training/"complete.json",training/"metrics.json",training/"model.ply",training/"progress.json",
                    training/"initialization_numerics.json"],
                   spec["timeout_s"],retry_failed=retry_failed,recover_stale=recover_stale,
                   fingerprint={"experiment":fingerprint_hash,"initialization":init})
        metrics=validate_metrics(training/"metrics.json",manifest["holdout"])
        report(run_dir,manifest,metrics)
        stages=[value for key,value in engine.ledger["stages"].items() if key.startswith(spec["run_id"]+"/")]
        cost=sum(a["gpu_hours"] for s in stages for a in s["attempts"] if "gpu_hours" in a)
        summary={"run_id":spec["run_id"],"scene":manifest["scene_id"],"status":"VALID",
           "mean":metrics["mean"],"gpu_hours_known_attempts":cost,
           "campaign_gpu_hours_charged":engine.ledger["charged_gpu_hours"],
           "end_to_end_seconds_known_attempts":sum(a["elapsed_s"] for s in stages for a in s["attempts"] if "elapsed_s" in a),
           "model_bytes":metrics["model_bytes"],"render_fps":metrics["render_fps"],
           "peak_allocated_bytes":metrics["peak_allocated_bytes"],
           "decision":"READY_FOR_DIAGNOSIS","new_algorithm_claim":False}
        atomic_json(run_dir/"summary.json",summary)
        atomic_json(root/"state.json",{"latest_run":spec["run_id"],"latest_status":"VALID",
            "next_action":"Inspect failure views; compare paired initialization baseline before proposing a mechanism",
            "remaining_gpu_hours":max(0.,spec["gpu_hours"]-engine.ledger["charged_gpu_hours"])})
    return summary


def worker(config,phase):
    spec=load_spec(config); manifest=validate_manifest(read_json(spec["manifest"]))
    validate_final_run(manifest,spec["run_id"])
    folder=Path(spec["root"])/"runs"/spec["run_id"]
    if phase=="initialize":
        from .initialize import initialize
        initialize(spec,manifest,folder/"input")
    else:
        from .train_adapter import execute
        execute(spec,manifest,folder/"input",folder/"training",viewer=phase=="viewer")
