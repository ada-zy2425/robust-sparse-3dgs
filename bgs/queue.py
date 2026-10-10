"""Persistent single-node experiment queue with one shared finite GPU budget.

The queue is deliberately small: it dispatches independent, already-configured
``bgs run`` or resume-qualification commands.  Each command retains its own
bounded runner receipt while this layer atomically accounts the aggregate
multi-GPU budget and prevents duplicate coordinators.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time

from .core import WorkflowError,atomic_json,digest,file_hash,pid_token,process_alive,read_json


TERMINAL={"SUCCEEDED","FAILED","TIMED_OUT","FAILED_MISSING_OUTPUT","FAILED_CONFIG_TAMPER",
          "FAILED_LOST_WORKER","FAILED_LAUNCH","BLOCKED_DEPENDENCY"}


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _positive(value,label,zero=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or value<0 or (not zero and value==0):
        raise WorkflowError(f"Invalid {label}: {value!r}")
    return value


def new_state(project_root,budget_gpu_hours,max_parallel,disk_budget_bytes,min_free_bytes):
    project_root=str(Path(project_root).resolve())
    _positive(budget_gpu_hours,"budget_gpu_hours")
    if not isinstance(max_parallel,int) or isinstance(max_parallel,bool) or not 1<=max_parallel<=4:
        raise WorkflowError("max_parallel must be an integer in [1,4]")
    _positive(disk_budget_bytes,"disk_budget_bytes")
    _positive(min_free_bytes,"min_free_bytes",zero=True)
    return {"schema":1,"created_at":utcnow(),"updated_at":utcnow(),"project_root":project_root,
            "budget_gpu_hours":float(budget_gpu_hours),"charged_gpu_hours":0.,"reserved_gpu_hours":0.,
            "max_parallel":max_parallel,"disk_budget_bytes":int(disk_budget_bytes),
            "min_free_bytes":int(min_free_bytes),"reserved_disk_bytes":0,"tasks":{},
            "coordinator":{"status":"NOT_STARTED"}}


def _normalized_task(spec):
    required=("task_id","kind","argv","cwd","env","dependencies","config_path","config_sha256",
              "gpu_index","gpu_uuid","gpu_count","max_gpu_hours","timeout_s","disk_reservation_bytes",
              "expected_outputs")
    missing=[x for x in required if x not in spec]
    if missing: raise WorkflowError(f"Missing task fields: {missing}")
    task_id=spec["task_id"]
    if not isinstance(task_id,str) or not task_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in task_id):
        raise WorkflowError(f"Invalid task_id: {task_id!r}")
    if spec["kind"] not in ("bgs_run","resume_qualification","render_benchmark","initialization_diagnostic"):
        raise WorkflowError("Queue only accepts bounded reconstruction, resume, render, or initialization-diagnostic tasks")
    argv=list(spec["argv"])
    if not argv or not all(isinstance(x,str) and x for x in argv): raise WorkflowError("argv must be nonempty strings")
    if spec["kind"]=="bgs_run" and not any(argv[i:i+3]==["-m","bgs","run"] for i in range(len(argv)-2)):
        raise WorkflowError("bgs_run must invoke python -m bgs run")
    if spec["kind"]=="resume_qualification" and not any(x.endswith("qualify_resume.py") for x in argv):
        raise WorkflowError("resume_qualification must invoke qualify_resume.py")
    if spec["kind"]=="render_benchmark" and not any(x.endswith("benchmark_render.py") for x in argv):
        raise WorkflowError("render_benchmark must invoke benchmark_render.py")
    if spec["kind"]=="initialization_diagnostic" and not any(x.endswith("vggt_confidence_probe.py") for x in argv):
        raise WorkflowError("initialization_diagnostic must invoke vggt_confidence_probe.py")
    cwd=str(Path(spec["cwd"]).resolve());config=str(Path(spec["config_path"]).resolve())
    if not Path(cwd).is_absolute() or not Path(config).is_absolute(): raise WorkflowError("cwd/config_path must be absolute")
    if not isinstance(spec["dependencies"],list) or not all(isinstance(x,str) for x in spec["dependencies"]):
        raise WorkflowError("dependencies must be task IDs")
    if not isinstance(spec["env"],dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in spec["env"].items()):
        raise WorkflowError("env must contain string keys and values")
    gpu_count=spec["gpu_count"]
    if gpu_count!=1: raise WorkflowError("This queue assigns exactly one GPU per independent experiment")
    if not isinstance(spec["gpu_index"],int) or isinstance(spec["gpu_index"],bool) or spec["gpu_index"]<0:
        raise WorkflowError("gpu_index must be a nonnegative integer")
    if not isinstance(spec["gpu_uuid"],str) or not spec["gpu_uuid"].startswith("GPU-"):
        raise WorkflowError("gpu_uuid must be explicit")
    max_hours=float(_positive(spec["max_gpu_hours"],"max_gpu_hours"))
    timeout=float(_positive(spec["timeout_s"],"timeout_s"))
    if timeout/3600>max_hours+1e-12: raise WorkflowError("timeout exceeds the task GPU-hour reservation")
    disk=int(_positive(spec["disk_reservation_bytes"],"disk_reservation_bytes",zero=True))
    outputs=[str(Path(x).resolve()) for x in spec["expected_outputs"]]
    if not outputs: raise WorkflowError("expected_outputs cannot be empty")
    immutable={"task_id":task_id,"kind":spec["kind"],"argv":argv,"cwd":cwd,"env":dict(spec["env"]),
       "dependencies":list(spec["dependencies"]),"config_path":config,"config_sha256":spec["config_sha256"],
       "gpu_index":spec["gpu_index"],"gpu_uuid":spec["gpu_uuid"],"gpu_count":1,
       "max_gpu_hours":max_hours,"timeout_s":timeout,"disk_reservation_bytes":disk,
       "expected_outputs":outputs,"priority":int(spec.get("priority",100))}
    immutable["identity_hash"]=digest(immutable)
    return immutable


def add_task(state,spec):
    task=_normalized_task(spec)
    if task["task_id"] in state["tasks"]: raise WorkflowError(f"Duplicate task_id: {task['task_id']}")
    unknown=set(task["dependencies"])-set(state["tasks"])
    if unknown: raise WorkflowError(f"Dependencies must be added first: {sorted(unknown)}")
    task.update(status="PENDING",attempts=[],waiting_reason=None,created_at=utcnow())
    state["tasks"][task["task_id"]]=task;state["updated_at"]=utcnow()
    return task


def can_reserve(state,task):
    return state["charged_gpu_hours"]+state["reserved_gpu_hours"]+task["max_gpu_hours"] <= state["budget_gpu_hours"]+1e-12


def set_parallel(state,max_parallel):
    if not isinstance(max_parallel,int) or isinstance(max_parallel,bool) or not 1<=max_parallel<=4:
        raise WorkflowError("max_parallel must be in [1,4]")
    state["max_parallel"]=max_parallel
    state["parallelism_changes"]=state.get("parallelism_changes",[])+[
        {"at":utcnow(),"max_parallel":max_parallel}]


def task_snapshot(task,attempt_no):
    """Build the immutable worker view, adding only explicit retry semantics."""
    snapshot={k:v for k,v in task.items() if k not in ("attempts","status","waiting_reason")}
    snapshot["argv"]=list(snapshot["argv"])
    if attempt_no>1 and snapshot["kind"]=="bgs_run" and "--retry-failed" not in snapshot["argv"]:
        snapshot["argv"].append("--retry-failed")
    return snapshot


def retry_task(state,task_id,reason):
    task=state["tasks"].get(task_id)
    if task is None:raise WorkflowError("Unknown task")
    if task["status"] not in TERMINAL-{"SUCCEEDED","BLOCKED_DEPENDENCY"}:
        raise WorkflowError("Only a terminal failed task can be retried")
    if len(task["attempts"])>=2:raise WorkflowError("Retry cap reached")
    task["status"]="PENDING";task["waiting_reason"]=f"explicit retry: {reason}"
    task["retry_reason"]=reason
    for key in ("finished_at","started_at","pid","pid_token","output_hashes"):
        task.pop(key,None)
    return task


def _lock(path,blocking=True):
    f=Path(path).open("a+")
    flags=fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try: fcntl.flock(f,flags)
    except OSError:
        f.close();raise WorkflowError(f"Lock already held: {path}")
    return f


def _save(path,state):
    state["updated_at"]=utcnow();atomic_json(path,state)


def _gpu_state():
    q=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                      "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=20)
    gpus={}
    for line in q.stdout.splitlines():
        fields=[x.strip() for x in line.split(",")]
        if len(fields)>=5:gpus[int(fields[0])]={"uuid":fields[1],"name":fields[2],"memory_used_mib":int(fields[3]),"utilization":int(fields[4])}
    p=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20)
    busy=set()
    if p.returncode==0:
        for line in p.stdout.splitlines():
            if line.strip():busy.add(line.split(",",1)[0].strip())
    return gpus,busy


def _project_bytes(path):
    p=subprocess.run(["du","-sb",str(Path(path).resolve())],capture_output=True,text=True,check=True,timeout=120)
    return int(p.stdout.split()[0])


def _category(log,status):
    if status=="TIMED_OUT":return "TIMEOUT"
    try:raw=Path(log).read_text(errors="replace")[-20000:]
    except OSError:raw=""
    text=raw.lower()
    # ``bgs run`` reports the bounded inner-stage log path in its outer log.
    # Include that tail so the aggregate queue does not collapse a concrete
    # NaN/OOM/registration failure into a generic process failure.
    match=re.search(r"see\s+(\S+)",raw,flags=re.IGNORECASE)
    if match:
        try:text+='\n'+Path(match.group(1)).read_text(errors='replace')[-20000:].lower()
        except OSError:pass
    patterns=(("out of memory","OOM"),("non-finite","NAN_NONFINITE"),("nan","NAN_NONFINITE"),
              ("registration coverage","REGISTRATION"),("budget insufficient","BUDGET"),
              ("identity","IDENTITY_CONFLICT"),("no space left","DISK_FULL"),
              ("resume","RESUME_MISMATCH"))
    return next((label for pattern,label in patterns if pattern in text),"PROCESS_FAILURE")


def _settle(state,task,receipt_path):
    receipt=read_json(receipt_path)
    if receipt.get("identity_hash")!=task["identity_hash"]: raise WorkflowError("Worker receipt identity mismatch")
    state["reserved_gpu_hours"]=max(0.,state["reserved_gpu_hours"]-task["max_gpu_hours"])
    state["reserved_disk_bytes"]=max(0,state["reserved_disk_bytes"]-task["disk_reservation_bytes"])
    actual=float(receipt["elapsed_s"])*task["gpu_count"]/3600
    state["charged_gpu_hours"]+=actual
    attempt=task["attempts"][-1];attempt.update(receipt);attempt["gpu_hours"]=actual
    if receipt["status"]=="SUCCEEDED":
        missing=[p for p in task["expected_outputs"] if not Path(p).is_file() or Path(p).stat().st_size==0]
        if missing:
            task["status"]="FAILED_MISSING_OUTPUT";attempt["category"]="MISSING_OUTPUT";attempt["missing_outputs"]=missing
        else:
            task["status"]="SUCCEEDED";task["output_hashes"]={p:file_hash(p) for p in task["expected_outputs"]}
    else:
        task["status"]=receipt["status"];attempt["category"]=_category(attempt["log"],task["status"])
    task["finished_at"]=receipt["finished_at"]


def worker(snapshot_path,receipt_path,snapshot_sha256):
    if file_hash(snapshot_path)!=snapshot_sha256: raise WorkflowError("Task snapshot hash mismatch")
    snapshot=read_json(snapshot_path);task=snapshot["task"]
    started=utcnow();start=time.monotonic();proc=None;status="FAILED";exit_code=None;error=None
    old={}
    def forward(signum,frame):
        nonlocal error
        error=f"worker received signal {signum}"
        if proc is not None and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):old[sig]=signal.signal(sig,forward)
        env=os.environ.copy();env.update(task["env"]);env["CUDA_VISIBLE_DEVICES"]=str(task["gpu_index"])
        env["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";env["PYTHONUNBUFFERED"]="1"
        proc=subprocess.Popen(task["argv"],cwd=task["cwd"],env=env,start_new_session=True)
        try:exit_code=proc.wait(timeout=task["timeout_s"])
        except subprocess.TimeoutExpired:
            status="TIMED_OUT";error=f"Exceeded {task['timeout_s']} seconds"
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        else:status="SUCCEEDED" if exit_code==0 else "FAILED"
    except BaseException as e:
        error=f"{type(e).__name__}: {e}"
        if proc is not None and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
            except (ProcessLookupError,subprocess.TimeoutExpired):pass
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
        atomic_json(receipt_path,{"identity_hash":task["identity_hash"],"started_at":started,
          "finished_at":utcnow(),"elapsed_s":time.monotonic()-start,"status":status,
          "exit_code":exit_code,"error":error})
    return 0 if status=="SUCCEEDED" else 1


def _tick(state_path):
    state_lock=_lock(str(state_path)+".lock")
    try:
        state=read_json(state_path);changed=False
        # A coordinator crash before Popen leaves an explicit, recoverable reservation.
        for task in state["tasks"].values():
            if task["status"]=="RESERVED":
                state["reserved_gpu_hours"]=max(0.,state["reserved_gpu_hours"]-task["max_gpu_hours"])
                state["reserved_disk_bytes"]=max(0,state["reserved_disk_bytes"]-task["disk_reservation_bytes"])
                task["status"]="PENDING";task["waiting_reason"]="recovered pre-launch reservation";changed=True
        for task in state["tasks"].values():
            if task["status"]!="RUNNING":continue
            receipt=task["attempts"][-1]["receipt"]
            if Path(receipt).is_file():_settle(state,task,receipt);changed=True;continue
            if not process_alive(task["pid"],task.get("pid_token")):
                state["reserved_gpu_hours"]=max(0.,state["reserved_gpu_hours"]-task["max_gpu_hours"])
                state["reserved_disk_bytes"]=max(0,state["reserved_disk_bytes"]-task["disk_reservation_bytes"])
                state["charged_gpu_hours"]+=task["max_gpu_hours"]
                task["status"]="FAILED_LOST_WORKER";task["finished_at"]=utcnow()
                task["attempts"][-1].update(finished_at=utcnow(),gpu_hours=task["max_gpu_hours"],category="LOST_WORKER_CONSERVATIVE_CHARGE")
                changed=True
        for task in state["tasks"].values():
            if task["status"]=="PENDING" and any(state["tasks"][d]["status"] in TERMINAL-{"SUCCEEDED"} for d in task["dependencies"]):
                task["status"]="BLOCKED_DEPENDENCY";task["waiting_reason"]="dependency failed";changed=True
        running=[t for t in state["tasks"].values() if t["status"]=="RUNNING"]
        slots=state["max_parallel"]-len(running);active={t["gpu_uuid"] for t in running}
        gpus,busy=_gpu_state()
        ready=sorted((t for t in state["tasks"].values() if t["status"]=="PENDING" and
                      all(state["tasks"][d]["status"]=="SUCCEEDED" for d in t["dependencies"])),
                     key=lambda t:(t["priority"],t["task_id"]))
        for task in ready:
            if slots<=0:break
            task["waiting_reason"]=None
            if not can_reserve(state,task):task["waiting_reason"]="global GPU budget unavailable";continue
            gpu=gpus.get(task["gpu_index"])
            if gpu is None or gpu["uuid"]!=task["gpu_uuid"]:
                task["waiting_reason"]="GPU index/UUID mapping mismatch";continue
            if task["gpu_uuid"] in busy or task["gpu_uuid"] in active:
                task["waiting_reason"]="assigned GPU is busy";continue
            if not Path(task["config_path"]).is_file() or file_hash(task["config_path"])!=task["config_sha256"]:
                task["status"]="FAILED_CONFIG_TAMPER";task["finished_at"]=utcnow();changed=True;continue
            usage=shutil.disk_usage(state["project_root"]);project_size=_project_bytes(state["project_root"])
            if (usage.free<state["min_free_bytes"]+task["disk_reservation_bytes"] or
                project_size+state["reserved_disk_bytes"]+task["disk_reservation_bytes"]>state["disk_budget_bytes"]):
                task["waiting_reason"]="disk reservation unavailable";continue
            attempt_no=len(task["attempts"])+1
            task_dir=Path(state_path).parent/"tasks"/task["task_id"];task_dir.mkdir(parents=True,exist_ok=True)
            snapshot=task_dir/f"snapshot.{attempt_no:02d}.json";receipt=task_dir/f"receipt.{attempt_no:02d}.json"
            log=task_dir/f"attempt.{attempt_no:02d}.log"
            atomic_json(snapshot,{"schema":1,"task":task_snapshot(task,attempt_no)})
            snapshot_hash=file_hash(snapshot)
            state["reserved_gpu_hours"]+=task["max_gpu_hours"];state["reserved_disk_bytes"]+=task["disk_reservation_bytes"]
            task["status"]="RESERVED";_save(state_path,state)
            try:
                with log.open("ab",buffering=0) as f:
                    proc=subprocess.Popen([sys.executable,"-m","bgs.queue","worker","--snapshot",str(snapshot),
                        "--receipt",str(receipt),"--snapshot-sha256",snapshot_hash],cwd=task["cwd"],
                        env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1])},stdout=f,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
            except Exception as e:
                state["reserved_gpu_hours"]-=task["max_gpu_hours"];state["reserved_disk_bytes"]-=task["disk_reservation_bytes"]
                task["status"]="FAILED_LAUNCH";task["finished_at"]=utcnow();task["attempts"].append({"number":attempt_no,
                    "log":str(log),"error":str(e),"category":"LAUNCH"});changed=True;continue
            task["status"]="RUNNING";task["pid"]=proc.pid;task["pid_token"]=pid_token(proc.pid);task["gpu_name"]=gpu["name"]
            task["started_at"]=utcnow();task["attempts"].append({"number":attempt_no,"started_at":task["started_at"],
                "log":str(log),"snapshot":str(snapshot),"snapshot_sha256":snapshot_hash,"receipt":str(receipt)})
            active.add(task["gpu_uuid"]);slots-=1;changed=True
        state["coordinator"].update(status="RUNNING",pid=os.getpid(),pid_token=pid_token(os.getpid()),heartbeat=utcnow())
        _save(state_path,state)
        return state
    finally:
        fcntl.flock(state_lock,fcntl.LOCK_UN);state_lock.close()


def run_coordinator(state_path,poll_s,max_runtime_s):
    guard=_lock(str(state_path)+".coordinator.lock",blocking=False);started=time.monotonic()
    try:
        while True:
            state=_tick(state_path)
            if all(t["status"] in TERMINAL for t in state["tasks"].values()):break
            if time.monotonic()-started>=max_runtime_s:break
            time.sleep(poll_s)
        lock=_lock(str(state_path)+".lock")
        try:
            state=read_json(state_path);running=any(t["status"]=="RUNNING" for t in state["tasks"].values())
            state["coordinator"].update(status="EXITED_WITH_RUNNING_TASKS" if running else "IDLE",finished_at=utcnow())
            _save(state_path,state)
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        return 0
    finally:fcntl.flock(guard,fcntl.LOCK_UN);guard.close()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("init");q.add_argument("--state",required=True);q.add_argument("--project-root",required=True)
    q.add_argument("--budget-gpu-hours",type=float,required=True);q.add_argument("--max-parallel",type=int,default=2)
    q.add_argument("--disk-budget-gib",type=float,required=True);q.add_argument("--min-free-gib",type=float,required=True)
    q=sub.add_parser("add");q.add_argument("--state",required=True);q.add_argument("--task",required=True)
    q=sub.add_parser("run");q.add_argument("--state",required=True);q.add_argument("--poll-s",type=float,default=5)
    q.add_argument("--max-runtime-s",type=float,default=14400)
    q=sub.add_parser("status");q.add_argument("--state",required=True)
    q=sub.add_parser("retry");q.add_argument("--state",required=True);q.add_argument("--task-id",required=True);q.add_argument("--reason",required=True)
    q=sub.add_parser("set-parallel");q.add_argument("--state",required=True);q.add_argument("--max-parallel",type=int,required=True)
    q=sub.add_parser("worker");q.add_argument("--snapshot",required=True);q.add_argument("--receipt",required=True);q.add_argument("--snapshot-sha256",required=True)
    args=p.parse_args()
    if args.command=="worker":return worker(args.snapshot,args.receipt,args.snapshot_sha256)
    state_path=Path(args.state).resolve()
    if args.command=="init":
        if state_path.exists():raise WorkflowError("Refusing to overwrite queue state")
        atomic_json(state_path,new_state(args.project_root,args.budget_gpu_hours,args.max_parallel,
            args.disk_budget_gib*2**30,args.min_free_gib*2**30));return 0
    if not state_path.is_file():raise WorkflowError("Queue state does not exist")
    if args.command=="run":return run_coordinator(state_path,args.poll_s,args.max_runtime_s)
    lock=_lock(str(state_path)+".lock")
    try:
        state=read_json(state_path)
        if args.command=="add":add_task(state,read_json(args.task));_save(state_path,state)
        elif args.command=="retry":
            retry_task(state,args.task_id,args.reason)
            _save(state_path,state)
        elif args.command=="set-parallel":
            set_parallel(state,args.max_parallel)
            _save(state_path,state)
        else:
            print(json.dumps({"budget_gpu_hours":state["budget_gpu_hours"],"charged_gpu_hours":state["charged_gpu_hours"],
                "reserved_gpu_hours":state["reserved_gpu_hours"],"remaining_unreserved_gpu_hours":max(0.,state["budget_gpu_hours"]-state["charged_gpu_hours"]-state["reserved_gpu_hours"]),
                "coordinator":state["coordinator"],"tasks":{k:{x:v.get(x) for x in ("status","gpu_uuid","pid","started_at","finished_at","waiting_reason")} for k,v in state["tasks"].items()}},indent=2))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return 0


if __name__=="__main__":
    try:raise SystemExit(main())
    except WorkflowError as e:print(f"BLOCKED: {e}",file=sys.stderr);raise SystemExit(2)
