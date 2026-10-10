import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys


def probe(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=15)
        return {"exit_code":p.returncode,"stdout":p.stdout[-12000:],"stderr":p.stderr[-3000:]}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {"error":str(e)}


def inspect(root):
    root=Path(root).expanduser().resolve()
    packages={}
    for name in ("torch","gsplat","numpy","pycolmap","torchmetrics","Pillow","viser","nerfview"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    info={"host":socket.gethostname(),"platform":platform.platform(),"python":sys.executable,
          "python_version":sys.version,"root":str(root),"root_exists":root.is_dir(),
          "root_writable":os.access(root,os.W_OK),"packages":packages,
          "binaries":{k:shutil.which(k) for k in ("git","colmap","nvcc","nvidia-smi","tmux","sbatch","codex")},
          "gpus":probe(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu","--format=csv,noheader"]),
          "gpu_processes":probe(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader"])}
    if root.exists():
        usage=shutil.disk_usage(root)
        info["disk"]={"total_bytes":usage.total,"used_bytes":usage.used,"free_bytes":usage.free}
    return info
