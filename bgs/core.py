"""Finite, locked stage execution. Standard library only; POSIX hosts."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


class WorkflowError(RuntimeError):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    def reject_constant(value):
        raise WorkflowError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def positive(value, label, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkflowError(f"{label} must be a number")
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise WorkflowError(f"Invalid {label}: {value}")
    return value


def identifier(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", value):
        raise WorkflowError(f"Invalid identifier: {value!r}")
    return value


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise WorkflowError(f"Another workflow is running, or filesystem locks unsupported: {path}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def pid_token(pid):
    """Linux PID-reuse protection; conservative fallback elsewhere."""
    p = Path(f"/proc/{pid}/stat")
    try:
        # The command field may contain spaces or parentheses.
        fields = p.read_text().rsplit(")", 1)[1].split()
        return fields[19]  # starttime, original field 22
    except (OSError, IndexError):
        return None


def process_alive(pid, token=None):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return token is None or pid_token(pid) == token


def stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit while grandchildren remain in its process group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def validate_metrics(path, expected_names):
    result = read_json(path)
    if result.get("status") != "VALID":
        raise WorkflowError("Evaluator did not report VALID")
    records = result.get("images", [])
    names = [x.get("name") for x in records]
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise WorkflowError("Evaluation denominator mismatch or duplicate image")
    if not names:
        raise WorkflowError("Empty evaluation set")
    for row in records:
        for metric in ("psnr", "ssim", "lpips"):
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise WorkflowError(f"Missing/non-finite {metric}: {row.get('name')}")
        if not -1 <= row["ssim"] <= 1 or row["lpips"] < 0:
            raise WorkflowError("Metric outside mathematical range")
    # Recompute aggregation rather than trusting a separately supplied average.
    result["mean"] = {m: sum(r[m] for r in records) / len(records)
                      for m in ("psnr", "ssim", "lpips")}
    return result


class Engine:
    """One campaign lock; stage outputs are immutable after success.

    A stale attempt is never silently relaunched. On a killed runner its full
    reservation stays charged until explicitly recovered after liveness checks.
    Wall-clock GPU occupancy is accounted, not device utilization.
    """

    def __init__(self, root, budget_gpu_hours, max_attempts=2):
        self.root = Path(root).resolve()
        self.budget = positive(budget_gpu_hours, "budget_gpu_hours", zero=True)
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= 5:
            raise WorkflowError("max_attempts must be between 1 and 5")
        self.max_attempts = max_attempts
        self.ledger_path = self.root / "ledger.json"

    def __enter__(self):
        self.guard = lock(self.root / ".engine.lock")
        self.guard.__enter__()
        try:
            self.ledger = read_json(self.ledger_path) if self.ledger_path.exists() else {
                "schema": 1, "charged_gpu_hours": 0.0, "stages": {}}
        except BaseException:
            self.guard.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *args):
        return self.guard.__exit__(*args)

    def save(self):
        self.ledger["updated_at"] = now()
        atomic_json(self.ledger_path, self.ledger)

    def run(self, run_id, stage, argv, cwd, env, outputs, timeout_s,
            gpu_count=1, retry_failed=False, recover_stale=False, fingerprint=None):
        identifier(run_id)
        identifier(stage)
        positive(timeout_s, "timeout_s")
        if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 0:
            raise WorkflowError("gpu_count must be a nonnegative integer")
        if not argv or not all(isinstance(x, str) and x for x in argv):
            raise WorkflowError("argv must be a nonempty array of strings")
        key = f"{run_id}/{stage}"
        info = {"argv": argv, "cwd": str(Path(cwd).resolve()), "env": env,
                "outputs": [str(Path(p).resolve()) for p in outputs],
                "timeout_s": timeout_s, "gpu_count": gpu_count, "input": fingerprint}
        task_hash = digest(info)
        old = self.ledger["stages"].get(key)
        if old and old["task_hash"] != task_hash:
            raise WorkflowError(f"Task changed under existing identity {key}; use a new run id")
        if old and old["status"] == "SUCCEEDED":
            for p, expected in old["output_hashes"].items():
                if not Path(p).is_file() or file_hash(p) != expected:
                    raise WorkflowError(f"Completed artifact changed/missing: {p}")
            return old
        if old and old["status"] == "RUNNING":
            if not recover_stale:
                raise WorkflowError(f"Unresolved RUNNING stage {key}; inspect then use --recover-stale")
            if old.get("host") != socket.gethostname():
                raise WorkflowError("Recover on the original host; remote process liveness is unknown")
            if old.get("pid") is None:
                raise WorkflowError("Runner died before recording its child PID; verify processes manually before recovery")
            if old.get("pid") and process_alive(old["pid"], old.get("pid_token")):
                raise WorkflowError("Original child process still exists; refusing a duplicate launch")
            old["status"] = "INTERRUPTED"
            old["conservative_charge"] = True
            self.save()
        attempts = old.get("attempts", []) if old else []
        if attempts and not retry_failed:
            raise WorkflowError(f"Previous stage failed: {key}; inspect log then use --retry-failed")
        if len(attempts) >= self.max_attempts:
            raise WorkflowError(f"Retry cap reached for {key}; diagnose before making a new experiment")
        # Reserve timeout plus termination allowance so an admitted job fits.
        reserved = (timeout_s + 5.0) * gpu_count / 3600
        if self.ledger["charged_gpu_hours"] + reserved > self.budget + 1e-12:
            raise WorkflowError(f"GPU budget insufficient: used/reserved={self.ledger['charged_gpu_hours']:.4f}, "
                                f"next={reserved:.4f}, cap={self.budget:.4f}")
        stage_dir = self.root / "runs" / run_id / "logs"
        stage_dir.mkdir(parents=True, exist_ok=True)
        attempt = {"number": len(attempts) + 1, "started_at": now()}
        attempts.append(attempt)
        record = {"status": "RUNNING", "task_hash": task_hash, "task": info,
                  "host": socket.gethostname(), "pid": None, "pid_token": None,
                  "attempts": attempts, "reserved_gpu_hours": reserved}
        self.ledger["stages"][key] = record
        self.ledger["charged_gpu_hours"] += reserved
        self.save()
        log = stage_dir / f"{stage}.{len(attempts):02d}.log"
        proc = None
        start = time.monotonic()
        last_beat = 0.0
        status, error = "FAILED", None
        old_handlers = {}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, interrupted)
            child_env = os.environ.copy()
            child_env.update(env)
            child_env["PYTHONUNBUFFERED"] = "1"
            with log.open("wb") as log_file:
                proc = subprocess.Popen(argv, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                        stdout=log_file, stderr=subprocess.STDOUT,
                                        start_new_session=True, close_fds=True)
                record["pid"] = proc.pid
                record["pid_token"] = pid_token(proc.pid)
                self.save()
                while proc.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed > timeout_s:
                        status = "TIMED_OUT"
                        raise TimeoutError(f"Stage exceeded {timeout_s}s")
                    if elapsed - last_beat >= 5:
                        record["heartbeat"] = now()
                        record["elapsed_s"] = elapsed
                        self.save()
                        last_beat = elapsed
                    time.sleep(0.1)
                if proc.returncode != 0:
                    raise WorkflowError(f"Exit code {proc.returncode}; see {log}")
            missing = [str(p) for p in outputs if not Path(p).is_file() or Path(p).stat().st_size == 0]
            if missing:
                raise WorkflowError(f"Expected artifacts missing/empty: {missing}")
            record["output_hashes"] = {str(Path(p).resolve()): file_hash(p) for p in outputs}
            status = "SUCCEEDED"
        except KeyboardInterrupt as e:
            status, error = "INTERRUPTED", str(e)
        except Exception as e:
            error = str(e)
        finally:
            if proc is not None:
                stop_group(proc)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            elapsed = time.monotonic() - start
            charge = elapsed * gpu_count / 3600
            self.ledger["charged_gpu_hours"] = max(0.0, self.ledger["charged_gpu_hours"] - reserved + charge)
            record["status"] = status
            attempt.update({"finished_at": now(), "elapsed_s": elapsed,
                            "gpu_hours": charge, "status": status, "log": str(log), "error": error})
            self.save()
        if status != "SUCCEEDED":
            raise WorkflowError(f"{key}: {status}: {error}")
        return record
