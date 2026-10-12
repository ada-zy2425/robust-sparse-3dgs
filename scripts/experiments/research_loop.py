#!/usr/bin/env python3
"""Finite Codex CLI turns. GPU jobs remain owned/budgeted by the server runner.
Run only after the first pipeline succeeds and deployment paths are configured."""
import argparse
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--round-timeout',type=int,default=1800);p.add_argument('--task',default='ROUND_TASK.txt')
    a=p.parse_args()
    if not 1<=a.rounds<=5 or not 60<=a.round_timeout<=3600:raise SystemExit('Use 1–5 rounds, 60–3600 seconds each')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root))
    from bgs.core import lock,stop_group,atomic_json
    task=(root/a.task).resolve()
    if not task.is_file():raise SystemExit('Task file missing')
    folder=root/'agent_runs'/time.strftime('%Y%m%d_%H%M%S');folder.mkdir(parents=True)
    with lock(root/'agent_runs/.loop.lock'):
        for i in range(a.rounds):
            decision=folder/f'round_{i+1:02d}.json'
            log=folder/f'round_{i+1:02d}.log'
            prompt=task.read_text()+f'\n本次为有限循环第 {i+1}/{a.rounds} 轮。最终按 schema 输出。'
            with log.open('w') as f:
                child=subprocess.Popen(['codex','exec','--cd',str(root),'--sandbox','workspace-write',
                    '--output-schema',str(root/'decision.schema.json'),'-o',str(decision),'-'],
                    stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT,text=True,start_new_session=True)
                try:
                    child.communicate(prompt,timeout=a.round_timeout)
                except BaseException:
                    stop_group(child)
                    raise
                if child.returncode!=0:raise SystemExit(f'Codex failed; inspect {log}; no automatic retry')
            result=json.loads(decision.read_text())
            if result.get('action') not in ('CONTINUE','NEEDS_INPUT','COMPLETE'):
                raise SystemExit('Invalid decision; no next round')
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
            if result['action']!='CONTINUE':break
    print('Finite agent loop ended. Use server bgs status to inspect any active training job.')


if __name__=='__main__':main()
