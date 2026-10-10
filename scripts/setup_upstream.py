#!/usr/bin/env python3
"""Fetch pinned upstream checkouts. Does not install or change system packages."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',default='third_party')
    args=parser.parse_args()
    pins=json.loads((Path(__file__).resolve().parents[1]/'upstream.lock.json').read_text())
    base=Path(args.destination).resolve();base.mkdir(parents=True,exist_ok=True)
    for name,info in pins.items():
        target=base/name
        if not target.exists():
            subprocess.run(['git','clone','--no-checkout',info['url'],str(target)],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--detach',info['commit']],check=True)
            subprocess.run(['git','-C',str(target),'submodule','update','--init','--recursive'],check=True)
        actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if actual!=info['commit']:
            raise SystemExit(f'{target}: unexpected existing checkout; preserve it and choose another destination')
        print(f'{name}: {actual} at {target}')
    print('Next: use dedicated environments, verify CUDA/PyTorch compatibility, then install each upstream project and its example dependencies.')


if __name__=='__main__': main()
