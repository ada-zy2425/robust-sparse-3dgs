#!/usr/bin/env python3
"""Verify published evidence and regenerate measured presentation figures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> dict:
    ledger = json.loads((root / 'reports/results.json').read_text())
    archive = root / 'reports' / ledger['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ledger['archive_sha256']:
        raise ValueError('Archive SHA-256 differs from the current ledger')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        manifest = json.loads(z.read('MANIFEST.json'))
        if manifest['source_commit'] != ledger['source_commit']:
            raise ValueError('Archive and ledger source commits differ')
        if set(names) != set(manifest['files']) | {'MANIFEST.json'}:
            raise ValueError('Archive manifest does not cover exactly every member')
        if z.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for path, expected in manifest['files'].items():
            raw = z.read(path)
            if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise ValueError(f'Archive integrity mismatch: {path}')
        records = list(ledger['records'].values())
        for record in records:
            sources = list(record['records'].values()) if 'records' in record else [record]
            for source in sources:
                raw = z.read(source['archive_path'])
                if hashlib.sha256(raw).hexdigest() != source['sha256'] or json.loads(raw) != source['data']:
                    raise ValueError(f'Ledger differs from archived source: {source["archive_path"]}')
    return ledger


def build_figures(ledger: dict, root: Path = ROOT) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root / 'docs/figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
        'svg.hashsalt':'robust-sparse-3dgs','figure.facecolor':'#f7f9fc',
        'axes.facecolor':'#f7f9fc','axes.labelcolor':'#203047','text.color':'#203047'})
    a = ledger['records']['ba']['data']
    fig, ax = plt.subplots(figsize=(10,4.2), layout='constrained')
    for i, scene in enumerate(['bonsai','room']):
        for j, d in enumerate(a['scenes'][scene]['paired_deltas']):
            x = i+(j-.5)*.27
            ax.bar(x,d['psnr'],width=.24,color=['#167d9a','#7b61b5'][j],
                   label=f'Training seed {d["seed"]}' if i == 0 else None)
            ax.text(x,d['psnr']+.05,f'+{d["psnr"]:.3f}',ha='center',weight='bold')
    ax.set_xticks([0,1],['Bonsai','Room']); ax.set_ylim(0,2.7)
    ax.set_ylabel('Held-out PSNR gain (dB)')
    ax.set_title('Matched BA operation | post minus pre',loc='left',weight='bold',pad=16)
    ax.legend(frameon=False,loc='upper left'); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.get_layout_engine().set(rect=(0,.10,1,.90))
    fig.text(.5,.025,'Same initialization snapshots; shared numerical guard; camera + point optimization jointly.',ha='center',fontsize=9)
    fig.savefig(out/'ba_gain.svg',metadata={'Date':None}); plt.close(fig)
    p = ledger['records']['pose_sensitivity']['data']['delta_from_zero']
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax, conditions, title, unit in zip(axes,[['zero','r05','r10'],['zero','t005','t010']],
            ['Rotation perturbation','Translation perturbation'],['degrees','% of camera radius']):
        for scene, color in [('bonsai','#167d9a'),('room','#7b61b5')]:
            vals = [p[f'{scene}/{c}']['psnr'] for c in conditions]
            ax.plot([0,.5,1],vals,'o-',color=color,lw=2.5,label=scene.title())
            ax.annotate(f'{vals[-1]:.3f} dB',(1,vals[-1]),xytext=(-7,9),
                        textcoords='offset points',ha='right',color=color,weight='bold')
        ax.set_title(title,loc='left',weight='bold'); ax.set_xlabel(unit)
        ax.set_ylabel('PSNR change from zero (dB)'); ax.set_ylim(-3.7,.4)
        ax.set_xticks([0,.5,1]); ax.grid(alpha=.15); ax.legend(frameon=False)
    fig.get_layout_engine().set(rect=(0,.10,1,.90))
    fig.text(.5,.025,'Fixed points, intrinsics, RGB, alignment and held-out cameras. One training seed / one noise draw.',ha='center',fontsize=9)
    fig.savefig(out/'pose_sensitivity.svg',metadata={'Date':None}); plt.close(fig)
    for path in out.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true',help='Check archive hashes and current ledger equality')
    parser.add_argument('--figures',action='store_true',help='Verify first, then regenerate SVG figures (requires matplotlib)')
    args = parser.parse_args(argv)
    if not (args.verify or args.figures):
        parser.print_help()
        return
    ledger = verify()
    print('Evidence archive and current ledger: verified')
    if args.figures:
        build_figures(ledger)
        print('Measured SVG figures: regenerated')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f'BLOCKED: {error}')
