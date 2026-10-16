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
                if 'file_path' in source:
                    raw = (root / source['file_path']).read_bytes()
                else:
                    raw = z.read(source['archive_path'])
                if hashlib.sha256(raw).hexdigest() != source['sha256']:
                    location = source.get('file_path', source.get('archive_path'))
                    raise ValueError(f'Ledger differs from evidence source: {location}')
                if 'data' in source and json.loads(raw) != source['data']:
                    location = source.get('file_path', source.get('archive_path'))
                    raise ValueError(f'Ledger data differs from evidence source: {location}')
            for path, expected in record.get('artifacts', {}).items():
                if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Published artifact integrity mismatch: {path}')
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
    v9_record = ledger['records']['rotation_transfer_v9']
    v9 = v9_record.get('data') or json.loads((root / v9_record['file_path']).read_text())
    seeds = ['2718','3141']
    colors = ['#167d9a','#7b61b5']
    fig, axes = plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    recovery = [v9['per_seed'][seed]['pose']['injected_rotation_recovery_percent'] for seed in seeds]
    relative = [100*v9['per_seed'][seed]['pose']['all_pairs_relative_rotation_fractional_improvement'] for seed in seeds]
    x = range(len(seeds)); width = .34
    axes[0].bar([i-width/2 for i in x],recovery,width,color=colors[0],label='Injected rotation')
    axes[0].bar([i+width/2 for i in x],relative,width,color=colors[1],label='All-pairs relative')
    axes[0].set_xticks(list(x),seeds); axes[0].set_ylim(0,105); axes[0].set_ylabel('Improvement (%)')
    axes[0].set_title('Pose signal',loc='left',weight='bold'); axes[0].legend(frameon=False,fontsize=8)
    psnr = [v9['per_seed'][seed]['C_minus_A_mean']['psnr'] for seed in seeds]
    axes[1].bar(seeds,psnr,color=colors); axes[1].axhline(.5,color='#d06b42',ls='--',lw=1,label='frozen gate')
    axes[1].set_ylim(0,3); axes[1].set_ylabel('C − A PSNR (dB)'); axes[1].set_title('Held-out quality',loc='left',weight='bold')
    axes[1].legend(frameon=False,fontsize=8)
    ssim = [v9['per_seed'][seed]['C_minus_A_mean']['ssim'] for seed in seeds]
    lpips = [-v9['per_seed'][seed]['C_minus_A_mean']['lpips'] for seed in seeds]
    axes[2].bar([i-width/2 for i in x],ssim,width,color=colors[0],label='SSIM gain')
    axes[2].bar([i+width/2 for i in x],lpips,width,color=colors[1],label='LPIPS reduction')
    axes[2].set_xticks(list(x),seeds); axes[2].set_ylim(0,.27); axes[2].set_title('Perceptual metrics',loc='left',weight='bold')
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set_xlabel('Perturbation seed'); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
        for container in ax.containers:
            ax.bar_label(container,fmt='%.2f',padding=3,fontsize=8,weight='bold')
    fig.get_layout_engine().set(rect=(0,.12,1,.91))
    fig.text(.5,.025,'Garden historical-development scene · 24 train / 24 held-out · 7k · training seed 42 · standard BA owns step-0 pose recovery.',ha='center',fontsize=9)
    fig.savefig(out/'rotation_transfer_v9.svg',metadata={'Date':None}); plt.close(fig)
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
