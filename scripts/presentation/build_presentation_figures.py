"""Regenerate presentation charts from committed evidence; no training required.
Requires matplotlib. Run from any working directory.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"docs/figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","figure.facecolor":"#f7f9fc","axes.facecolor":"#f7f9fc","axes.labelcolor":"#203047","text.color":"#203047"})
a=json.loads((ROOT/"reports/evidence/ba_operation_ablation_floor_seed_aggregate_v1.json").read_text())
fig,ax=plt.subplots(figsize=(10,4.2),layout="constrained")
for i,scene in enumerate(["bonsai","room"]):
 for j,d in enumerate(a["scenes"][scene]["paired_deltas"]):
  x=i+(j-.5)*.27
  ax.bar(x,d["psnr"],width=.24,color=["#167d9a","#7b61b5"][j],label=f"Training seed {d['seed']}" if i==0 else None)
  ax.text(x,d["psnr"]+.05,f"+{d['psnr']:.3f}",ha="center",weight="bold")
ax.set_xticks([0,1],["Bonsai","Room"]);ax.set_ylim(0,2.7);ax.set_ylabel("Held-out PSNR gain (dB)");ax.set_title("Matched BA operation | post minus pre",loc="left",weight="bold",pad=16);ax.legend(frameon=False,loc="upper left");ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
fig.get_layout_engine().set(rect=(0, .10, 1, .90))
fig.text(.5,.025,"Same initialization snapshots; shared numerical guard; camera + point optimization jointly.",ha="center",fontsize=9)
fig.savefig(OUT/"ba_gain.svg");plt.close(fig)
p=json.loads((ROOT/"reports/evidence/pose_sensitivity_v1/results.json").read_text())["delta_from_zero"]
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
for ax,conditions,title,unit in zip(axes,[["zero","r05","r10"],["zero","t005","t010"]],["Rotation perturbation","Translation perturbation"],["degrees","% of camera radius"]):
 for scene,color in [("bonsai","#167d9a"),("room","#7b61b5")]:
  vals=[p[f"{scene}/{c}"]["psnr"] for c in conditions]
  ax.plot([0,.5,1],vals,"o-",color=color,lw=2.5,label=scene.title())
  ax.annotate(f"{vals[-1]:.3f} dB",(1,vals[-1]),xytext=(-7,9),textcoords="offset points",ha="right",color=color,weight="bold")
 ax.set_title(title,loc="left",weight="bold");ax.set_xlabel(unit);ax.set_ylabel("PSNR change from zero (dB)");ax.set_ylim(-3.7,.4);ax.set_xticks([0,.5,1]);ax.grid(alpha=.15);ax.legend(frameon=False)
fig.get_layout_engine().set(rect=(0, .10, 1, .90))
fig.text(.5,.025,"Fixed points, intrinsics, RGB, alignment and held-out cameras. One training seed / one noise draw.",ha="center",fontsize=9)
fig.savefig(OUT/"pose_sensitivity.svg");plt.close(fig)
