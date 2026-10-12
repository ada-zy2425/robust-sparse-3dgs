# Tools by purpose

The application entry points are `python -m bgs` and `python -m bgs.queue`. These scripts support setup, experiments and analysis.

| Folder | Purpose | Main tools |
|---|---|---|
| `setup/` | Download data and pin dependencies | `setup_upstream.py`, `validate_processed_scene.py` |
| `experiments/` | Build immutable tasks and freeze protocols | `build_ba_operation_ablation.py`, `build_pose_sensitivity.py`, `freeze_final_protocol.py` |
| `diagnostics/` | Locate failures and validate controls | `diagnose_splat_initialization.py`, `probe_pose_perturbation.py`, `qualify_resume.py` |
| `evaluation/` | Collect results and measure performance | `collect_ba_operation_ablation.py`, `collect_pose_sensitivity.py`, `benchmark_render.py` |
| `presentation/` | Regenerate public figures | `build_presentation_figures.py`, `build_quality_cost_chart.py` |

Use the exact commands in [reproduction](../docs/reproduction.md). Historical task receipts retain their original paths and hashes; regenerate tasks for this checkout rather than replaying old command paths.
