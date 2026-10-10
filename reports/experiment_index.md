# Experiment index

| Evidence | Scope | Primary artifact | Status |
|---|---|---|---|
| Bootstrap integration | First public scene, 200-step COLMAP/VGGT acceptance | [`formal_stage_v1.md`](formal_stage_v1.md) | Complete; integration only |
| Development baselines | bonsai/room/garden, nested 6/12/24 views, initialization reliability and 7k quality | [`mipnerf360_development_v1.md`](mipnerf360_development_v1.md) | Complete |
| Frozen final confirmation | counter/kitchen/bicycle, untouched fixed 24-view protocol | [`mipnerf360_final_v1.md`](mipnerf360_final_v1.md) | Complete and sealed |
| Official-branch configuration audit | Effective BA/no-BA branch fields and confidence-gate evidence | [`vggt_effective_configuration_correction_v2.json`](vggt_effective_configuration_correction_v2.json) | Complete |
| Duplicate-point root cause | Exact first non-finite operation and conventional scale floor | [`diagnostics/splat_initialization_nonfinite_v1.json`](diagnostics/splat_initialization_nonfinite_v1.json) | Complete |
| Matched BA operation | Same-reconstruction pre/post joint camera-and-point BA estimate | [`ba_operation_ablation_floor_s42_v1/results.md`](ba_operation_ablation_floor_s42_v1/results.md) | Complete on two development snapshots; seed-43 direction check also retained |
| Controlled pose sensitivity | Fixed post-BA snapshots, rotation/translation perturbation on bonsai/room | [`pose_sensitivity_v1/analysis.md`](pose_sensitivity_v1/analysis.md) | Complete; one training/noise seed, development only |
| Performance | All-heldout-camera render timing and quality/cost control | [`../docs/performance.md`](../docs/performance.md) | Complete at stated checkpoints |
| Resume qualification | State/order/numerical first-divergence diagnosis | [`../docs/resume_diagnosis.md`](../docs/resume_diagnosis.md) | Diagnosed, not production-qualified |
| Private viewer/demo | Browser interaction and held-out comparison video/panels | [`../docs/demo.md`](../docs/demo.md) | Validated privately; not public deployment |

Historical failures and superseded identities remain in their original paths and denominators. This index links corrected interpretations; it does not overwrite old receipts.
