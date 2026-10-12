# Evidence index

For the visual story, read the [project report](../docs/project_report_zh.md). Start with the **three primary evidence groups** below.

| Priority | Question | Primary evidence |
|---|---|---|
| **1 — Root cause** | Why did initialization fail? | [Exact diagnostic](evidence/diagnostics/splat_initialization_nonfinite_v1.json) |
| **2 — Matched BA** | What changed in paired pre/post runs? | [Two-seed summary](evidence/ba_operation_ablation_floor_seed_aggregate_v1.json), [seed 42](evidence/ba_operation_ablation_floor_s42_v1/results.md), [seed 43](evidence/ba_operation_ablation_floor_s43_v3/results.md) |
| **3 — Pose sensitivity** | What happens when only training poses change? | [Analysis](evidence/pose_sensitivity_v1/analysis.md), [raw results](evidence/pose_sensitivity_v1/results.json) |

## Supporting validation

| Topic | Entry |
|---|---|
| Sparse-view baseline sweep | [Development matrix](evidence/mipnerf360_development_v1.md) |
| Sealed three-scene confirmation | [Final matrix](evidence/mipnerf360_final_v1.md) |
| Quality / cost and render timing | [Performance](../docs/reference/performance.md) |
| Clean checkout render reproduction | [Receipt](evidence/clean_reproduction_validation_floor_v2.json) |
| Resume limitation | [Diagnosis](../docs/reference/resume_diagnosis.md) |
| Historical branch corrections and failures | [Evidence status](../docs/reference/evidence_status.md) |

`evidence/` retains original raw JSON/CSV receipts, including failed and superseded experiments. JSON/CSV content, run identities, source hashes and historical private paths are preserved. Those paths describe the original execution; current commands are in [reproduction](../docs/reproduction.md).
