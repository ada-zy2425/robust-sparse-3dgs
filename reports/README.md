# Results and evidence

Start with the [visual project report](../docs/project_report_zh.md). This directory has **one current ledger and one historical archive**.

| Study | Key in [`results.json`](results.json) |
|---|---|
| Duplicate-point root cause and scale guard | `initialization` |
| Matched BA operation | `ba`, `ba_seed42`, `ba_seed43` |
| Controlled training-pose perturbation | `pose_sensitivity` |
| Rotation-only BA transfer and Garden confirmation | `rotation_transfer_v9` |
| Sealed three-scene confirmation | `final_confirmation` |
| Render distributions and quality/cost control | `performance` |
| Resume boundary and clean render reproduction | `resume`, `reproduction` |

Records retain primary data, original SHA-256 and source paths. `performance.records` holds individual benchmark receipts. The v9 entry is backed by the compact checked-in [`rotation_transfer_v9.json`](rotation_transfer_v9.json): both seed-level endpoints, all 48 held-out A/C rows, contracts, costs, negative history and the DSC08108 CPU audit are present without raw runtime directories.

[`archive.zip`](archive.zip) contains exact pre-v9 consolidated JSON/CSV, historical reports, detailed notes and merged script sources. `MANIFEST.json` records every member's bytes and SHA-256; failed and superseded runs remain available. The archive remains byte-identical, while the v9 file and its two hashed figures are verified as direct evidence sources. No dataset RGB, model weights, checkpoints, PLY or process logs are included.

```bash
python scripts/report.py --verify
python scripts/report.py --figures
```

Verification checks archive integrity, member hashes, ledger/source equality and archive SHA-256. Figure regeneration requires matplotlib and does not run training.
