# Results and evidence

Start with the [visual project report](../docs/project_report_zh.md). This directory has **one current ledger and one historical archive**.

| Study | Key in [`results.json`](results.json) |
|---|---|
| Duplicate-point root cause and scale guard | `initialization` |
| Matched BA operation | `ba`, `ba_seed42`, `ba_seed43` |
| Controlled training-pose perturbation | `pose_sensitivity` |
| Sealed three-scene confirmation | `final_confirmation` |
| Render distributions and quality/cost control | `performance` |
| Resume boundary and clean render reproduction | `resume`, `reproduction` |

Records retain full primary data, original SHA-256 and source paths. `performance.records` holds individual benchmark receipts.

[`archive.zip`](archive.zip) contains exact pre-consolidation JSON/CSV, historical reports, detailed notes and merged script sources. `MANIFEST.json` records every member's bytes and SHA-256; failed and superseded runs remain available. No private RGB, model weights or PLY are included.

```bash
python scripts/report.py --verify
python scripts/report.py --figures
```

Verification checks archive integrity, member hashes, ledger/source equality and archive SHA-256. Figure regeneration requires matplotlib and does not run training.
