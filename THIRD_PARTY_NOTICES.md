# Third-party notices

This repository does not vendor the following projects or model weights. Exact source revisions are recorded in `upstream.lock.json`.

- **gsplat**, nerfstudio-project — Apache-2.0 upstream source; preserve its notices and license when obtaining it.
- **COLMAP** — BSD-3-Clause upstream software.
- **VGGT code**, Meta — use under the license shipped with the pinned repository.
- **VGGT-1B original checkpoint** — research use here; upstream explicitly states that the original checkpoint is non-commercial. It is not included in this repository. Commercial use requires the separately gated commercial checkpoint and its terms.
- **LightGlue** — Apache-2.0 upstream dependency; the deployed revision is pinned separately in the runtime evidence.
- **ALIKED n16 checkpoint** — BSD-3-Clause as declared by the upstream ALIKED repository; loaded by the LightGlue extractor in the VGGT BA diagnostic and not included here.
- **SuperPoint v1 checkpoint** — loaded through the pinned LightGlue extractor for the VGGT BA diagnostic and not included here; retain the upstream LightGlue/SuperPoint notices and terms.
- **VGGSfM v2 tracker checkpoint** — CC-BY-NC-4.0 as declared by the upstream model repository; used only for the official VGGT bundle-adjustment diagnostic and not included here.
- **DINOv2 code and ViT-B/14-register checkpoint** — Apache-2.0 code with upstream model terms; used for the VGGT BA query-frame ranking path and not included here.

Dataset rights are independent of code/model rights. South Building assets and derived visual material remain private until redistribution permission is confirmed.
