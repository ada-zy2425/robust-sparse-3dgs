# Private demo validation

The interactive viewer was validated with the completed 24-view room official-BA checkpoint (546,566 Gaussians). It listened on localhost only. A headless WebGL browser displayed a recognizable calibrated room view and the documented controls; after a 16-pixel orbit drag the canvas remained non-black and recognizable and its pixel hash changed. The viewer, browser, SSH tunnel and debug ports were then stopped.

The first deliberately larger 150-pixel drag produced a black canvas. That predecessor is retained in private evidence rather than omitted; the smaller deterministic interaction was used to demonstrate a usable local orbit, not to claim an unrestricted navigation envelope.

A separate 39.9667-second, 1280×720, 30 FPS H.264 video presents three held-out comparisons for each development scene: bonsai (object-centric), room (indoor) and garden (outdoor). Every segment labels the columns as reference, prediction and absolute error and states that it is held-out evaluation, not a novel-view trajectory. The private artifact SHA256 is `5ea9158825741e6aa71e888e4dcd3f0ec34fc0c1201c41278f113d78b019df84`.

The video and source panels are intentionally excluded from Git because redistribution terms for derived benchmark imagery were not resolved. They are presentation evidence, not training inputs and not a public deployment.

After the final matrix completed, one private held-out canvas from each successful final scene was also inspected. Counter, kitchen and bicycle predictions were recognizable and aligned; errors concentrated around reflective metal/fine texture, foliage/object edges and bicycle foreground grass respectively. The latter contains visible blur/floater artifacts. These final canvases remain private and are not substituted for the separate browser interaction test.

The industrial-delivery pose demo uses preregistered holdouts `DSCF5565.JPG` (bonsai) and `DSCF4667.JPG` (room). For each scene, five private canvases show zero, rotation 0.5°/1.0° and translation 0.5%/1.0% conditions in reference | prediction | absolute-error layout. Their immutable relative paths and metric hashes are listed in `reports/pose_sensitivity_v1/results.json`; the committed rotation/translation SVGs contain no benchmark RGB.

Visual review found progressively stronger double contours, texture smearing and thin-structure loss. Bonsai errors concentrate on the flower branches, bicycle spokes/tubes and tablecloth; room errors concentrate on the chair/table/TV boundaries, keyboard, glassware and cables. The zero panels already contain disocclusion and poorly observed boundary errors, so those are not attributed to perturbation. The private panels remain excluded from Git.

The reusable interactive entry remains the validated room zero-perturbation checkpoint:

```bash
"$GSPLAT_PY" -m bgs view \
  --config "$PROJECT_ROOT/configs/ba_operation_ablation_floor_v1/m360_room_dev_v24_vggt_ba_post_s42_7k_floor_v1.json"
```

Keep the viewer on localhost, use an SSH local forward when remote, and stop the viewer/tunnel after inspection.
