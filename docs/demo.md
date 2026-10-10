# Private demo validation

The interactive viewer was validated with the completed 24-view room official-BA checkpoint (546,566 Gaussians). It listened on localhost only. A headless WebGL browser displayed a recognizable calibrated room view and the documented controls; after a 16-pixel orbit drag the canvas remained non-black and recognizable and its pixel hash changed. The viewer, browser, SSH tunnel and debug ports were then stopped.

The first deliberately larger 150-pixel drag produced a black canvas. That predecessor is retained in private evidence rather than omitted; the smaller deterministic interaction was used to demonstrate a usable local orbit, not to claim an unrestricted navigation envelope.

A separate 39.9667-second, 1280×720, 30 FPS H.264 video presents three held-out comparisons for each development scene: bonsai (object-centric), room (indoor) and garden (outdoor). Every segment labels the columns as reference, prediction and absolute error and states that it is held-out evaluation, not a novel-view trajectory. The private artifact SHA256 is `5ea9158825741e6aa71e888e4dcd3f0ec34fc0c1201c41278f113d78b019df84`.

The video and source panels are intentionally excluded from Git because redistribution terms for derived benchmark imagery were not resolved. They are presentation evidence, not training inputs and not a public deployment.

After the final matrix completed, one private held-out canvas from each successful final scene was also inspected. Counter, kitchen and bicycle predictions were recognizable and aligned; errors concentrated around reflective metal/fine texture, foliage/object edges and bicycle foreground grass respectively. The latter contains visible blur/floater artifacts. These final canvases remain private and are not substituted for the separate browser interaction test.
