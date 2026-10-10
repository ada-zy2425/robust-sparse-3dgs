# Interview notes

## Three-minute version

I built a reproducible sparse-view 3D Gaussian Splatting workflow for static industrial-style captures. The hard part was not invoking a trainer; it was preventing convincing but invalid results. I isolated held-out images, pinned upstream sources, added immutable receipts and an aggregate multi-GPU budget, and evaluated novel views after training-camera-only coordinate alignment.

Two technical decisions mattered. First, initialization receives a copied train-only directory because a COLMAP version canonicalized symlinks and broke the image-name/data-isolation contract. Second, I test resume against the same persisted prefix across a real densification boundary, including optimizer, strategy, RNG and held-out renders—not just whether a checkpoint loads.

The most useful failure was sparse COLMAP registration. Wide trajectory sampling registered only a fraction of views; shortening the calibrated trajectory and increasing overlap produced a 24/24 arm while the nested 12-view arm still failed. That converted a vague “3DGS quality problem” into a measurable capture/initialization reliability problem.

The first formal comparison produced a second concrete decision. A 100k-point VGGT initialization was worse than a 9.8k-point COLMAP initialization because the predicted cameras had much larger alignment residuals; this ruled out “more points should be better” and redirected the next control toward camera refinement. Official VGGT bundle adjustment then cut the mean center residual from 0.0499 to 0.00475 and improved all three held-out metrics versus no-BA, supporting the diagnosis without pretending BA is my algorithm. Separately, a 3k schedule cut optimization time by 60.8% and PLY size by 65.6% while improving PSNR/SSIM, but LPIPS regressed, so I present it as a product-facing quality/cost knob rather than a blanket win.

## Claims discipline

Say what is measured: task isolation, failure accounting, held-out metrics, artifact export, a real interactive viewer check and bounded execution. Do not claim robust resume, commercial VGGT rights, geometry accuracy, cross-scene generalization or a new algorithm until the corresponding evidence gate passes.
