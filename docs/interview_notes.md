# Interview notes

## Three-minute version

I built a reproducible sparse-view 3D Gaussian Splatting workflow for static industrial-style captures. The hard part was not invoking a trainer; it was preventing convincing but invalid results. I isolated held-out images, pinned upstream sources, added immutable receipts and an aggregate multi-GPU budget, and evaluated novel views after training-camera-only coordinate alignment.

Two technical decisions mattered. First, initialization receives a copied train-only directory because a COLMAP version canonicalized symlinks and broke the image-name/data-isolation contract. Second, I test resume against the same persisted prefix across a real densification boundary, including optimizer, strategy, RNG and held-out renders—not just whether a checkpoint loads.

The most useful failure was sparse COLMAP registration. A frozen three-scene, 6/12/24-view matrix produced full coverage in only 1/9 COLMAP arms; the unsuccessful partial reconstructions stayed in the denominator. That converted a vague “3DGS quality problem” into a measurable capture/initialization reliability problem.

The first formal comparison produced a second concrete decision. A 100k-point VGGT initialization was worse than a 9.8k-point COLMAP initialization because the predicted cameras had much larger alignment residuals; this ruled out “more points should be better” and redirected the next control toward camera refinement. Official VGGT bundle adjustment then gave full-coverage 24-view initializations on all three development scenes and improved all held-out metrics on two. The outdoor scene improved reliability and LPIPS but lost PSNR/SSIM and became larger/slower, so I report a scoped diagnosis rather than pretending BA is my algorithm or a universal win.

I froze three unseen final scenes before scoring. Official BA completed all three at 24/24 coverage; the paired COLMAP runs reached only 20/24, 21/24 and 4/24 and were kept as failures. That is useful external confirmation of the reliability policy, but it is not a conditional quality comparison because failed baselines have no valid score. The hardest bicycle result also shows the remaining fidelity gap: the object is recognizable, while foreground grass contains strong blur/floater artifacts.

For performance, I benchmarked every fixed held-out camera with CUDA synchronization, five warmups and 30 repeats per camera. Across eight development checkpoints, the results made Gaussian count and scene-dependent model growth visible instead of quoting a cherry-picked camera. A separate 3k schedule cut optimization time by 60.8% and PLY size by 65.6% versus 7k, but LPIPS regressed, so it remains a product-facing quality/cost knob.

## Claims discipline

Say what is measured: task isolation, complete failure accounting, three development plus three frozen final scenes, artifact export, multi-camera performance, a real interactive viewer check and bounded execution. Do not claim robust resume, commercial VGGT rights, geometry accuracy, strong statistical generalization, a novel BA method or custom CUDA work.
