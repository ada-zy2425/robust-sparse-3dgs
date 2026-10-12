# Industrial delivery v1

Scope: a private, single-node A800 delivery for static sparse-view 3DGS. This version freezes the evidence available through `pose_sensitivity_v1`; it does not reopen or modify `final_v1`.

Included:

- pinned COLMAP/VGGT-to-gsplat workflow with train/holdout isolation, bounded multi-GPU execution and immutable receipts;
- six-scene development/final reliability evidence with partial-registration ratios retained;
- conventional duplicate-point scale guard and corrected failure attribution;
- two-scene same-reconstruction joint BA-operation comparison;
- two-scene controlled pose-sensitivity matrix, curves and private fixed-view panels;
- PLY/report output, measured rendering and a browser-validated localhost viewer;
- verified reproduction commands and evidence/claim index.

Measured pose-sensitivity addition: 8/8 new 7k runs succeeded once for 3.987086 GPUh. Rotation 0.5°/1.0° changed PSNR by -0.312/-0.910 dB on bonsai and -2.034/-3.198 dB on room; translation 0.5%/1.0% of camera radius changed it by -0.081/-0.207 and -0.278/-1.372 dB. These are development-scene sensitivities for one training seed and one noise-direction set.

Explicit exclusions: public hosting, public RGB/PLY/checkpoint redistribution, commercial VGGT rights, geometry accuracy, SOTA claims, custom CUDA, self-authored BA, production-qualified resume and an independently new robustness algorithm.

Primary entry points: [`../industrial_delivery_zh.md`](../reference/industrial_delivery_zh.md), [`../../reports/README.md`](../../reports/README.md), [`../../reports/evidence/pose_sensitivity_v1/analysis.md`](../../reports/evidence/pose_sensitivity_v1/analysis.md), [`../reproduction.md`](../reproduction.md), and [`../demo.md`](../reference/demo.md).
