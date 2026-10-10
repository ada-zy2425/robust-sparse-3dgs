# Evidence-backed resume bullets

- Built and GPU-validated a pinned COLMAP/VGGT-to-gsplat reconstruction harness with train/hold-out isolation, immutable receipts, bounded multi-GPU execution, PLY export, all-heldout-camera performance receipts and a browser-tested localhost viewer on NVIDIA A800 GPUs.
- Kept every failed/partial initialization in frozen three-development/three-final-scene evidence; traced an apparent step-100 pre-BA failure back to duplicate-point `log(0)` during initialization, added a versioned scale-equivariant guard, and obtained matched joint-BA quality estimates on two development snapshots without claiming separate camera/point effects or self-authored BA.
- Localized the first cross-densification resume difference after exact state/sample/forward replay to upstream rasterizer backward-scale numerical perturbations (`4.84e-8`–`6.92e-8`) and measured later grow/prune amplification; kept continuous runs as the formal boundary instead of relaxing the qualification tolerance or claiming a custom CUDA kernel.

## 中文版本

- 构建并在 NVIDIA A800 上实测 COLMAP/VGGT→gsplat 稀疏视图重建系统，完成训练/留出隔离、24 GPUh 多卡有界调度、PLY/HTML/私有 viewer 和全留出相机性能收据；六场景实验保留全部失败与部分注册比例。
- 将历史 step-100 非有限追溯到初始化重复点触发的 KNN `log(0)`，实现 dtype/场景尺度相关的常规正尺度保护，仅修复 bonsai/room 的 4/8 个退化点且不改变点/颜色；随后完成同快照联合 BA 操作配对，seed 42 获得 +1.606/+2.166 dB PSNR，明确不声称自研 BA 或拆分相机/点贡献。
- 预注册并完成两场景受控位姿敏感性矩阵（8/8 个新增 7k 运行一次成功，3.987 GPUh）：1° 旋转使 bonsai/room PSNR 分别下降 0.910/3.198 dB，1% 相机半径平移下降 0.207/1.372 dB；通过真实 Dataset 读回验证扰动生效，并将结论限定为单 seed/单方向集合。
