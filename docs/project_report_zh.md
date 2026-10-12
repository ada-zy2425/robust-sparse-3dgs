# Robust Sparse 3DGS：稀疏视角重建与失效诊断

**用少量照片完成重建、留出渲染和交互查看，并分析初始化与位姿误差造成的失效。**

[首页](../README.md) · [复现](reproduction.md) · [技术细节](technical_notes.md) · [结果账本](../reports/README.md)

## 1. 初始化稳健性：定位零尺度，保留点云信息

历史运行在 step 100 报错，逐阶段回放却发现：异常在第一次 forward 之前就已出现。重复点簇使三个非自身近邻距离为零，初始尺度取 log 后得到负无穷。

![系统与初始化机制示意图](figures/initialization.svg)

我比较去重与正尺度下界两种修复。去重会改变点数与颜色；最终选择与 dtype / 场景尺度相适应的下界，对 BA 前后使用相同规则。bonsai / room 仅修正 **4 / 8 个尺度**，点数、颜色、覆盖保持不变，post-BA 不触发下界。四臂通过逐步有限值检查，seed 42 的四个 7k 运行完成。

实现：[splat_numerics.py](../bgs/splat_numerics.py)。证据：账本的 `initialization`。

## 2. 匹配 BA：把分支对比推进到操作对比

完整 VGGT BA 分支还改变跟踪和点构造，直接比较分支无法把收益归给 BA。我在同一次重建中保存 BA 前后不可变快照，用相同数值修复及下游配置进行配对训练。

![两场景、两训练 seed 的配对收益](figures/ba_gain.svg)

四个配对的留出 **PSNR 提升 1.591–2.166 dB**，SSIM / LPIPS 全部改善。两个训练 seed 复用同一初始化，因此验证的是训练随机性重复；BA 同时优化相机与点，收益仍归属于联合操作。

证据：账本的 `ba`、`ba_seed42`、`ba_seed43`。

## 3. 位姿敏感性：固定点云，只扰动训练相机

固定初始点、内参、RGB、对齐及留出相机；只添加预注册的旋转或平移误差。各强度共享噪声方向，平移按相机半径归一化。

![位姿扰动与留出质量变化](figures/pose_sensitivity.svg)

1° 旋转使 bonsai / room 的 PSNR 下降 **0.910 / 3.198 dB**；1% 半径平移下降 **0.207 / 1.372 dB**。这组控制显示 room 对旋转更敏感。两场景、一个训练 seed 和一个噪声方向集合，结论限定在该实验范围；没有三维真值精度结论。

证据：[冻结协议](../protocols/pose_sensitivity_v1.json)与账本的 `pose_sensitivity`。

## 系统交付与成本

实现 train-only 数据隔离、源代码锁定、预算调度、真实留出评估、PLY 导出与本地 WebGL 交互验证。South Building 单场景 3k / 7k 对照减少训练时间 **60.8%**、PLY 大小 **65.6%**，但 LPIPS 从 0.3516 变为 0.4633，属于质量与成本取舍。

真实留出对照视频已私下验证，公开仓库只放机制图和数据图。当前跨 densification 的生产级 resume 尚未通过验收。完整归属、参数、可视化范围和限制集中在 [技术说明](technical_notes.md)，历史记录统一在 [归档](../reports/archive.zip)。
