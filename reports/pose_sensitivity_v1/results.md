# Controlled pose sensitivity (development only)

This protocol adds controlled error to existing estimated **training** poses. It is not known absolute pose error, geometry accuracy, or a general robustness proof. Alignment was fixed before perturbation; initial points, intrinsics, RGB and held-out cameras were unchanged.

| Scene | Condition | Status | PSNR (delta) | SSIM (delta) | LPIPS (delta) | GS | train s | peak GiB | queue GPUh |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| bonsai | zero | SUCCEEDED | 17.3846 (+0.0000) | 0.5018 (+0.0000) | 0.3512 (+0.0000) | 738354 | 1741.8 | 5.62 | 0.4985 |
| bonsai | r05 | SUCCEEDED | 17.0726 (-0.3120) | 0.4898 (-0.0120) | 0.4118 (+0.0605) | 814686 | 1805.6 | 6.07 | 0.5166 |
| bonsai | r10 | SUCCEEDED | 16.4750 (-0.9095) | 0.4693 (-0.0324) | 0.4667 (+0.1155) | 786091 | 1791.0 | 5.81 | 0.5126 |
| bonsai | t005 | SUCCEEDED | 17.3039 (-0.0807) | 0.4980 (-0.0038) | 0.3756 (+0.0244) | 784420 | 1766.2 | 5.94 | 0.5057 |
| bonsai | t010 | SUCCEEDED | 17.1780 (-0.2066) | 0.4914 (-0.0103) | 0.4015 (+0.0503) | 800094 | 1744.7 | 6.03 | 0.4993 |
| room | zero | SUCCEEDED | 19.7719 (+0.0000) | 0.6735 (+0.0000) | 0.2688 (+0.0000) | 547025 | 1689.7 | 3.60 | 0.4846 |
| room | r05 | SUCCEEDED | 17.7381 (-2.0338) | 0.5960 (-0.0776) | 0.3846 (+0.1159) | 616602 | 1693.8 | 3.90 | 0.4860 |
| room | r10 | SUCCEEDED | 16.5736 (-3.1982) | 0.5595 (-0.1140) | 0.4524 (+0.1836) | 562262 | 1701.7 | 3.53 | 0.4882 |
| room | t005 | SUCCEEDED | 19.4937 (-0.2782) | 0.6568 (-0.0167) | 0.3256 (+0.0568) | 651303 | 1717.7 | 4.17 | 0.4926 |
| room | t010 | SUCCEEDED | 18.3994 (-1.3725) | 0.6137 (-0.0598) | 0.3676 (+0.0988) | 599465 | 1694.4 | 3.80 | 0.4861 |

New matrix cost: **3.987086 GPUh**; total ledger: **20.604399/24 GPUh**.

Curves: [rotation](rotation_sensitivity.svg) · [translation](translation_sensitivity.svg). Fixed-view private panel paths are recorded in `results.json`.

One training seed and one noise-direction set are shown. No monotonicity or cross-scene generalization is assumed.
