# Experimental protocol

- One scene, method, split, seed and predeclared total schedule define one immutable run ID.
- Formal candidate schedule: 7,000 total steps with diagnostics at 1k, 3k and 7k; no changing `max_steps` midway.
- Comparisons fix train views, held-out views, GPU model, resolution, seed and renderer.
- Held-out RGB is excluded from initialization, BA, confidence filtering and parameter selection.
- Calibration from full-capture SfM is reported as calibration metadata, not ground-truth geometry. Sim(3) uses training cameras only.
- Registration failures remain in the success-rate denominator. Quality metrics report the valid-scene denominator explicitly.
- A changed parameter, source adapter or data split creates a new identity. A failed task may be retried unchanged at most once and only with a recorded reason.
- Final scenes are evaluated after freezing the method; they do not feed back into tuning.

## Locked 7k trainer schedule

The exact pinned gsplat `DefaultStrategy` starts refinement only after step 500, refines every 100 steps, resets opacity every 3,000 steps and would stop refinement at 15,000 steps (therefore refinement remains active through this 7k schedule). Its fixed thresholds are opacity prune 0.005, image-gradient growth 0.0002, 3D growth scale 0.01, 2D growth scale 0.05 and 3D prune scale 0.1. Batch size is one, so upstream step rescaling leaves these values unchanged.

Initial Adam learning rates are means `1.6e-4 × scene_scale`, scales `5e-3`, opacities `5e-2`, quaternions `1e-3`, SH0 `2.5e-3` and higher-order SH `1.25e-4`. Only the means rate follows the upstream exponential schedule to 1% of its initial value at the predeclared final step. SH degree is three with a 1,000-step interval; pose optimization and opacity/scale regularizers are disabled.

This explains why the 200-step, 208-point integration smoke remained at 208 Gaussians: it ended before `refine_start_iter=500`. Formal runs showed actual structural changes and checkpointed Gaussian counts; for example, the successful garden COLMAP development run ended at 1,389,863 Gaussians. Low smoke quality is not attributed to a broken densifier.
