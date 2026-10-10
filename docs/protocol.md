# Experimental protocol

- One scene, method, split, seed and predeclared total schedule define one immutable run ID.
- Formal candidate schedule: 7,000 total steps with diagnostics at 1k, 3k and 7k; no changing `max_steps` midway.
- Comparisons fix train views, held-out views, GPU model, resolution, seed and renderer.
- Held-out RGB is excluded from initialization, BA, confidence filtering and parameter selection.
- Calibration from full-capture SfM is reported as calibration metadata, not ground-truth geometry. Sim(3) uses training cameras only.
- Registration failures remain in the success-rate denominator. Quality metrics report the valid-scene denominator explicitly.
- A changed parameter, source adapter or data split creates a new identity. A failed task may be retried unchanged at most once and only with a recorded reason.
- Final scenes are evaluated after freezing the method; they do not feed back into tuning.
