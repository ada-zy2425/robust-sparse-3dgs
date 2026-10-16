from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/diagnostics/build_rotation_only_camera_override.py"
SPEC = importlib.util.spec_from_file_location("rotation_transfer_builder", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RotationTransferV9Publication(unittest.TestCase):
    def test_rotation_only_builder_preserves_centers(self):
        names = ["camera0.jpg", "camera1.jpg"]
        a = np.repeat(np.eye(4)[None], 2, axis=0)
        a[1, :3, 3] = [1.0, 2.0, 3.0]
        b = a.copy()
        angle = np.radians(0.5)
        b[1, :3, :3] = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        b[1, :3, 3] = [-4.0, 5.0, 6.0]
        c = BUILDER.build_rotation_only(names, a, names, b)
        self.assertTrue(np.array_equal(c[:, :3, :3], b[:, :3, :3]))
        self.assertTrue(np.array_equal(c[:, :3, 3], a[:, :3, 3]))
        self.assertTrue(np.array_equal(c[0], a[0]))

    def test_frozen_configs_match_runner_and_science_contract(self):
        runner = ROOT / "scripts/diagnostics/fixed_camera_rotation_transfer_v9.py"
        runner_hash = sha256(runner)
        expected = {
            "garden_p2718_A.json": (2718, "A_original_perturbed_fixed"),
            "garden_p2718_C_retry_authorized_02.json": (2718, "C_ba_rotation_input_center_fixed"),
            "garden_p3141_A.json": (3141, "A_original_perturbed_fixed"),
            "garden_p3141_C.json": (3141, "C_ba_rotation_input_center_fixed"),
        }
        for filename, (seed, arm) in expected.items():
            config = read(ROOT / "configs/rotation_transfer_v9" / filename)
            self.assertEqual(config["runner_sha256"], runner_hash)
            self.assertEqual(config["perturbation_seed"], seed)
            self.assertEqual(config["arm"], arm)
            spec = config["spec"]
            self.assertEqual(spec["seed"], 42)
            self.assertEqual(spec["steps"], 7000)
            self.assertEqual(spec["max_edge"], 1024)
            self.assertFalse(spec["pose_opt"])
            self.assertEqual(spec["densification_policy"], "default")
            self.assertEqual(spec["rotation_degrees"], 1.0)
            self.assertEqual(spec["translation_fraction"], 0.0)

    def test_machine_summary_retains_counterexample_and_history(self):
        result = read(ROOT / "reports/rotation_transfer_v9.json")
        self.assertEqual(result["decision"], "CROSS_SCENE_CONFIRMATION_PASS")
        self.assertFalse(result["history_preserved"]["v3"]["repair_accepted"])
        self.assertEqual(result["history_preserved"]["v7"]["status"], "FAIL")
        self.assertEqual(result["history_preserved"]["v8"]["status"], "PASS_DEVELOPMENT")
        self.assertEqual(result["per_seed"]["2718"]["heldout_improvement_counts"]["psnr"]["improved"], 24)
        self.assertEqual(result["per_seed"]["3141"]["heldout_improvement_counts"]["psnr"]["improved"], 23)
        for seed in ("2718", "3141"):
            self.assertEqual(result["per_seed"][seed]["heldout_improvement_counts"]["ssim"]["improved"], 24)
            self.assertEqual(result["per_seed"][seed]["heldout_improvement_counts"]["lpips"]["improved"], 24)
        row = next(row for row in result["per_seed"]["3141"]["per_image"] if row["name"] == "DSC08108.JPG")
        self.assertLess(row["C_minus_A"]["psnr"], 0.0)
        self.assertGreater(row["C_minus_A"]["ssim"], 0.0)
        self.assertLess(row["C_minus_A"]["lpips"], 0.0)

    def test_protocol_and_public_artifacts_are_explicit(self):
        protocol = read(ROOT / "protocols/rotation_transfer_cross_scene_confirmation_v9.json")
        self.assertEqual(protocol["scene_and_seeds"]["perturbation_seeds"], [2718, 3141])
        self.assertEqual(protocol["scene_and_seeds"]["training_seed"], 42)
        self.assertEqual(protocol["training_contract"]["steps"], 7000)
        self.assertFalse(protocol["method_frozen_from_v8"]["ba_points_enter_GS"])
        ledger = read(ROOT / "reports/results.json")
        record = ledger["records"]["rotation_transfer_v9"]
        self.assertEqual(record["sha256"], sha256(ROOT / record["file_path"]))
        for path, expected in record["artifacts"].items():
            self.assertEqual(expected, sha256(ROOT / path))


if __name__ == "__main__":
    unittest.main()
