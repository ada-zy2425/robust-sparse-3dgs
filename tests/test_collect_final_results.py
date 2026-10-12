import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("collect_final_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FinalCollectionTests(unittest.TestCase):
    def test_registration_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            logs = run / "logs"
            logs.mkdir()
            (logs / "initialize.01.log").write_text(
                "setup\nBLOCKED: Registration coverage 20/24; missing={'a.jpg'}\n"
            )
            self.assertEqual(
                MODULE.registration_failure(run),
                (20, 24, "BLOCKED: Registration coverage 20/24; missing={'a.jpg'}"),
            )

    def test_missing_failure_log_is_explicitly_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                MODULE.registration_failure(Path(directory)), (None, None, None)
            )


if __name__ == "__main__":
    unittest.main()
