"""CPU-only tests of diagnostics, not tests of model inference or quantization."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/preflight.py"
SPEC = importlib.util.spec_from_file_location("preflight", PATH)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def test_parse_two_gpus(self):
        rows = preflight.parse_gpu_csv("0, NVIDIA GeForce RTX 4090, 24564, 560.00\n1, NVIDIA GeForce RTX 4090, 24564, 560.00\n")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["index"], 1)
        self.assertEqual(rows[0]["memory_total_mib"], 24564)

    def test_malformed_csv(self):
        with self.assertRaises(ValueError):
            preflight.parse_gpu_csv("0, missing columns")

    def test_skipped_probe_not_ready(self):
        ready, problems = preflight.assess_torch({"status": "skipped"}, 2)
        self.assertFalse(ready)
        self.assertIn("torch_cuda_unavailable", problems)

    def test_insufficient_devices(self):
        ready, _ = preflight.assess_torch({"status": "ok", "cuda_available": True, "devices": [{}]}, 2)
        self.assertFalse(ready)

    def test_two_devices_ready(self):
        ready, problems = preflight.assess_torch({"status": "ok", "cuda_available": True, "devices": [{}, {}]}, 2)
        self.assertTrue(ready)
        self.assertEqual(problems, [])

    def test_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            preflight.write_report(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                preflight.write_report(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text())["a"], 1)
            preflight.write_report(path, {"a": 2}, overwrite=True)
            self.assertEqual(json.loads(path.read_text())["a"], 2)

    def test_missing_command(self):
        with patch.object(preflight.subprocess, "run", side_effect=FileNotFoundError):
            result = preflight.run_command(["missing"])
        self.assertEqual(result["error"], "command_not_found")


if __name__ == "__main__":
    unittest.main()
