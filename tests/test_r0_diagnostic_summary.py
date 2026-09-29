"""읽기 전용 요약의 집계·검열 처리 검사. 기존 모델 점검을 재실행하지 않습니다."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_r0_diagnostics import LINEARS, group_errors, summarize


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(folder):
    metric = {"rms": 0.02, "max_abs": 0.04, "relative_rms": 0.01}
    for layer in range(28):
        write_json(folder / f"awq_layer_{layer:03d}.json", {
            "layer": layer, "quantized_linears": list(LINEARS),
            "scale_only_output_difference": metric, "canonicalized_output_difference": metric,
            "canonical_weight_error": [
                {"parameter": n + ".weight", "elements": 16, "rms_error": 0.1,
                 "relative_rms_error": 0.2} for n in LINEARS],
            "canonical_auxiliary_error": [
                {"parameter": n + ".weight", "elements": 4, "rms_error": 0.0}
                for n in ("input_layernorm", "post_attention_layernorm")],
        })
    attempts = []
    for arm in ("bf16", "awq_w3"):
        for item in (0, 1):
            attempts.append({"arm": arm, "fixture_id": item, "seed": 42,
                             "input_ids": [1, 2], "input_token_count": 2,
                             "generated_ids": [3] * 128,
                             "generated_token_count_including_eos": 128,
                             "finish_reason": "length", "budget_reached": True,
                             "right_censored": True})
    run = {"status": "R0_EXECUTED_DIAGNOSTICS_PENDING_REVIEW",
           "awq_status": "AWQ_TRANSFORM_APPLIED_NOT_EQUIVALENCE_CERTIFIED",
           "model_ready": False, "completed_awq_layers": 28, "attempts": attempts}
    write_json(folder / "run.json", run)
    return run


class SummaryTest(unittest.TestCase):
    def test_weighted_rms_and_undefined_relative_are_not_silently_averaged(self):
        rows = [{"parameter": "p", "layer": 0, "elements": 1, "rms_error": 4.0,
                 "relative_rms_error": None},
                {"parameter": "p", "layer": 1, "elements": 3, "rms_error": 2.0,
                 "relative_rms_error": 0.5}]
        result = group_errors(rows, relative=True)[0]
        self.assertAlmostEqual(result["element_weighted_rms_error"], math.sqrt(7))
        self.assertEqual(result["undefined_relative_layers"], [0])
        self.assertEqual(result["max_relative_rms_layer"], 1)

    def test_complete_censored_jsons_are_read_without_writes_or_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            report = summarize(folder)
            self.assertEqual(report["structural_issues"], [])
            self.assertEqual(report["status"], "READ_ONLY_R0_SUMMARY_NOT_CERTIFICATION")
            self.assertEqual(len(report["source_files"]), 29)
            self.assertEqual(sum(x["records"] for x in report["canonical_weight_error"]), 196)
            self.assertTrue(all(x["right_censored"] for x in report["generation_id_review"]))
            self.assertNotIn("generated_ids", json.dumps(report))
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_eos_at_budget_is_observed_not_censored(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run = fixture(folder)
            row = run["attempts"][0]
            row["generated_ids"][-1] = 151643
            row.update(finish_reason="eos", right_censored=False)
            write_json(folder / "run.json", run)
            report = summarize(folder)
            self.assertEqual(report["structural_issues"], [])
            result = report["generation_id_review"][0]
            self.assertEqual(result["finish_reason"], "eos")
            self.assertTrue(result["budget_reached"])
            self.assertFalse(result["right_censored"])

    def test_missing_layer_and_token_record_inconsistency_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run = fixture(folder)
            (folder / "awq_layer_027.json").unlink()
            run["attempts"][0]["generated_ids"][2] = 151643
            run["attempts"][1]["right_censored"] = False
            run["attempts"][3]["input_ids"] = [1, 3]
            write_json(folder / "run.json", run)
            report = summarize(folder)
            self.assertEqual(report["layer_coverage"]["missing"], [27])
            self.assertEqual(len(report["structural_issues"]), 4)

    def test_duplicate_layer_and_nonfinite_metrics_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            row = json.loads((folder / "awq_layer_000.json").read_text())
            write_json(folder / "awq_layer_copy.json", row)
            report = summarize(folder)
            self.assertEqual(report["layer_coverage"]["duplicates"], [0])
            row["canonical_auxiliary_error"][0]["rms_error"] = float("inf")
            write_json(folder / "awq_layer_000.json", row)
            with self.assertRaises(ValueError):
                summarize(folder)

    def test_cli_returns_nonzero_when_existing_results_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/summarize_r0_diagnostics.py"),
                "--result-dir", tmp], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("기존 R0 JSON 읽기 중단", result.stderr)


if __name__ == "__main__":
    unittest.main()
